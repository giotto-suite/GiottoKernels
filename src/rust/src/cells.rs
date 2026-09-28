//! Shared driver for kernels that need each cell's entries together.
//!
//! A stream of `(row_id, col_id, value)` triplets (row = cell, 1-based) is cut
//! into batches wherever the producer chose. Each batch is split: its first
//! and last cell, which a boundary may have cut, go to a pool resolved after
//! the stream ends; every cell strictly inside is complete and goes straight
//! to a worker. That is exact whatever the batch sizes, as long as no cell
//! interior to one batch appears in any other batch. Two checks enforce it:
//! each batch sorted by `row_id`, and batch `row_id` ranges overlapping at most
//! at one shared edge cell. A stream sorted by `row_id` passes both; so does
//! one grouped by cell whose splits fall at batch edges, in any batch order.
//!
//! Batch `k` always goes to worker `k % n_threads`, so a fixed thread count
//! assigns work identically on every run.

use std::sync::mpsc;
use std::thread;

use crate::arrow_c::{Column, Field, Stream};

pub struct Triplets {
    pub r: Vec<i32>,
    pub c: Vec<i32>,
    pub v: Vec<f64>,
}

impl Triplets {
    pub fn new() -> Self {
        Triplets { r: Vec::new(), c: Vec::new(), v: Vec::new() }
    }
    fn extend(&mut self, from: &Triplets, range: std::ops::Range<usize>) {
        self.r.extend_from_slice(&from.r[range.clone()]);
        self.c.extend_from_slice(&from.c[range.clone()]);
        self.v.extend_from_slice(&from.v[range]);
    }
    /// Consecutive runs of equal `r`: one cell each.
    pub fn cells(&self) -> CellRuns<'_> {
        CellRuns { t: self, at: 0 }
    }
}

pub struct CellRuns<'a> {
    t: &'a Triplets,
    at: usize,
}

impl<'a> Iterator for CellRuns<'a> {
    type Item = std::ops::Range<usize>;
    fn next(&mut self) -> Option<Self::Item> {
        let n = self.t.r.len();
        if self.at >= n {
            return None;
        }
        let a0 = self.at;
        let mut a1 = a0 + 1;
        while a1 < n && self.t.r[a1] == self.t.r[a0] {
            a1 += 1;
        }
        self.at = a1;
        Some(a0..a1)
    }
}

const FIELDS: [Field; 3] = [
    Field { name: "row_id", format: "i" },
    Field { name: "col_id", format: "i" },
    Field { name: "value", format: "g" },
];

fn into_triplets(cols: Vec<Column>) -> Triplets {
    let mut it = cols.into_iter();
    match (it.next(), it.next(), it.next()) {
        (Some(Column::I32(r)), Some(Column::I32(c)), Some(Column::F64(v))) => Triplets { r, c, v },
        _ => unreachable!("column types are fixed by FIELDS"),
    }
}

/// Run `step` over every complete cell on `n_threads` workers, each owning a
/// state made by `init`. Returns the worker states in worker order and the
/// edge pool grouped by cell (stable, so each cell's pieces keep their
/// arrival order).
///
/// `n_cols` bounds `col_id` to `1..=n_cols`; `max_row`, when given, bounds
/// `row_id` to `1..=max_row`.
pub fn run<S, I, F>(
    stream: &Stream,
    n_cols: usize,
    max_row: Option<usize>,
    n_threads: usize,
    init: I,
    step: F,
) -> Result<(Vec<S>, Triplets), String>
where
    S: Send,
    I: Fn() -> S + Sync,
    F: Fn(&mut S, &Triplets) + Sync,
{
    let nt = n_threads.max(1);
    let cols = stream.resolve(&FIELDS)?;
    let mut pool = Triplets::new();
    let mut ranges: Vec<(i32, i32)> = Vec::new();
    let init = &init;
    let step = &step;

    let (states, read_result) = thread::scope(|sc| {
        let mut senders = Vec::with_capacity(nt);
        let mut workers = Vec::with_capacity(nt);
        for _ in 0..nt {
            let (tx, rx) = mpsc::sync_channel::<Triplets>(2);
            senders.push(tx);
            workers.push(sc.spawn(move || {
                let mut state = init();
                for t in rx {
                    step(&mut state, &t);
                }
                state
            }));
        }

        let read_result = (|| -> Result<(), String> {
            let mut k = 0usize;
            while let Some((n, batch)) = stream.next_batch(&cols)? {
                if n == 0 {
                    continue;
                }
                let t = into_triplets(batch);
                if t.c.iter().any(|&c| c < 1 || c as usize > n_cols) {
                    return Err(format!("col_id outside 1..{}", n_cols));
                }
                if let Some(m) = max_row {
                    if t.r.iter().any(|&r| r < 1 || r as usize > m) {
                        return Err(format!("row_id outside 1..{}", m));
                    }
                }
                if t.r.windows(2).any(|w| w[1] < w[0]) {
                    return Err("a batch is not sorted by row_id".into());
                }
                let (first, last) = (t.r[0], t.r[n - 1]);
                ranges.push((first, last));
                // [0, lo) is the first cell, [hi, n) the last; both may be cut
                let lo = t.r.iter().position(|&x| x != first).unwrap_or(n);
                let hi = t.r.iter().rposition(|&x| x != last).map_or(lo, |q| q + 1);
                if hi > lo {
                    pool.extend(&t, 0..lo);
                    pool.extend(&t, hi..n);
                    let mut inner = Triplets::new();
                    inner.extend(&t, lo..hi);
                    if senders[k % nt].send(inner).is_err() {
                        return Err("a worker thread stopped unexpectedly".into());
                    }
                    k += 1;
                } else {
                    pool.extend(&t, 0..n); // one or two cells: all edge
                }
            }
            Ok(())
        })();

        drop(senders);
        let states: Vec<S> =
            workers.into_iter().map(|w| w.join().expect("kernel worker panicked")).collect();
        (states, read_result)
    });
    read_result?;

    // Batch ranges may share at most one edge cell. More overlap means some
    // cell was interior to one batch (already handed to a worker) and also
    // appears in another, whose part would be combined without the rest.
    ranges.sort_unstable();
    if ranges.windows(2).any(|w| w[1].0 < w[0].1) {
        return Err("batches overlap in row_id: the stream is not grouped by cell".into());
    }

    let mut order: Vec<usize> = (0..pool.r.len()).collect();
    order.sort_by_key(|&k| pool.r[k]);
    let edge = Triplets {
        r: order.iter().map(|&k| pool.r[k]).collect(),
        c: order.iter().map(|&k| pool.c[k]).collect(),
        v: order.iter().map(|&k| pool.v[k]).collect(),
    };
    Ok((states, edge))
}
