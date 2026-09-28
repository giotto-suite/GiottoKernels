//! Gram accumulation over a stream of expression triplets.
//!
//! Computes `G = sum_c x_c x_c^T` and `s = sum_c x_c` over cells `c`, where
//! `x_c` is cell `c`'s feature vector, from `(row_id, col_id, value)` triplets
//! (row = cell, col = feature, 1-based). Only nonzero pairs within a cell are
//! touched, so cost is `sum_c nnz_c^2`, not `n_cells * P^2`.
//!
//! A cell's contribution needs all of its entries at once, but batch
//! boundaries fall wherever the producer puts them. So each batch is split:
//! its first and last cell, which a boundary may have cut, go to a pool
//! resolved after the stream ends; every cell strictly inside is complete and
//! is computed straight away. That is exact whatever the batch sizes, as long
//! as no cell that is interior to one batch appears in any other batch. Two
//! checks enforce it: each batch sorted by `row_id`, and batch `row_id` ranges
//! overlapping at most at one shared edge cell. A stream sorted by `row_id`
//! passes both; so does one grouped by cell whose splits fall at batch edges.
//!
//! Reproducibility: batch `k` always goes to worker `k % n_threads` and the
//! partials are summed in worker order, so a fixed thread count gives
//! bit-identical results across runs. Different thread counts group the
//! floating-point sums differently and agree to rounding.

use std::sync::mpsc;
use std::thread;

use crate::arrow_c::{Column, Field, Stream};

pub struct Gram {
    pub g: Vec<f64>, // P x P, column-major, full symmetric
    pub s: Vec<f64>, // P
}

struct Triplets {
    r: Vec<i32>,
    c: Vec<i32>,
    v: Vec<f64>,
}

impl Triplets {
    fn new() -> Self {
        Triplets { r: Vec::new(), c: Vec::new(), v: Vec::new() }
    }
    fn extend(&mut self, from: &Triplets, range: std::ops::Range<usize>) {
        self.r.extend_from_slice(&from.r[range.clone()]);
        self.c.extend_from_slice(&from.c[range.clone()]);
        self.v.extend_from_slice(&from.v[range]);
    }
}

/// Pairwise products within each run of equal `r` (one cell). Features
/// within a cell need not be sorted, so write the upper triangle by min/max.
fn accumulate(t: &Triplets, p: usize, g: &mut [f64], s: &mut [f64]) {
    let n = t.r.len();
    let mut a0 = 0usize;
    while a0 < n {
        let mut a1 = a0 + 1;
        while a1 < n && t.r[a1] == t.r[a0] {
            a1 += 1;
        }
        for a in a0..a1 {
            let ga = (t.c[a] - 1) as usize;
            let xa = t.v[a];
            s[ga] += xa;
            for b in a..a1 {
                let gb = (t.c[b] - 1) as usize;
                let (lo, hi) = if ga <= gb { (ga, gb) } else { (gb, ga) };
                g[lo + hi * p] += xa * t.v[b];
            }
        }
        a0 = a1;
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

pub fn gram_stream(stream: &Stream, p: usize, n_threads: usize) -> Result<Gram, String> {
    let nt = n_threads.max(1);
    let cols = stream.resolve(&FIELDS)?;

    let mut pool = Triplets::new();
    let mut ranges: Vec<(i32, i32)> = Vec::new();

    let (partials, read_result) = thread::scope(|sc| {
        let mut senders = Vec::with_capacity(nt);
        let mut workers = Vec::with_capacity(nt);
        for _ in 0..nt {
            let (tx, rx) = mpsc::sync_channel::<Triplets>(2);
            senders.push(tx);
            workers.push(sc.spawn(move || {
                let mut g = vec![0.0f64; p * p];
                let mut s = vec![0.0f64; p];
                for t in rx {
                    accumulate(&t, p, &mut g, &mut s);
                }
                (g, s)
            }));
        }

        let read_result = (|| -> Result<(), String> {
            let mut k = 0usize;
            while let Some((n, batch)) = stream.next_batch(&cols)? {
                if n == 0 {
                    continue;
                }
                let t = into_triplets(batch);
                if t.c.iter().any(|&c| c < 1 || c as usize > p) {
                    return Err(format!("col_id outside 1..{}", p));
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
        let partials: Vec<(Vec<f64>, Vec<f64>)> =
            workers.into_iter().map(|w| w.join().expect("gram worker panicked")).collect();
        (partials, read_result)
    });
    read_result?;

    // Batch ranges may share at most one edge cell. More overlap means some
    // cell was interior to one batch (already computed) and also appears in
    // another, whose part would be summed without its cross terms.
    ranges.sort_unstable();
    if ranges.windows(2).any(|w| w[1].0 < w[0].1) {
        return Err("batches overlap in row_id: the stream is not grouped by cell".into());
    }

    // Pool: group fragments by cell (stable, so each cell's pieces keep their
    // arrival order), then the same kernel.
    let mut order: Vec<usize> = (0..pool.r.len()).collect();
    order.sort_by_key(|&k| pool.r[k]);
    let edge = Triplets {
        r: order.iter().map(|&k| pool.r[k]).collect(),
        c: order.iter().map(|&k| pool.c[k]).collect(),
        v: order.iter().map(|&k| pool.v[k]).collect(),
    };

    let mut g = vec![0.0f64; p * p];
    let mut s = vec![0.0f64; p];
    for (gl, sl) in &partials {
        for (a, b) in g.iter_mut().zip(gl) {
            *a += b;
        }
        for (a, b) in s.iter_mut().zip(sl) {
            *a += b;
        }
    }
    accumulate(&edge, p, &mut g, &mut s);

    for col in 0..p {
        for row in 0..col {
            g[col + row * p] = g[row + col * p];
        }
    }
    Ok(Gram { g, s })
}
