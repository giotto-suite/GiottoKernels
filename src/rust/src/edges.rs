//! Per-feature sums over the edges of a graph: for each feature `g`,
//! `T[g] = sum_e w_e * x[i_e, g] * x[j_e, g]`, where `x` is the sparse
//! `n_rows x n_features` matrix a stream of `(row_id, col_id, value)` triplets
//! encodes (row = cell) and edge `e` joins rows `i_e` and `j_e`.
//!
//! The stream is read once into a row-compressed table (each row's features
//! sorted), so neither batch order nor row order matters: entries are packed
//! as `(row << 32 | col, value)`, 16 bytes each, and sorted in place --
//! partitioned into row-range buckets, which are sorted in parallel -- unless
//! the stream already came sorted. 16 bytes a stored value is both the peak
//! and what is held while the edges are summed. Each edge then
//! walks its two rows' feature lists together, touching only features both
//! rows store.
//!
//! Edges are cut into a fixed number of blocks, independent of the thread
//! count; each block sums into its own accumulator and the blocks are added in
//! block order, so the result is bit-identical at any thread count.

use crate::arrow_c::{Column, Field, Stream};

const FIELDS: [Field; 3] = [
    Field { name: "row_id", format: "i" },
    Field { name: "col_id", format: "i" },
    Field { name: "value", format: "g" },
];

/// Upper bound on edge blocks: enough to balance threads, few enough that the
/// per-block accumulators (`n_features` doubles each) stay small.
const MAX_BLOCKS: usize = 64;

/// Row-compressed sparse matrix: row `r`'s entries are `ent[off[r]..off[r +
/// 1]]`, each `(row << 32 | col, value)`, features ascending.
struct Csr {
    off: Vec<usize>,
    ent: Vec<(u64, f64)>,
}

/// Buckets for the parallel sort: enough for the threads, few enough that
/// partitioning into them writes to a handful of cursors.
const SORT_BUCKETS: usize = 64;

fn read_csr(stream: &Stream, n_rows: usize, n_features: usize, n_threads: usize)
            -> Result<Csr, String> {
    let cols = stream.resolve(&FIELDS)?;
    let mut ent: Vec<(u64, f64)> = Vec::new();
    while let Some((n, mut batch)) = stream.next_batch(&cols)? {
        if n == 0 {
            continue;
        }
        let v = match batch.pop() { Some(Column::F64(v)) => v, _ => unreachable!() };
        let c = match batch.pop() { Some(Column::I32(c)) => c, _ => unreachable!() };
        let r = match batch.pop() { Some(Column::I32(r)) => r, _ => unreachable!() };
        ent.reserve(n);
        for k in 0..n {
            let (ri, ci) = (r[k], c[k]);
            if ri < 1 || ri as usize > n_rows {
                return Err(format!("row_id {ri} outside 1..{n_rows}"));
            }
            if ci < 1 || ci as usize > n_features {
                return Err(format!("col_id {ci} outside 1..{n_features}"));
            }
            ent.push(((((ri - 1) as u64) << 32) | (ci - 1) as u64, v[k]));
        }
    }
    ent.shrink_to_fit();

    // Sort by (row, col) in place, unless the stream came sorted. Partition
    // into row-range buckets first (American flag over a few buckets, so the
    // writes go to a handful of cursors), then sort the buckets in parallel.
    if ent.windows(2).any(|w| w[0].0 >= w[1].0) {
        let nb = SORT_BUCKETS.min(n_rows.max(1));
        let bucket = |key: u64| ((key >> 32) as usize * nb) / n_rows.max(1);
        let mut start = vec![0usize; nb + 1];
        for e in &ent {
            start[bucket(e.0) + 1] += 1;
        }
        for i in 0..nb {
            start[i + 1] += start[i];
        }
        let mut next = start[..nb].to_vec();
        for b in 0..nb {
            while next[b] < start[b + 1] {
                let i = next[b];
                let t = bucket(ent[i].0);
                if t == b {
                    next[b] += 1;
                } else {
                    ent.swap(i, next[t]);
                    next[t] += 1;
                }
            }
        }
        let mut slices: Vec<&mut [(u64, f64)]> = Vec::with_capacity(nb);
        let mut rest: &mut [(u64, f64)] = &mut ent;
        for b in 0..nb {
            let (head, tail) = rest.split_at_mut(start[b + 1] - start[b]);
            slices.push(head);
            rest = tail;
        }
        let n_threads = n_threads.max(1).min(nb);
        std::thread::scope(|s| {
            let mut groups: Vec<Vec<&mut [(u64, f64)]>> = (0..n_threads).map(|_| Vec::new()).collect();
            for (k, sl) in slices.into_iter().enumerate() {
                groups[k % n_threads].push(sl);
            }
            for g in groups {
                s.spawn(move || {
                    for sl in g {
                        sl.sort_unstable_by_key(|e| e.0);
                    }
                });
            }
        });
    }
    if ent.windows(2).any(|w| w[0].0 == w[1].0) {
        return Err("a (row_id, col_id) pair appears more than once".into());
    }

    let mut off = vec![0usize; n_rows + 1];
    for e in &ent {
        off[(e.0 >> 32) as usize + 1] += 1;
    }
    for i in 0..n_rows {
        off[i + 1] += off[i];
    }
    Ok(Csr { off, ent })
}

/// Sum over edges `range` into `acc`.
fn block_sums(m: &Csr, from: &[i32], to: &[i32], w: Option<&[f64]>,
              range: std::ops::Range<usize>, acc: &mut [f64]) {
    for e in range {
        let (i, j) = ((from[e] - 1) as usize, (to[e] - 1) as usize);
        let we = w.map_or(1.0, |w| w[e]);
        let (mut a, ae) = (m.off[i], m.off[i + 1]);
        let (mut b, be) = (m.off[j], m.off[j + 1]);
        while a < ae && b < be {
            let (ca, cb) = (m.ent[a].0 as u32, m.ent[b].0 as u32);
            if ca == cb {
                acc[ca as usize] += we * m.ent[a].1 * m.ent[b].1;
                a += 1;
                b += 1;
            } else if ca < cb {
                a += 1;
            } else {
                b += 1;
            }
        }
    }
}

pub fn edge_products_stream(
    stream: &Stream,
    from: &[i32],
    to: &[i32],
    w: Option<&[f64]>,
    n_rows: usize,
    n_features: usize,
    n_threads: usize,
) -> Result<Vec<f64>, String> {
    if from.len() != to.len() || w.map_or(false, |w| w.len() != from.len()) {
        return Err("`from`, `to` and `w` must have one entry per edge".into());
    }
    for (&i, &j) in from.iter().zip(to) {
        if i < 1 || j < 1 || i as usize > n_rows || j as usize > n_rows {
            return Err(format!("edge end outside 1..{n_rows}"));
        }
    }
    let m = read_csr(stream, n_rows, n_features, n_threads)?;

    let n_edges = from.len();
    let n_blocks = MAX_BLOCKS.min(n_edges.max(1));
    let size = (n_edges + n_blocks - 1) / n_blocks.max(1);
    let ranges: Vec<std::ops::Range<usize>> = (0..n_blocks)
        .map(|k| (k * size).min(n_edges)..((k + 1) * size).min(n_edges))
        .collect();
    let mut accs: Vec<Vec<f64>> = vec![Vec::new(); n_blocks];

    let n_threads = n_threads.max(1).min(n_blocks);
    std::thread::scope(|s| {
        let m = &m;
        let ranges = &ranges;
        // thread t takes blocks t, t + n_threads, ...; each block's sum is
        // fixed by its edges alone, whatever thread computes it
        let handles: Vec<_> = (0..n_threads)
            .map(|t| {
                s.spawn(move || {
                    let mut out = Vec::new();
                    let mut k = t;
                    while k < ranges.len() {
                        let mut acc = vec![0f64; n_features];
                        block_sums(m, from, to, w, ranges[k].clone(), &mut acc);
                        out.push((k, acc));
                        k += n_threads;
                    }
                    out
                })
            })
            .collect();
        for h in handles {
            for (k, acc) in h.join().expect("edge_products worker panicked") {
                accs[k] = acc;
            }
        }
    });

    let mut total = vec![0f64; n_features];
    for acc in &accs {
        for (t, a) in total.iter_mut().zip(acc) {
            *t += a;
        }
    }
    Ok(total)
}
