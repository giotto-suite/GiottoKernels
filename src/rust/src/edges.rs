//! Per-feature sums over the edges of a graph: for each feature `g`,
//! `T[g] = sum_e w_e * x[i_e, g] * x[j_e, g]`, where `x` is the sparse
//! `n_rows x n_features` matrix a stream of `(row_id, col_id, value)` triplets
//! encodes (row = cell) and edge `e` joins rows `i_e` and `j_e`.
//!
//! The stream is read once into a row-compressed table (each row's features
//! sorted), so neither batch order nor row order matters. Each edge then
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

/// Row-compressed sparse matrix: row `r`'s entries are
/// `col[off[r]..off[r + 1]]` / `val[..]`, features ascending.
struct Csr {
    off: Vec<usize>,
    col: Vec<u32>,
    val: Vec<f64>,
}

fn read_csr(stream: &Stream, n_rows: usize, n_features: usize) -> Result<Csr, String> {
    let cols = stream.resolve(&FIELDS)?;
    let mut r_all: Vec<u32> = Vec::new();
    let mut c_all: Vec<u32> = Vec::new();
    let mut v_all: Vec<f64> = Vec::new();
    while let Some((n, mut batch)) = stream.next_batch(&cols)? {
        if n == 0 {
            continue;
        }
        let v = match batch.pop() { Some(Column::F64(v)) => v, _ => unreachable!() };
        let c = match batch.pop() { Some(Column::I32(c)) => c, _ => unreachable!() };
        let r = match batch.pop() { Some(Column::I32(r)) => r, _ => unreachable!() };
        for k in 0..n {
            let (ri, ci) = (r[k], c[k]);
            if ri < 1 || ri as usize > n_rows {
                return Err(format!("row_id {ri} outside 1..{n_rows}"));
            }
            if ci < 1 || ci as usize > n_features {
                return Err(format!("col_id {ci} outside 1..{n_features}"));
            }
            r_all.push((ri - 1) as u32);
            c_all.push((ci - 1) as u32);
        }
        v_all.extend_from_slice(&v);
    }

    // counting sort by row
    let mut off = vec![0usize; n_rows + 1];
    for &r in &r_all {
        off[r as usize + 1] += 1;
    }
    for i in 0..n_rows {
        off[i + 1] += off[i];
    }
    let nnz = r_all.len();
    let mut col = vec![0u32; nnz];
    let mut val = vec![0f64; nnz];
    let mut fill = off.clone();
    for k in 0..nnz {
        let r = r_all[k] as usize;
        col[fill[r]] = c_all[k];
        val[fill[r]] = v_all[k];
        fill[r] += 1;
    }
    drop((r_all, c_all, v_all, fill));

    // features ascending within each row; a repeated (row, col) is an error
    for r in 0..n_rows {
        let (a, b) = (off[r], off[r + 1]);
        if b - a < 2 {
            continue;
        }
        if col[a..b].windows(2).any(|w| w[0] >= w[1]) {
            let mut idx: Vec<usize> = (a..b).collect();
            idx.sort_by_key(|&k| col[k]);
            let c2: Vec<u32> = idx.iter().map(|&k| col[k]).collect();
            let v2: Vec<f64> = idx.iter().map(|&k| val[k]).collect();
            if c2.windows(2).any(|w| w[0] == w[1]) {
                return Err(format!("row {} stores a col_id more than once", r + 1));
            }
            col[a..b].copy_from_slice(&c2);
            val[a..b].copy_from_slice(&v2);
        }
    }
    Ok(Csr { off, col, val })
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
            let (ca, cb) = (m.col[a], m.col[b]);
            if ca == cb {
                acc[ca as usize] += we * m.val[a] * m.val[b];
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
    let m = read_csr(stream, n_rows, n_features)?;

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
