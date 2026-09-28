//! Sparse x dense product over a stream of triplets: `out = A %*% V`, where
//! `A` is the `n_rows x P` sparse matrix the stream encodes (row = cell) and
//! `V` is a dense `P x k` matrix. Only stored entries are touched: cost is
//! `nnz * k`.
//!
//! Each output row is a sum over one cell's entries. `cells::run` hands every
//! complete cell to exactly one worker, so every row is built by one thread
//! from its entries in stream order, and cut cells are finished from the edge
//! pool by one thread after the stream ends. Nothing is summed across threads,
//! so the result is bit-identical at any thread count.

use crate::arrow_c::Stream;
use crate::cells::{self, Triplets};

/// Rows computed by one worker: row ids (0-based) and their `k` values each.
struct Rows {
    id: Vec<usize>,
    val: Vec<f64>,
}

/// `v_rm` is `V` row-major (`P x k`), so a stored value's contribution is one
/// contiguous run of `k` multiply-adds.
fn project_cells(t: &Triplets, v_rm: &[f64], k: usize, out: &mut Rows) {
    for cell in t.cells() {
        let base = out.val.len();
        out.val.resize(base + k, 0.0);
        let acc = &mut out.val[base..base + k];
        for a in cell.clone() {
            let x = t.v[a];
            let vr = &v_rm[(t.c[a] - 1) as usize * k..][..k];
            for (o, w) in acc.iter_mut().zip(vr) {
                *o += x * w;
            }
        }
        out.id.push((t.r[cell.start] - 1) as usize);
    }
}

/// `v` is column-major `P x k` (as R stores it). Returns `n_rows x k`,
/// column-major.
pub fn project_stream(
    stream: &Stream,
    v: &[f64],
    p: usize,
    k: usize,
    n_rows: usize,
    n_threads: usize,
) -> Result<Vec<f64>, String> {
    let mut v_rm = vec![0.0f64; p * k];
    for j in 0..k {
        for i in 0..p {
            v_rm[i * k + j] = v[i + j * p];
        }
    }
    let v_rm = &v_rm;

    let (parts, edge) = cells::run(
        stream,
        p,
        Some(n_rows),
        n_threads,
        || Rows { id: Vec::new(), val: Vec::new() },
        |st: &mut Rows, t: &Triplets| project_cells(t, v_rm, k, st),
    )?;
    let mut last = Rows { id: Vec::new(), val: Vec::new() };
    project_cells(&edge, v_rm, k, &mut last);

    let mut out = vec![0.0f64; n_rows * k];
    for part in parts.iter().chain(std::iter::once(&last)) {
        for (q, &row) in part.id.iter().enumerate() {
            let src = &part.val[q * k..(q + 1) * k];
            for j in 0..k {
                out[row + j * n_rows] = src[j];
            }
        }
    }
    Ok(out)
}
