//! Gram accumulation over a stream of expression triplets.
//!
//! Computes `G = sum_c x_c x_c^T` and `s = sum_c x_c` over cells `c`, where
//! `x_c` is cell `c`'s feature vector. Only nonzero pairs within a cell are
//! touched, so cost is `sum_c nnz_c^2`, not `n_cells * P^2`. A cell's
//! contribution needs all of its entries at once; `cells::run` guarantees that
//! and states the layout it requires.
//!
//! Reproducibility: each worker sums into its own `P x P` partial and the
//! partials are added in worker order, so a fixed thread count gives
//! bit-identical results across runs. Different thread counts group the
//! floating-point sums differently and agree to rounding.

use crate::arrow_c::Stream;
use crate::cells::{self, Triplets};

pub struct Gram {
    pub g: Vec<f64>, // P x P, column-major, full symmetric
    pub s: Vec<f64>, // P
}

/// Pairwise products within each cell. Features within a cell need not be
/// sorted, so write the upper triangle by min/max.
fn accumulate(t: &Triplets, p: usize, g: &mut [f64], s: &mut [f64]) {
    for cell in t.cells() {
        for a in cell.clone() {
            let ga = (t.c[a] - 1) as usize;
            let xa = t.v[a];
            s[ga] += xa;
            for b in a..cell.end {
                let gb = (t.c[b] - 1) as usize;
                let (lo, hi) = if ga <= gb { (ga, gb) } else { (gb, ga) };
                g[lo + hi * p] += xa * t.v[b];
            }
        }
    }
}

pub fn gram_stream(stream: &Stream, p: usize, n_threads: usize) -> Result<Gram, String> {
    let (partials, edge) = cells::run(
        stream,
        p,
        None,
        n_threads,
        || (vec![0.0f64; p * p], vec![0.0f64; p]),
        |st: &mut (Vec<f64>, Vec<f64>), t: &Triplets| accumulate(t, p, &mut st.0, &mut st.1),
    )?;

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
