use extendr_api::prelude::*;

mod arrow_c;
mod cells;
mod edges;
mod gram;
mod project;

use arrow_c::{ArrowArrayStream, Stream};

/// Kernels compiled into this build. `has_kernel()` reads this, so a caller
/// on an older install falls back instead of calling a kernel it lacks.
/// @noRd
#[extendr]
fn kernels_rs() -> Strings {
    Strings::from_values(["gram_stream", "project_stream", "edge_products_stream"])
}

/// Gram pass over a nanoarrow_array_stream of (row_id, col_id, value).
/// Argument checks live in the R wrapper `gram_stream()`.
/// @noRd
#[extendr]
fn gram_stream_rs(stream: Robj, n_features: i32, n_threads: i32) -> extendr_api::Result<List> {
    let st = stream_arg(&stream, "gram_stream")?;
    let p = n_features as usize;
    let out = gram::gram_stream(&st, p, n_threads as usize)
        .map_err(|e| Error::Other(format!("[gram_stream] {e}")))?;

    let g = RMatrix::<f64>::new_matrix(p, p, |r, c| out.g[r + c * p]);
    Ok(list!(G = g, s = out.s))
}

/// `A %*% V` over a nanoarrow_array_stream of (row_id, col_id, value).
/// Argument checks live in the R wrapper `project_stream()`.
/// @noRd
#[extendr]
fn project_stream_rs(stream: Robj, v: RMatrix<f64>, n_rows: i32, n_threads: i32) -> extendr_api::Result<RMatrix<f64>> {
    let st = stream_arg(&stream, "project_stream")?;
    let (p, k, n) = (v.nrows(), v.ncols(), n_rows as usize);
    let out = project::project_stream(&st, v.data(), p, k, n, n_threads as usize)
        .map_err(|e| Error::Other(format!("[project_stream] {e}")))?;
    Ok(RMatrix::<f64>::new_matrix(n, k, |r, c| out[r + c * n]))
}

/// Per-feature sums over graph edges from a nanoarrow_array_stream of
/// (row_id, col_id, value). Argument checks live in the R wrapper
/// `edge_products_stream()`; `w` is NULL for unit weights.
/// @noRd
#[extendr]
fn edge_products_stream_rs(stream: Robj, from: Robj, to: Robj, w: Robj,
                           n_rows: i32, n_features: i32, n_threads: i32)
                           -> extendr_api::Result<Doubles> {
    let st = stream_arg(&stream, "edge_products_stream")?;
    let int_slice = |x: &Robj, nm: &str| x.as_integer_slice().ok_or_else(||
        Error::Other(format!("[edge_products_stream] `{nm}` must be integer")));
    let from = int_slice(&from, "from")?;
    let to = int_slice(&to, "to")?;
    let wv: Option<Vec<f64>> = if w.is_null() {
        None
    } else {
        Some(w.as_real_slice()
            .ok_or_else(|| Error::Other("[edge_products_stream] `w` must be double".into()))?
            .to_vec())
    };
    let out = edges::edge_products_stream(&st, from, to, wv.as_deref(),
        n_rows as usize, n_features as usize, n_threads as usize)
        .map_err(|e| Error::Other(format!("[edge_products_stream] {e}")))?;
    Ok(Doubles::from_values(out))
}

fn stream_arg(stream: &Robj, site: &str) -> extendr_api::Result<Stream> {
    if !stream.inherits("nanoarrow_array_stream") || !stream.is_external_pointer() {
        return Err(Error::Other(format!("[{site}] `stream` must be a nanoarrow_array_stream")));
    }
    let ptr = unsafe { stream.external_ptr_addr::<ArrowArrayStream>() };
    unsafe { Stream::from_ptr(ptr) }.map_err(|e| Error::Other(format!("[{site}] {e}")))
}

extendr_module! {
    mod GiottoKernels;
    fn kernels_rs;
    fn gram_stream_rs;
    fn project_stream_rs;
    fn edge_products_stream_rs;
}
