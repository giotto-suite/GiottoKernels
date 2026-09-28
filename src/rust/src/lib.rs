use extendr_api::prelude::*;

mod arrow_c;
mod gram;

use arrow_c::{ArrowArrayStream, Stream};

/// Kernels compiled into this build. `has_kernel()` reads this, so a caller
/// on an older install falls back instead of calling a kernel it lacks.
/// @noRd
#[extendr]
fn kernels_rs() -> Strings {
    Strings::from_values(["gram_stream"])
}

/// Gram pass over a nanoarrow_array_stream of (row_id, col_id, value).
/// Argument checks live in the R wrapper `gram_stream()`.
/// @noRd
#[extendr]
fn gram_stream_rs(stream: Robj, n_features: i32, n_threads: i32) -> extendr_api::Result<List> {
    if !stream.inherits("nanoarrow_array_stream") || !stream.is_external_pointer() {
        return Err(Error::Other("[gram_stream] `stream` must be a nanoarrow_array_stream".into()));
    }
    let p = n_features as usize;
    let ptr = unsafe { stream.external_ptr_addr::<ArrowArrayStream>() };
    let st = unsafe { Stream::from_ptr(ptr) }.map_err(|e| Error::Other(format!("[gram_stream] {e}")))?;
    let out = gram::gram_stream(&st, p, n_threads as usize)
        .map_err(|e| Error::Other(format!("[gram_stream] {e}")))?;

    let g = RMatrix::<f64>::new_matrix(p, p, |r, c| out.g[r + c * p]);
    Ok(list!(G = g, s = out.s))
}

extendr_module! {
    mod GiottoKernels;
    fn kernels_rs;
    fn gram_stream_rs;
}
