#' Gram matrix and feature sums from a stream of triplets
#'
#' Computes `G = sum_c x_c x_c^T` and `s = sum_c x_c` over cells `c`, where
#' `x_c` is cell `c`'s feature vector, reading sparse `(row_id, col_id, value)`
#' triplets from an Arrow stream: row = cell, col = feature. Only nonzero pairs
#' within a cell are multiplied, so the cost is the sum over cells of
#' `nnz_c^2` rather than `n_cells * n_features^2`, and memory stays at one
#' batch plus the `n_features x n_features` result.
#'
#' This is the first pass of Gram-eigen PCA: centering, scaling and the
#' eigendecomposition follow from `G`, `s` and the cell count.
#'
#' @section Layout the stream must have:
#' Batch boundaries may fall anywhere, including inside a cell. What is
#' required is that no cell interior to one batch also appears in another.
#' A stream sorted by `row_id` always satisfies this. Two checks enforce it,
#' and the call errors rather than returning a wrong result if either fails:
#' each batch must be sorted by `row_id`, and batches' `row_id` ranges may
#' overlap only at one shared edge cell.
#'
#' Features within a cell need not be sorted, but each `(row_id, col_id)` pair
#' must appear at most once, as in any sparse store: a repeated pair is not
#' summed first, so its diagonal term would be wrong.
#'
#' @param stream anything [nanoarrow::as_nanoarrow_array_stream()] accepts: an
#'   `arrow` `RecordBatchReader`, a `nanoarrow_array_stream`, or a data frame.
#'   It must have columns `row_id` (int32), `col_id` (int32, 1-based feature
#'   index) and `value` (float64), without nulls; other columns are ignored.
#'   The stream is consumed.
#' @param n_features number of features `P`; every `col_id` must be in
#'   `1..n_features`.
#' @param n_threads worker threads; see [kernel_threads()]. A fixed count gives
#'   bit-identical results across runs. Each thread keeps its own
#'   `n_features x n_features` accumulator, so the count is lowered, if
#'   needed, to keep all of them within `options(GiottoKernels.memory_gb)`
#'   (default 4).
#' @returns a list with `G`, the `n_features x n_features` symmetric Gram
#'   matrix, and `s`, the length-`n_features` feature sums.
#' @examples
#' tri <- data.frame(
#'     row_id = c(1L, 1L, 2L),
#'     col_id = c(1L, 2L, 2L),
#'     value  = c(1, 2, 3)
#' )
#' gram_stream(tri, n_features = 2L, n_threads = 1L)
#' @export
gram_stream <- function(stream, n_features, n_threads = kernel_threads()) {
    n_features <- suppressWarnings(as.integer(n_features))
    if (length(n_features) != 1L || is.na(n_features) || n_features < 1L) {
        stop("[gram_stream] `n_features` must be a positive integer", call. = FALSE)
    }
    n_threads <- suppressWarnings(as.integer(n_threads))
    if (length(n_threads) != 1L || is.na(n_threads) || n_threads < 1L) {
        stop("[gram_stream] `n_threads` must be a positive integer", call. = FALSE)
    }
    # One P x P accumulator per worker plus the result.
    n_threads <- min(n_threads, .threads_within_budget(n_features))
    stream <- nanoarrow::as_nanoarrow_array_stream(stream)
    gram_stream_rs(stream, n_features, n_threads)
}

.threads_within_budget <- function(n_features) {
    budget <- getOption("GiottoKernels.memory_gb", 4) * 1e9
    per <- as.numeric(n_features)^2 * 8
    as.integer(max(1, floor(budget / per) - 1))
}
