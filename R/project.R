#' Sparse x dense product from a stream of triplets
#'
#' Computes `A %*% V`, where `A` is the `n_rows x nrow(V)` sparse matrix encoded
#' by `(row_id, col_id, value)` triplets in an Arrow stream (row = cell, col =
#' feature) and `V` is a dense matrix, one column per component. Only stored
#' entries are touched, so the cost is `nnz * ncol(V)`, and memory is one batch
#' plus the `n_rows x ncol(V)` result.
#'
#' This is the second pass of Gram-eigen PCA: cell coordinates from the
#' loadings. Centering is left to the caller, since it is one subtraction per
#' component: `(A - 1 mu^T) V = A V - 1 (mu^T V)`.
#'
#' @section Layout the stream must have:
#' The same as [gram_stream()]: batch boundaries may fall anywhere, but no cell
#' interior to one batch may also appear in another, which a stream sorted by
#' `row_id` always satisfies. The call checks this and errors rather than
#' return a wrong result. Each `(row_id, col_id)` pair must appear at most once.
#'
#' Every output row is built by one thread from its entries in stream order,
#' so the result is bit-identical at any thread count.
#'
#' @param stream anything [nanoarrow::as_nanoarrow_array_stream()] accepts, with
#'   columns `row_id` (int32, 1-based, at most `n_rows`), `col_id` (int32,
#'   1-based row of `V`) and `value` (float64), without nulls. Other columns
#'   are ignored. The stream is consumed.
#' @param V numeric matrix, one row per feature.
#' @param n_rows number of rows in the result; rows with no stored values are 0.
#' @param n_threads worker threads, resolved by [kernel_threads()]: `NULL`
#'   (default) uses `options(gkernels.n_threads)`, else 1.
#' @returns the `n_rows x ncol(V)` matrix `A %*% V`.
#' @examples
#' tri <- data.frame(
#'     row_id = c(1L, 1L, 3L),
#'     col_id = c(1L, 2L, 2L),
#'     value  = c(1, 2, 3)
#' )
#' project_stream(tri, V = diag(2), n_rows = 3L)
#' @export
project_stream <- function(stream, V, n_rows, n_threads = NULL) {
    if (!is.matrix(V) || !is.numeric(V) || nrow(V) < 1L || ncol(V) < 1L) {
        stop("[project_stream] `V` must be a non-empty numeric matrix", call. = FALSE)
    }
    n_rows <- suppressWarnings(as.integer(n_rows))
    if (length(n_rows) != 1L || is.na(n_rows) || n_rows < 1L) {
        stop("[project_stream] `n_rows` must be a positive integer", call. = FALSE)
    }
    n_threads <- kernel_threads(n_threads)
    storage.mode(V) <- "double"
    stream <- nanoarrow::as_nanoarrow_array_stream(stream)
    out <- project_stream_rs(stream, V, n_rows, n_threads)
    dimnames(out) <- list(NULL, colnames(V))
    out
}
