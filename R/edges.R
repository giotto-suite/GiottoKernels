#' Per-feature sums over graph edges from a stream of triplets
#'
#' For each feature `g`, computes `sum_e w_e * x[i_e, g] * x[j_e, g]`, where
#' `x` is the `n_rows x n_features` sparse matrix encoded by `(row_id, col_id,
#' value)` triplets in an Arrow stream (row = cell, col = feature) and edge
#' `e` joins rows `from[e]` and `to[e]`. List each edge once. This is the
#' cross-cell term of Moran's I, Geary's C and join counts.
#'
#' The stream is read into memory once, 16 bytes per stored value, so a caller that must bound memory streams a subset of the
#' features per call and combines the results; the sums of different features
#' are independent.
#'
#' @section Layout the stream must have:
#' None in particular: batches and rows may arrive in any order, and features
#' in any order within a row. Each `(row_id, col_id)` pair must appear at most
#' once; a repeat is an error.
#'
#' Edges are cut into a fixed number of blocks summed in block order, so the
#' result is bit-identical at any thread count.
#'
#' @param stream anything [nanoarrow::as_nanoarrow_array_stream()] accepts, with
#'   columns `row_id` (int32, 1-based, at most `n_rows`), `col_id` (int32,
#'   1-based, at most `n_features`) and `value` (float64), without nulls. Other
#'   columns are ignored. The stream is consumed.
#' @param from,to integer vectors of row ids (1-based), one entry per edge.
#' @param n_rows number of rows (cells).
#' @param n_features number of features; the result has one value each.
#' @param w optional numeric edge weights, one per edge; `NULL` for 1.
#' @param n_threads worker threads, resolved by [kernel_threads()].
#' @returns a numeric vector of length `n_features`.
#' @examples
#' tri <- data.frame(
#'     row_id = c(1L, 1L, 2L, 3L),
#'     col_id = c(1L, 2L, 1L, 2L),
#'     value  = c(1, 2, 3, 4)
#' )
#' edge_products_stream(tri, from = c(1L, 1L), to = c(2L, 3L),
#'     n_rows = 3L, n_features = 2L)
#' @export
edge_products_stream <- function(stream, from, to, n_rows, n_features,
                                 w = NULL, n_threads = NULL) {
    int1 <- function(x, nm) {
        x <- suppressWarnings(as.integer(x))
        if (length(x) != 1L || is.na(x) || x < 1L) {
            stop("[edge_products_stream] `", nm, "` must be a positive integer",
                 call. = FALSE)
        }
        x
    }
    n_rows <- int1(n_rows, "n_rows")
    n_features <- int1(n_features, "n_features")
    from <- as.integer(from)
    to <- as.integer(to)
    if (length(from) != length(to) || anyNA(from) || anyNA(to)) {
        stop("[edge_products_stream] `from` and `to` must be non-NA and of ",
             "equal length", call. = FALSE)
    }
    if (!is.null(w)) {
        w <- as.double(w)
        if (length(w) != length(from) || anyNA(w)) {
            stop("[edge_products_stream] `w` must be non-NA, one per edge",
                 call. = FALSE)
        }
    }
    n_threads <- kernel_threads(n_threads)
    stream <- nanoarrow::as_nanoarrow_array_stream(stream)
    edge_products_stream_rs(stream, from, to, w, n_rows, n_features, n_threads)
}
