#' Canonical row pairs of a graph from a stream of edges
#'
#' Maps each edge of an Arrow stream of `(from_id, to_id[, weight])` (node ids)
#' to rows through `node_row`, drops edges with an end that maps to no row, and
#' returns each pair once as `a <= b`. With `dedupe = TRUE`, a pair listed more
#' than once -- as a directed graph lists a mutual pair -- becomes one pair at
#' the mean of its weights. This is the edge list [edge_products_stream()]
#' takes, built without materialising the edges in R.
#'
#' Memory is about 20 bytes per edge while the pairs are deduplicated.
#'
#' @param stream anything [nanoarrow::as_nanoarrow_array_stream()] accepts, with
#'   columns `from_id` and `to_id` (int32, or int64 within the int32 range,
#'   1-based node ids) and, when `weighted`, `weight` (float64 or float32).
#'   Other columns are ignored. The stream is consumed.
#' @param node_row integer vector: `node_row[id]` is the row (1-based) of node
#'   `id`, or 0 / `NA` for a node with no row.
#' @param weighted logical. Read `weight`; otherwise every edge weighs 1.
#' @param dedupe logical. Collapse repeated pairs to one at their mean weight.
#' @returns a list of `a`, `b` (integer rows, `a <= b`) and `w` (double).
#' @examples
#' ed <- data.frame(from_id = c(1L, 2L, 3L), to_id = c(2L, 1L, 9L))
#' graph_pairs_stream(ed, node_row = c(10L, 20L, 30L))
#' @export
graph_pairs_stream <- function(stream, node_row, weighted = FALSE, dedupe = TRUE) {
    node_row <- as.integer(node_row)
    node_row[is.na(node_row)] <- 0L
    stream <- nanoarrow::as_nanoarrow_array_stream(stream)
    graph_pairs_stream_rs(stream, node_row, isTRUE(weighted), isTRUE(dedupe))
}
