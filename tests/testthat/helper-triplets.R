# Sparse triplets, one row per nonzero, sorted by cell.
sim_triplets <- function(n_cells = 300L, n_features = 25L, density = 0.2, seed = 1L) {
    set.seed(seed)
    keep <- matrix(stats::runif(n_cells * n_features) < density, n_cells)
    idx <- which(keep, arr.ind = TRUE)
    idx <- idx[order(idx[, 1L], idx[, 2L]), , drop = FALSE]
    data.frame(
        row_id = as.integer(idx[, 1L]),
        col_id = as.integer(idx[, 2L]),
        value  = stats::rexp(nrow(idx))
    )
}

# Dense reference: G = X^T X, s = colSums(X), X cells x features.
ref_gram <- function(df, n_features) {
    cells <- unique(df$row_id)
    X <- matrix(0, length(cells), n_features)
    X[cbind(match(df$row_id, cells), df$col_id)] <- df$value
    list(G = crossprod(X), s = colSums(X))
}

# Cut `df` into consecutive batches of the given row counts and stream them.
as_batches <- function(df, sizes) {
    ends <- cumsum(sizes)
    stopifnot(utils::tail(ends, 1L) == nrow(df))
    starts <- c(1L, utils::head(ends, -1L) + 1L)
    lapply(seq_along(sizes), function(k) df[starts[k]:ends[k], , drop = FALSE])
}

stream_of <- function(batches) {
    nanoarrow::basic_array_stream(lapply(batches, nanoarrow::as_nanoarrow_array))
}

random_sizes <- function(n, max_size, seed = 2L) {
    set.seed(seed)
    out <- integer()
    while (sum(out) < n) out <- c(out, sample.int(max_size, 1L))
    out[length(out)] <- out[length(out)] - (sum(out) - n)
    out[out > 0L]
}
