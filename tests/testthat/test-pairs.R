ref_pairs <- function(ed, node_row, weighted, dedupe) {
    ra <- node_row[ed$from_id]; rb <- node_row[ed$to_id]
    ok <- !is.na(ra) & !is.na(rb) & ra > 0 & rb > 0
    d <- data.frame(a = pmin(ra, rb)[ok], b = pmax(ra, rb)[ok],
                    w = if (weighted) ed$weight[ok] else 1)
    if (dedupe) d <- stats::aggregate(w ~ a + b, data = d, FUN = mean)
    d[order(d$a, d$b), ]
}
set.seed(11L)
ed <- data.frame(from_id = sample.int(50L, 300L, TRUE), to_id = sample.int(50L, 300L, TRUE),
                 weight = round(stats::runif(300L, 0.5, 2), 3))
node_row <- sample(c(seq_len(45L), rep(0L, 5L)))

test_that("pairs match the R reference, deduplicated or not", {
    for (wt in c(FALSE, TRUE)) {
        got <- as.data.frame(graph_pairs_stream(ed, node_row, weighted = wt))
        want <- ref_pairs(ed, node_row, wt, TRUE)
        expect_equal(got$a, want$a)
        expect_equal(got$b, want$b)
        expect_equal(got$w, want$w)
    }
    raw <- as.data.frame(graph_pairs_stream(ed, node_row, TRUE, dedupe = FALSE))
    want <- ref_pairs(ed, node_row, TRUE, FALSE)
    expect_equal(nrow(raw), nrow(want))
    expect_equal(sort(paste(raw$a, raw$b, raw$w)), sort(paste(want$a, want$b, want$w)))
})

test_that("int64 ids and float32 weights are read, out-of-range ids drop", {
    ed64 <- nanoarrow::as_nanoarrow_array(ed, schema = nanoarrow::na_struct(list(
        from_id = nanoarrow::na_int64(), to_id = nanoarrow::na_int64(),
        weight = nanoarrow::na_float())))
    got <- graph_pairs_stream(ed64, node_row, weighted = TRUE)
    want <- ref_pairs(ed, node_row, TRUE, TRUE)
    expect_equal(got$a, want$a)
    expect_equal(got$w, want$w, tolerance = 1e-6)
    short <- graph_pairs_stream(ed, node_row[1:10])
    expect_true(all(c(short$a, short$b) %in% node_row[1:10]))
})

test_that("the pairs feed edge_products_stream", {
    expect_true(has_kernel("graph_pairs_stream"))
    tri <- sim_triplets(n_cells = 45L, n_features = 6L)
    p <- graph_pairs_stream(ed, node_row, weighted = TRUE)
    X <- matrix(0, 45L, 6L); X[cbind(tri$row_id, tri$col_id)] <- tri$value
    expect_equal(edge_products_stream(tri, p$a, p$b, 45L, 6L, w = p$w),
                 colSums(X[p$a, ] * X[p$b, ] * p$w), tolerance = 1e-12)
})
