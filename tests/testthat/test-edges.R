P <- 12L
tri <- sim_triplets(n_cells = 200L, n_features = P)
N <- max(tri$row_id) + 2L
set.seed(4L)
E <- 600L
ed <- unique(data.frame(from = sample.int(N, E, TRUE), to = sample.int(N, E, TRUE)))
ed <- ed[ed$from != ed$to, ]
wt <- stats::runif(nrow(ed), 0.5, 2)
ref <- function(df, from, to, w = rep(1, length(from))) {
    X <- matrix(0, N, P)
    X[cbind(df$row_id, df$col_id)] <- df$value
    colSums(X[from, , drop = FALSE] * X[to, , drop = FALSE] * w)
}

test_that("matches the dense edge sum, weighted and not", {
    expect_equal(edge_products_stream(tri, ed$from, ed$to, N, P),
                 ref(tri, ed$from, ed$to), tolerance = 1e-12)
    expect_equal(edge_products_stream(tri, ed$from, ed$to, N, P, w = wt),
                 ref(tri, ed$from, ed$to, wt), tolerance = 1e-12)
})

test_that("batch order, row order and thread count do not change the result", {
    one <- edge_products_stream(tri, ed$from, ed$to, N, P, n_threads = 1L)
    set.seed(9L)
    shuffled <- tri[sample.int(nrow(tri)), ]
    for (max_size in c(1L, 5L, 50L)) {
        b <- as_batches(shuffled, random_sizes(nrow(shuffled), max_size))
        for (nt in c(1L, 3L)) {
            expect_identical(
                edge_products_stream(stream_of(b), ed$from, ed$to, N, P, n_threads = nt),
                edge_products_stream(stream_of(as_batches(shuffled, nrow(shuffled))),
                    ed$from, ed$to, N, P, n_threads = 1L))
        }
    }
    expect_equal(edge_products_stream(shuffled, ed$from, ed$to, N, P), one,
                 tolerance = 1e-12)
})

test_that("a subset of features gives those features' sums", {
    sub <- tri[tri$col_id %in% c(2L, 7L), ]
    got <- edge_products_stream(sub, ed$from, ed$to, N, P)
    full <- ref(tri, ed$from, ed$to)
    expect_equal(got[c(2L, 7L)], full[c(2L, 7L)], tolerance = 1e-12)
    expect_true(all(got[-c(2L, 7L)] == 0))
})

test_that("bad input errors instead of mis-summing", {
    expect_error(edge_products_stream(rbind(tri, tri[1, ]), ed$from, ed$to, N, P),
                 "more than once")
    expect_error(edge_products_stream(tri, ed$from, ed$to, N - 5L, P), "outside")
    expect_error(edge_products_stream(tri, 1:2, 1L, N, P), "equal length")
    expect_error(edge_products_stream(tri, ed$from, ed$to, N, P, w = 1), "one per edge")
    expect_true(has_kernel("edge_products_stream"))
})
