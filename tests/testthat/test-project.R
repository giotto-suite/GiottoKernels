P <- 25L
K <- 4L
tri <- sim_triplets(n_features = P)
N <- max(tri$row_id) + 3L # trailing rows with no stored values
set.seed(7L)
V <- matrix(stats::rnorm(P * K), P, K, dimnames = list(NULL, paste0("PC", 1:K)))
ref <- local({
    X <- matrix(0, N, P)
    X[cbind(tri$row_id, tri$col_id)] <- tri$value
    X %*% V
})

test_that("one batch matches the dense product", {
    res <- project_stream(tri, V, N, n_threads = 1L)
    expect_equal(res, ref, tolerance = 1e-12)
    expect_identical(colnames(res), colnames(V))
    expect_identical(unname(res[N, ]), rep(0, K))
})

test_that("cut batches and thread counts give bit-identical results", {
    one <- project_stream(tri, V, N, n_threads = 1L)
    for (max_size in c(1L, 3L, 7L, 40L)) {
        b <- as_batches(tri, random_sizes(nrow(tri), max_size))
        for (nt in c(1L, 2L)) {
            expect_identical(project_stream(stream_of(b), V, N, n_threads = nt), one)
        }
    }
})

test_that("a cell spanning several batches is exact", {
    big <- data.frame(row_id = 5L, col_id = seq_len(P), value = seq_len(P) / 10)
    df <- rbind(tri[tri$row_id < 5L, ], big, tri[tri$row_id > 5L, ])
    X <- matrix(0, N, P)
    X[cbind(df$row_id, df$col_id)] <- df$value
    b <- as_batches(df, random_sizes(nrow(df), 4L, seed = 3L))
    expect_equal(project_stream(stream_of(b), V, N, 2L), X %*% V, tolerance = 1e-12)
})

test_that("layouts that would give a wrong answer error instead", {
    set.seed(5L)
    expect_error(project_stream(tri[sample(nrow(tri)), ], V, N, 1L), "not sorted by row_id")
    odd <- tri[tri$row_id %% 2L == 1L, ]
    even <- tri[tri$row_id %% 2L == 0L, ]
    expect_error(project_stream(stream_of(list(odd, even)), V, N, 1L), "overlap in row_id")
})

test_that("bad input is rejected", {
    expect_error(project_stream(tri, V, max(tri$row_id) - 1L, 1L), "row_id outside")
    expect_error(project_stream(tri, V[1:10, ], N, 1L), "col_id outside")
    expect_error(project_stream(tri, as.vector(V), N, 1L), "`V` must be a non-empty numeric matrix")
    expect_error(project_stream(tri, V, 0L, 1L), "`n_rows` must be a positive integer")
})
