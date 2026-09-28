P <- 25L
tri <- sim_triplets(n_features = P)
ref <- ref_gram(tri, P)

test_that("one batch matches the dense reference", {
    res <- gram_stream(tri, P, n_threads = 1L)
    expect_equal(res$G, ref$G, tolerance = 1e-12)
    expect_equal(res$s, ref$s, tolerance = 1e-12)
    expect_true(isSymmetric(res$G))
})

test_that("cells cut by batch boundaries are exact", {
    for (max_size in c(1L, 3L, 7L, 40L)) {
        b <- as_batches(tri, random_sizes(nrow(tri), max_size))
        res <- gram_stream(stream_of(b), P, n_threads = 2L)
        expect_equal(res$G, ref$G, tolerance = 1e-12)
        expect_equal(res$s, ref$s, tolerance = 1e-12)
    }
})

test_that("a cell spanning several batches is exact", {
    big <- data.frame(row_id = 5L, col_id = seq_len(P), value = seq_len(P) / 10)
    df <- rbind(tri[tri$row_id < 5L, ], big, tri[tri$row_id > 5L, ])
    r <- ref_gram(df, P)
    b <- as_batches(df, random_sizes(nrow(df), 4L, seed = 3L))
    res <- gram_stream(stream_of(b), P, n_threads = 2L)
    expect_equal(res$G, r$G, tolerance = 1e-12)
    expect_equal(res$s, r$s, tolerance = 1e-12)
})

test_that("features within a cell need not be sorted", {
    set.seed(4L)
    shuffled <- do.call(rbind, lapply(split(tri, tri$row_id), function(d) d[sample(nrow(d)), ]))
    res <- gram_stream(shuffled, P, n_threads = 1L)
    expect_equal(res$G, ref$G, tolerance = 1e-12)
})

test_that("a fixed thread count is bit-identical across runs", {
    b <- function() stream_of(as_batches(tri, random_sizes(nrow(tri), 30L)))
    # at most 2 threads: CRAN checks allow no more
    for (nt in c(1L, 2L)) {
        expect_identical(gram_stream(b(), P, nt), gram_stream(b(), P, nt))
    }
    expect_equal(gram_stream(b(), P, 1L), gram_stream(b(), P, 2L), tolerance = 1e-12)
})

test_that("extra columns are ignored", {
    df <- tri
    df$source_id <- "a"
    expect_equal(gram_stream(df, P, 1L)$G, ref$G, tolerance = 1e-12)
})

test_that("an empty stream gives zeros", {
    res <- gram_stream(tri[0, ], P, 1L)
    expect_equal(res$G, matrix(0, P, P))
    expect_equal(res$s, numeric(P))
})

test_that("the memory budget lowers the thread count, not the result", {
    old <- options(gkernels.memory_gb = 0)
    on.exit(options(old), add = TRUE)
    expect_identical(GiottoKernels:::.threads_within_budget(P), 1L)
    expect_equal(gram_stream(tri, P, n_threads = 2L)$G, ref$G, tolerance = 1e-12)
})

test_that("layouts that would give a wrong answer error instead", {
    # rows out of order inside one batch
    set.seed(5L)
    expect_error(gram_stream(tri[sample(nrow(tri)), ], P, 1L), "not sorted by row_id")
    # each batch sorted, but a cell interior to one batch also appears in another
    odd <- tri[tri$row_id %% 2L == 1L, ]
    even <- tri[tri$row_id %% 2L == 0L, ]
    expect_error(gram_stream(stream_of(list(odd, even)), P, 1L), "overlap in row_id")
})

test_that("batches may share one edge cell", {
    # cell 10 ends batch 1 and starts batch 2
    cut <- which(tri$row_id == 10L)
    b <- list(tri[seq_len(cut[2L]), ], tri[-seq_len(cut[2L]), ])
    expect_equal(gram_stream(stream_of(b), P, 1L)$G, ref$G, tolerance = 1e-12)
})

test_that("bad input is rejected", {
    bad <- tri
    bad$col_id[1L] <- P + 1L
    expect_error(gram_stream(bad, P, 1L), "col_id outside")
    bad$col_id[1L] <- 0L
    expect_error(gram_stream(bad, P, 1L), "col_id outside")

    bad <- tri
    bad$value[3L] <- NA
    expect_error(gram_stream(bad, P, 1L), "must not contain nulls")

    bad <- tri
    bad$value <- as.integer(round(bad$value * 10))
    expect_error(gram_stream(bad, P, 1L), "column 'value' has Arrow type 'i', expected 'g'")

    expect_error(gram_stream(tri[, c("row_id", "value")], P, 1L), "no column 'col_id'")
    expect_error(gram_stream(tri, 0L, 1L), "`n_features` must be a positive integer")
    expect_error(gram_stream(tri, P, 0L), "`n_threads` must be a positive integer")
    expect_error(gram_stream(tri, P, "two"), "`n_threads` must be a positive integer")
})
