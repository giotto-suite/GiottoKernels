test_that("has_kernel reports compiled kernels", {
    expect_true(has_kernel("gram_stream"))
    expect_true(has_kernel("project_stream"))
    expect_false(has_kernel("not_a_kernel"))
    expect_error(has_kernel(c("a", "b")), "single string")
})

test_that("kernel_threads: explicit value, then option, then 1", {
    old <- options(gkernels.n_threads = NULL)
    on.exit(options(old), add = TRUE)
    expect_identical(kernel_threads(), 1L)
    expect_identical(kernel_threads(2), 2L)
    expect_identical(kernel_threads("2"), 2L)

    options(gkernels.n_threads = 2L)
    expect_identical(kernel_threads(), 2L)
    expect_identical(kernel_threads(1L), 1L) # explicit beats the option

    options(gkernels.n_threads = 0L)
    expect_error(kernel_threads(), "option `gkernels.n_threads` must be a positive integer")
    expect_error(kernel_threads(NA), "`n_threads` must be a positive integer")
    expect_error(kernel_threads(c(1, 2)), "`n_threads` must be a positive integer")
})

test_that("gram_stream uses the option when n_threads is NULL", {
    old <- options(gkernels.n_threads = 2L)
    on.exit(options(old), add = TRUE)
    tri <- sim_triplets()
    expect_identical(gram_stream(tri, 25L), gram_stream(tri, 25L, n_threads = 2L))
})
