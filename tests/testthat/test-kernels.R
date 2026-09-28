test_that("has_kernel reports compiled kernels", {
    expect_true(has_kernel("gram_stream"))
    expect_false(has_kernel("not_a_kernel"))
    expect_error(has_kernel(c("a", "b")), "single string")
})

test_that("kernel_threads reads the option", {
    old <- options(GiottoKernels.threads = 3L)
    on.exit(options(old), add = TRUE)
    expect_identical(kernel_threads(), 3L)
    options(GiottoKernels.threads = 0L)
    expect_error(kernel_threads(), "positive integer")
    options(GiottoKernels.threads = NULL)
    expect_true(kernel_threads() >= 1L)
})
