#' GiottoKernels: optional compiled kernels for the Giotto suite
#'
#' Small compiled routines for the hot inner loops of Giotto suite
#' computations. A kernel takes plain vectors or an Arrow C stream and returns
#' plain vectors; deciding what to compute, and feeding the data in, stays with
#' the calling package.
#'
#' Nothing requires this package. A caller checks [has_kernel()] and falls
#' back to its own R implementation when the kernel is absent, which also
#' covers an older install that predates a kernel. Results match that R path
#' to floating-point rounding.
#'
#' Format- or platform-specific readers (for example a single vendor's file
#' format) do not belong here: they change when the format changes, and ship as
#' their own packages.
#'
#' @section Threads:
#' Multithreaded kernels take `n_threads`, default `NULL`, resolved by
#' [kernel_threads()]: an explicit value is used as given, otherwise
#' `options(gkernels.n_threads)`, otherwise 1. Threads are Rust `std::thread`
#' workers inside the call, so no R process is forked. Results are
#' bit-identical across runs for a fixed thread count ([gram_stream()]) or for
#' any thread count ([project_stream()]).
#'
#' @keywords internal
"_PACKAGE"
