#' Is a kernel available in this build?
#'
#' Checks the kernels compiled into the installed package, so a caller can
#' fall back on an older install instead of calling a kernel it does not have:
#'
#' ```r
#' use_kernel <- requireNamespace("GiottoKernels", quietly = TRUE) &&
#'     GiottoKernels::has_kernel("gram_stream")
#' ```
#'
#' @param name kernel name, e.g. `"gram_stream"`.
#' @returns `TRUE` or `FALSE`.
#' @examples
#' has_kernel("gram_stream")
#' @export
has_kernel <- function(name) {
    if (!is.character(name) || length(name) != 1L || is.na(name)) {
        stop("[has_kernel] `name` must be a single string", call. = FALSE)
    }
    name %in% kernels_rs()
}

#' Resolve the thread count for a kernel call
#'
#' An explicit `n_threads` is used as given. `NULL` falls back to
#' `options(gkernels.n_threads)`, and to 1 when that is unset, so a kernel runs
#' single-threaded unless the caller or the session asks for more.
#'
#' @param n_threads `NULL`, or a positive whole number.
#' @returns a positive integer.
#' @examples
#' kernel_threads()   # 1, or options(gkernels.n_threads)
#' kernel_threads(4)  # 4
#' @export
kernel_threads <- function(n_threads = NULL) {
    src <- "`n_threads`"
    if (is.null(n_threads)) {
        n_threads <- getOption("gkernels.n_threads", 1L)
        src <- "option `gkernels.n_threads`"
    }
    n <- suppressWarnings(as.integer(n_threads))
    if (length(n) != 1L || is.na(n) || n < 1L) {
        stop("[kernel_threads] ", src, " must be a positive integer", call. = FALSE)
    }
    n
}
