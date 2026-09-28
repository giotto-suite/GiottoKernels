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

#' Default thread count for multithreaded kernels
#'
#' Reads `options(GiottoKernels.threads)`. When unset, uses up to 8 of the
#' cores `parallel::detectCores()` reports; past about 8 the Gram kernel
#' measured no faster.
#'
#' @returns a positive integer.
#' @examples
#' kernel_threads()
#' @export
kernel_threads <- function() {
    n <- getOption("GiottoKernels.threads", NULL)
    if (is.null(n)) {
        n <- min(8L, parallel::detectCores(logical = FALSE), na.rm = TRUE)
    }
    n <- suppressWarnings(as.integer(n))
    if (length(n) != 1L || is.na(n) || n < 1L) {
        stop("[kernel_threads] option `GiottoKernels.threads` must be a positive integer",
            call. = FALSE)
    }
    n
}
