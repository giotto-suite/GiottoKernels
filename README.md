# GiottoKernels

Optional compiled kernels for the Giotto suite, written in Rust and exposed to
R through [extendr](https://extendr.github.io/).

A kernel here is the hot inner loop of a computation: plain vectors or an
Arrow C stream in, plain vectors out. It knows nothing about Giotto's classes.
Deciding what to compute, and feeding the data in, stays with the calling
package.

Nothing requires this package. Callers check for a kernel and otherwise run
their own R implementation, so installing it changes speed, not results
beyond floating-point rounding:

```r
use_kernel <- requireNamespace("GiottoKernels", quietly = TRUE) &&
    GiottoKernels::has_kernel("gram_stream")
```

`has_kernel()` checks the installed build, so a caller on an older release
that lacks a kernel falls back rather than failing.

Readers tied to one file format or vendor belong in their own packages, not
here.

## Kernels

### `gram_stream()`

The first pass of Gram-eigen PCA: `G = sum_c x_c x_c^T` and the feature sums,
over cells `c`, from a stream of sparse `(row_id, col_id, value)` triplets.
Only nonzero pairs within each cell are multiplied, memory stays at one batch
plus the `P x P` result, and no R process is forked.

```r
library(arrow)
reader <- Scanner$create(open_dataset("expr_hvf/"),
    projection = c("row_id", "col_id", "value"))$ToRecordBatchReader()
res <- GiottoKernels::gram_stream(reader, n_features = 2000L)
str(res) # $G 2000 x 2000, $s length 2000
```

Batch boundaries may cut a cell. What the stream must not do is put part of a
cell that is interior to one batch into another batch; a stream sorted by
`row_id` never does. The call checks this and errors rather than return a
wrong result.

Measured on the pass it replaces, 169,420 cells x 2,000 features,
17.6M nonzeros, Apple M-series:

| path | time |
|---|---:|
| R, one process | 9.1 s |
| R, 8 forked workers | 1.8 s |
| `gram_stream()`, 1 thread | 1.6 s |
| `gram_stream()`, 8 threads | 0.64 s |

## Threads

Multithreaded kernels take `n_threads`, defaulting to `kernel_threads()`:
`options(GiottoKernels.threads)`, or up to 8 physical cores when unset. A
fixed thread count gives bit-identical results across runs. Each Gram worker
holds its own `P x P` accumulator, so the thread count is lowered when needed
to stay within `options(GiottoKernels.memory_gb)` (default 4).

## Installation

Building from source needs a Rust toolchain (`cargo`, `rustc` >= 1.65):

```r
remotes::install_github("giotto-suite/GiottoKernels")
```
