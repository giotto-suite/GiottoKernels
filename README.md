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

### `project_stream()`

The second pass of Gram-eigen PCA: `A %*% V`, cell coordinates from the
loadings, from the same kind of stream. Only stored entries are touched
(`nnz * ncol(V)`), and every output row is built by one thread from its
entries in stream order, so the result is bit-identical at any thread count.
Centering is left to the caller: `(A - 1 mu^T) V = A V - 1 (mu^T V)`.

```r
coords <- GiottoKernels::project_stream(reader, V, n_rows = n_cells)
```

Its layout requirement and checks are the same as `gram_stream()`'s.

### `edge_products_stream()`

The cross-cell term of Moran's I, Geary's C and join counts: for each feature,
`sum_e w_e x[i_e, g] x[j_e, g]` over the edges of a spatial graph. The stream
is read once into a row-compressed table (about 12 bytes per stored value
held, 28 at peak), so batch and row order do not matter, and each edge walks
its two cells' feature lists together. Feature sums are independent, so a
caller bounds memory by streaming a subset of features per call. Edge blocks
are summed in a fixed order, so the result is bit-identical at any thread
count.

```r
T <- GiottoKernels::edge_products_stream(reader, from, to,
    n_rows = n_cells, n_features = n_genes, w = weights)
```

### `graph_pairs_stream()`

The edge list `edge_products_stream()` takes, built from a stream of graph
edges in node ids and a node -> row map: each pair once as `a <= b`, edges
leaving the rows dropped, and (with `dedupe`) a pair a directed graph lists
twice collapsed to one at its mean weight. It reads int32 or int64 ids and
float32 or float64 weights, so it takes an edge store's stream as is, and
never materialises the edges in R.

```r
p <- GiottoKernels::graph_pairs_stream(edge_reader, node_row, weighted = TRUE)
```

## Threads

Multithreaded kernels take `n_threads`, default `NULL`:

1. an explicit value is used as given (coerced to integer);
2. `NULL` falls back to `options(gkernels.n_threads)`;
3. and that defaults to 1.

So a kernel is single-threaded unless the caller or the session asks for more:

```r
options(gkernels.n_threads = 8L)            # session default
GiottoKernels::gram_stream(reader, 2000L)   # 8 threads
GiottoKernels::gram_stream(reader, 2000L, n_threads = 2L)
```

`gram_stream()` gives bit-identical results across runs for a fixed thread
count; `project_stream()` for any thread count. Each Gram worker
holds its own `P x P` accumulator, so the thread count is lowered when needed
to stay within `options(gkernels.memory_gb)` (default 4).

## Installation

Building from source needs a Rust toolchain (`cargo`, `rustc` >= 1.65):

```r
remotes::install_github("giotto-suite/GiottoKernels")
```
