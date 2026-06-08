# compehndly

`compehndly` is a cross-language collection of derived-variable functions for
Personal Exposure Health data.

The project currently centers on a Polars-first Python implementation, with an
R implementation kept aligned through shared conformance vectors.

## Documentation Map

- [Python overview](python/index.md): current implementation, public APIs, and
  function behavior.
- [Python integration patterns](python/integration.md): how external callers
  use stable entrypoint paths and Polars `map_batches`.
- [R overview](r/index.md): placeholder structure for the R implementation.
- [Shared conformance](shared/conformance.md): how Python and R stay aligned.

## Repository Layout

- `python/`: Python package, Polars kernels, adapters, and tests.
- `R/`: R package scaffold and implementation.
- `shared/conformance/`: shared JSON test vectors.
- `docs/`: this documentation site.
