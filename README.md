# compehndly

`compehndly` is a cross-language collection of derived-variable functions for
Personal Exposure Health (PEH) data.

Current focus:

- Python implementation (Polars-first, including LazyFrame workflows)
- R implementation aligned to the same function behavior
- shared conformance vectors to keep Python and R outputs consistent

## Repository Layout

- `python/`: Python package and tests
- `R/`: R package scaffolding and tests
- `shared/conformance/`: cross-language conformance cases

## Documentation

- Python usage and architecture:
  - [python/README.md](python/README.md)
- R usage and test setup:
  - [R/README.md](R/README.md)
- Shared conformance format:
  - [shared/conformance/README.md](shared/conformance/README.md)

## Conformance

Python and R test runners consume this same file to detect drift across
implementations.
