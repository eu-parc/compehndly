# Python Overview

The Python package exposes derived-variable functions for eager `polars.Series`
workflows and lazy `polars.Expr` / `polars.LazyFrame` workflows.

The implementation has three layers:

1. Derived-variable modules define one eager kernel and one lazy expression
   builder per function.
2. `compehndly.api` discovers `FUNCTION_SPEC` and `FUNCTION_SPECS` from
   `compehndly.derived_variables`.
3. Public helpers dispatch to the correct eager or lazy path.

## Public Helpers

Use these imports for most Python workflows:

```python
from compehndly import apply, get_map_fn, list_functions, with_derived_column
```

- `list_functions()` returns the registered derived-variable names.
- `apply(function_name, ...)` applies a function to `polars.Series` or
  `polars.Expr` inputs.
- `with_derived_column(...)` adds an output column to a `polars.DataFrame` or
  `polars.LazyFrame`.
- `get_map_fn(function_name, **params)` builds a stable callable for external
  `map_batches` style integrations.

Stable path-addressable wrappers also live in:

```python
compehndly.entrypoints
```
