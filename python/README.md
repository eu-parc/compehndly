# compehndly (Python)

## Architecture (Polars-First)

This package now follows a two-layer execution model for derived variables:

1. `compehndly.polars.kernels`
- Polars-native series kernels for external `map_batches` usage.
- Contract: `kernel(*series: pl.Series, **params) -> pl.Series`

2. `compehndly.polars.adapters`
- Frame adapters that apply kernels to:
  - `polars.DataFrame` (eager kernel path)
  - `polars.LazyFrame` (expression path)

## Public Entry Points

External integrations (already have a lazy orchestrator like `apply_map`):

```python
from compehndly import get_map_fn

map_fn = get_map_fn("summation", all_required=True)
```

For config/YAML path-based loading, use stable wrappers in `compehndly.entrypoints`:

```yaml
map_fn: compehndly.entrypoints.summation
# or
map_fn: compehndly.entrypoints.normalize_specific_gravity
```

Direct usage (you want this package to add a derived column):

```python
from compehndly import with_derived_column

out = with_derived_column(
    frame=df_or_lf,
    function_name="summation",
    input_columns=["a", "b"],
    output_column="sum_col",
    all_required=True,
)

# For non-commutative functions use a named mapping
out = with_derived_column(
    frame=df_or_lf,
    function_name="normalize_specific_gravity",
    input_columns={"measured": "measurement_col", "sg_measured": "sg_col"},
    output_column="normalized",
    sg_ref=1.024,
)
```

Discover available functions:

```python
from compehndly import list_functions

print(list_functions())
```

Direct series/expression application:

```python
from compehndly import apply

out_series = apply("summation", df["a"], df["b"], all_required=False)
out_expr = apply("summation", pl.col("a"), pl.col("b"), all_required=False)

# Named data kwargs are supported
out_named = apply(
    "normalize_specific_gravity",
    measured=df["measured"],
    sg_measured=df["sg_measured"],
    sg_ref=1.024,
)
```

## Current Scope

The pattern is applied to:

- `derived_variables.summation`
- `derived_variables.correction`
- `derived_variables.imputation`

## Cross-Language Conformance

Shared test vectors live in:

- `shared/conformance/derived_variables_cases.json`

Python and R should both execute this same file to verify parity.
The Python runner is in:

- `python/tests/test_conformance_shared_vectors.py`

## Contributor Workflow (Two Steps)

Add one module in `compehndly.derived_variables` and do only:

1. Implement one kernel:
- `def my_kernel(*series: pl.Series, scalar_a: float, ...) -> pl.Series`

2. Implement one expression builder:
- `def my_expr(*exprs: pl.Expr, scalar_a: float, ...) -> pl.Expr`

Then expose one of:

```python
FUNCTION_SPEC = DerivedFunctionSpec(
    name="my_function",
    kernel=my_kernel,
    expr_builder=my_expr,
)
```

```python
FUNCTION_SPECS = [
    DerivedFunctionSpec(...),
    DerivedFunctionSpec(...),
]
```

`compehndly.api` auto-discovers `FUNCTION_SPEC` / `FUNCTION_SPECS`, so there
is no central registry file to edit.

No per-function adapter wrappers are required.
