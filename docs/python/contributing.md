# Contributing Python Functions

Derived-variable modules live in:

```text
python/src/compehndly/derived_variables/
```

Each function should provide:

1. An eager kernel that operates on `polars.Series`.
2. A lazy expression builder that operates on `polars.Expr`.
3. A `DerivedFunctionSpec` registration.

## Single Function Module

```python
from compehndly.polars.kernels import DerivedFunctionSpec


def my_kernel(measured: pl.Series, factor: float) -> pl.Series:
    return measured * factor


def my_expr(measured: pl.Expr, factor: float) -> pl.Expr:
    return measured * factor


FUNCTION_SPEC = DerivedFunctionSpec(
    name="my_function",
    kernel=my_kernel,
    expr_builder=my_expr,
)
```

## Multiple Functions In One Module

```python
FUNCTION_SPECS = [
    DerivedFunctionSpec(
        name="first_function",
        kernel=first_kernel,
        expr_builder=first_expr,
    ),
    DerivedFunctionSpec(
        name="second_function",
        kernel=second_kernel,
        expr_builder=second_expr,
    ),
]
```

The public registry is discovered automatically; there is no central registry
file to edit.

## Tests

Add focused unit tests under:

```text
python/tests/derived_variables/
```

If behavior should be shared across languages, add a case to:

```text
shared/conformance/derived_variables_cases.json
```

If a function has a stable config-driven entrypoint, add an integration test in:

```text
python/tests/test_polars_integration.py
```
