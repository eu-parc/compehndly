# Python Integration Patterns

External configuration systems often need a stable import path rather than a
function object. Use wrappers in `compehndly.entrypoints` for that case.

## Stable Entrypoint Paths

Examples:

```yaml
map_fn: compehndly.entrypoints.summation
```

```yaml
map_fn: compehndly.entrypoints.normalize_specific_gravity
```

These wrappers are intentionally thin. Their signatures mirror the arguments
that configuration-driven callers pass explicitly.

## Polars `map_batches`

The integration tests exercise this pattern:

```python
import importlib
import polars as pl


def extract_callable(path: str):
    module_name, func_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


map_fn = extract_callable("compehndly.entrypoints.summation_allow_partial")
df = pl.DataFrame({"a": [1.0, None], "b": [3.0, 4.0]})

mapped = pl.struct(pl.col("a"), pl.col("b")).map_batches(
    lambda s: map_fn(
        a=s.struct.field("a"),
        b=s.struct.field("b"),
    ),
    return_dtype=pl.Float64,
)

out = df.lazy().select(mapped.alias("sum_col")).collect()
```

## Named Arguments

For non-commutative functions, pass fields by explicit argument name:

```python
map_fn = extract_callable("compehndly.entrypoints.normalize_specific_gravity")

mapped = pl.struct(
    pl.col("measurement").alias("measured"),
    pl.col("specific_gravity").alias("sg_measured"),
).map_batches(
    lambda s: map_fn(
        measured=s.struct.field("measured"),
        sg_measured=s.struct.field("sg_measured"),
        sg_ref=1.024,
    ),
    return_dtype=pl.Float64,
)
```

This is the safest pattern for derived variables whose inputs are not
interchangeable.
