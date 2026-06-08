# Python Usage

## Direct Series Usage

```python
import polars as pl
from compehndly import apply

a = pl.Series([1.0, None, 3.0])
b = pl.Series([None, 2.0, None])

out = apply("summation", a, b, all_required=False)
assert out.to_list() == [1.0, 2.0, 3.0]
```

## Lazy Expression Usage

```python
import polars as pl
from compehndly import apply

df = pl.DataFrame({"a": [1.0, None], "b": [3.0, 4.0]})

expr = apply(
    "summation",
    pl.col("a"),
    pl.col("b"),
    all_required=False,
).alias("sum_col")

out = df.lazy().select(expr).collect()
```

## Add A Derived Column

```python
from compehndly import with_derived_column

out = with_derived_column(
    frame=df.lazy(),
    function_name="summation",
    input_columns=["a", "b"],
    output_column="sum_col",
    all_required=False,
).collect()
```

Use a named input mapping when function argument names matter:

```python
out = with_derived_column(
    frame=df,
    function_name="normalize_specific_gravity",
    input_columns={"measured": "measurement", "sg_measured": "specific_gravity"},
    output_column="normalized",
    sg_ref=1.024,
)
```

## Discover Functions

```python
from compehndly import list_functions

print(list_functions())
```
