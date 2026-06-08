# Python Functions

This page documents the current registered Python functions.

## Summation

### `summation`

Sums one or more series or expressions. Row-level nulls are treated as zero.

Parameters:

- `all_required`: defaults to `True`. When `True` and no `cutoff` is provided,
  any entirely-null input series makes the full result null.
- `cutoff`: optional fraction from `0` to `1`. When provided, the result is
  returned if at least one input has a non-null fraction greater than or equal
  to `cutoff`; otherwise the full result is null. `cutoff` takes precedence over
  `all_required`.

```python
out = apply("summation", a, b, all_required=False)
```

## Correction And Standardization

### `standardize`

Computes:

```text
measured * 100 / standard
```

### `standardize_creatinine`

Creatinine-specific wrapper around `standardize`, using `crt` as the standard.

### `normalize_specific_gravity`

Computes:

```text
measured * (sg_ref - 1) / sg_measured
```

### `total_lipid_concentration`

Computes:

```text
chol * 2.27 + trigl + 62.3
```

### `standardize_lipid`

Lipid-specific wrapper around `standardize`.

### `coalesce_by_priority`

Returns the first non-null value according to a named priority sequence.

```python
out = apply(
    "coalesce_by_priority",
    primary=df["lab_a"],
    secondary=df["lab_b"],
    fallback=df["lab_c"],
    priority=("primary", "secondary", "fallback"),
)
```

### `consolidate_lipid_value`

Uses the lipid priority order:

1. `lipid_enz_harm`
2. `lipid_enz_imp`
3. `lipid_imp`

## Imputation

### `lab_sensitivity_dichotomization`

Produces a boolean expression indicating whether a measurement is below the
limit of quantification or, when provided, below the limit of detection.

### `medium_bound_imputation_scalar_input`

Uses scalar `loq` and optional scalar `lod` thresholds.

### `medium_bound_imputation`

Uses series-valued `loq` and optional series-valued `lod` thresholds.

### `random_single_imputation_scalar_input`

Runs random single imputation with scalar `lod` and `loq` thresholds.

### `random_single_imputation`

Runs random single imputation with series-valued `lod` and `loq` thresholds.

Optional imputation controls include:

- `min_unique_values`
- `min_observed_percentage`
- `seed`
