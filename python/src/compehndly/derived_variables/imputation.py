from __future__ import annotations

import numpy as np
import polars as pl

from compehndly.derived_variables.statsutils import fit_censored_lognorm
from compehndly.polars.kernels import DerivedFunctionSpec


def _validate_scalar_thresholds(loq: float, lod: float | None = None) -> None:
    if loq <= 0:
        raise ValueError("loq must be > 0")

    if lod is not None:
        if lod <= 0:
            raise ValueError("lod must be > 0")
        if lod >= loq:
            raise ValueError("lod must be < loq")


def _parse_bin_decoding_pairs(
    kwargs: dict[str, object],
) -> list[tuple[int, float, pl.Series | pl.Expr]]:
    filter_value_by_index: dict[int, float] = {}
    copy_from_by_index: dict[int, pl.Series | pl.Expr] = {}
    invalid_names: list[str] = []

    for name, value in kwargs.items():
        if name.startswith("filter_value_"):
            suffix = name.removeprefix("filter_value_")
            if not suffix.isdigit():
                invalid_names.append(name)
                continue
            try:
                filter_value_by_index[int(suffix)] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a float") from exc
        elif name.startswith("copy_from_"):
            suffix = name.removeprefix("copy_from_")
            if not suffix.isdigit():
                invalid_names.append(name)
                continue
            if not isinstance(value, (pl.Series, pl.Expr)):
                raise TypeError(f"{name} must be a Polars Series or Expr")
            copy_from_by_index[int(suffix)] = value
        else:
            invalid_names.append(name)

    if invalid_names:
        names = ", ".join(sorted(invalid_names))
        raise ValueError(
            "Unexpected arguments for bin_decoding: "
            f"{names}. Use filter_value_N/copy_from_N pairs."
        )

    indices = set(filter_value_by_index) | set(copy_from_by_index)
    if not indices:
        raise ValueError(
            "At least one filter_value_N/copy_from_N pair is required"
        )

    missing_filter_values = sorted(indices - set(filter_value_by_index))
    missing_copy_from = sorted(indices - set(copy_from_by_index))
    if missing_filter_values or missing_copy_from:
        message_parts = []
        if missing_filter_values:
            message_parts.append(
                "missing filter_value_"
                + ", filter_value_".join(
                    str(index) for index in missing_filter_values
                )
            )
        if missing_copy_from:
            message_parts.append(
                "missing copy_from_"
                + ", copy_from_".join(
                    str(index) for index in missing_copy_from
                )
            )
        raise ValueError("; ".join(message_parts))

    expected_indices = set(range(1, max(indices) + 1))
    if indices != expected_indices:
        missing_indices = sorted(expected_indices - indices)
        raise ValueError(
            "filter_value_N/copy_from_N indices must start at 1 and be "
            "contiguous; missing indices: "
            + ", ".join(str(index) for index in missing_indices)
        )

    filter_values = list(filter_value_by_index.values())
    non_nan_filter_values = [
        filter_value
        for filter_value in filter_values
        if not np.isnan(filter_value)
    ]
    if len(non_nan_filter_values) != len(set(non_nan_filter_values)):
        raise ValueError("filter_value_N values must be unique")
    if sum(np.isnan(filter_value) for filter_value in filter_values) > 1:
        raise ValueError("filter_value_N values must be unique")

    return [
        (
            index,
            filter_value_by_index[index],
            copy_from_by_index[index],
        )
        for index in sorted(indices)
    ]


def lab_sensitivity_dichotomization_kernel(
    measurement: pl.Series,
    loq: pl.Series,
    lod: pl.Series | None = None,
) -> pl.Series:
    length = len(measurement)
    if len(loq) != length:
        raise ValueError("measurement and loq must have the same length")
    if lod is not None and len(lod) != length:
        raise ValueError("measurement and lod must have the same length")

    result = [None] * len(measurement)
    return pl.Series(name=measurement.name, values=result, dtype=pl.Float64)


def lab_sensitivity_dichotomization_expr(
    measurement: pl.Expr,
    loq: pl.Expr,
    lod: pl.Expr | None = None,
) -> pl.Expr:
    if lod is None:
        return pl.when(measurement < loq).then(True).otherwise(False)

    return pl.when(measurement < lod).then(True).otherwise(False)


def medium_bound_imputation_scalar_input_kernel(
    measurement: pl.Series,
    loq: float,
    lod: float | None = None,
) -> pl.Series:
    _validate_scalar_thresholds(loq, lod)

    measurement_np = measurement.cast(pl.Float64).to_numpy()
    result = measurement_np.copy()

    if lod is None:
        mask = (measurement_np < loq) & ~np.isnan(measurement_np)
        result[mask] = loq / 2
        return pl.Series(
            name=measurement.name, values=result, dtype=pl.Float64
        )

    mask_below_lod = (measurement_np < lod) & ~np.isnan(measurement_np)
    result[mask_below_lod] = lod / 2

    midpoint = (lod + loq) / 2
    mask_between = (
        (measurement_np >= lod)
        & (measurement_np < loq)
        & ~np.isnan(measurement_np)
    )
    result[mask_between] = midpoint
    return pl.Series(name=measurement.name, values=result, dtype=pl.Float64)


def medium_bound_imputation_scalar_input_expr(
    measurement: pl.Expr,
    loq: float,
    lod: float | None = None,
) -> pl.Expr:
    _validate_scalar_thresholds(loq, lod)

    result = measurement
    if lod is None:
        return pl.when(measurement < loq).then(loq / 2).otherwise(result)

    result = pl.when(measurement < lod).then(lod / 2).otherwise(result)
    midpoint = (lod + loq) / 2
    return (
        pl.when((measurement >= lod) & (measurement < loq))
        .then(midpoint)
        .otherwise(result)
    )


def medium_bound_imputation_kernel(
    measurement: pl.Series,
    loq: pl.Series,
    lod: pl.Series | None = None,
) -> pl.Series:
    length = len(measurement)
    if len(loq) != length:
        raise ValueError("measurement and loq must have the same length")
    if lod is not None and len(lod) != length:
        raise ValueError("measurement and lod must have the same length")

    measurement_np = measurement.cast(pl.Float64).to_numpy()
    loq_np = loq.cast(pl.Float64).to_numpy()
    result = measurement_np.copy()

    if lod is None:
        mask = (
            (measurement_np < loq_np)
            & ~np.isnan(measurement_np)
            & ~np.isnan(loq_np)
        )
        result[mask] = loq_np[mask] / 2
        return pl.Series(
            name=measurement.name, values=result, dtype=pl.Float64
        )

    lod_np = lod.cast(pl.Float64).to_numpy()

    mask_below_lod = (
        (measurement_np < lod_np)
        & ~np.isnan(measurement_np)
        & ~np.isnan(lod_np)
    )
    result[mask_below_lod] = lod_np[mask_below_lod] / 2

    midpoint = (lod_np + loq_np) / 2
    mask_between = (
        (measurement_np >= lod_np)
        & (measurement_np < loq_np)
        & ~np.isnan(measurement_np)
        & ~np.isnan(lod_np)
        & ~np.isnan(loq_np)
    )
    result[mask_between] = midpoint[mask_between]

    return pl.Series(name=measurement.name, values=result, dtype=pl.Float64)


def medium_bound_imputation_expr(
    measurement: pl.Expr,
    loq: pl.Expr,
    lod: pl.Expr | None = None,
) -> pl.Expr:
    result = measurement
    if lod is None:
        return pl.when(measurement < loq).then(loq / 2).otherwise(result)

    result = pl.when(measurement < lod).then(lod / 2).otherwise(result)
    midpoint = (lod + loq) / 2
    return (
        pl.when((measurement >= lod) & (measurement < loq))
        .then(midpoint)
        .otherwise(result)
    )


def bin_decoding_kernel(
    *,
    values: pl.Series,
    **kwargs: object,
) -> pl.Series:
    """
    Replace sentinel values by copying from paired Series inputs.

    Call contract:
      values=<Series>,
      filter_value_1=<scalar>, copy_from_1=<Series>,
      filter_value_2=<scalar>, copy_from_2=<Series>,
      ...

    Rule indices must start at 1, be contiguous, form complete pairs, and
    use unique `filter_value_N` sentinel values. No other kwargs are accepted.
    """
    pairs = _parse_bin_decoding_pairs(kwargs)

    values_np = values.cast(pl.Float64).to_numpy()
    result = values_np.copy()

    for _, filter_value, copy_from in pairs:
        if len(copy_from) != len(values):
            raise ValueError(
                "values and copy_from_N inputs must have the same length"
            )

        copy_from_np = copy_from.cast(pl.Float64).to_numpy()
        if np.isnan(filter_value):
            mask = np.isnan(values_np)
        else:
            mask = values_np == filter_value
        result[mask] = copy_from_np[mask]

    return pl.Series(name=values.name, values=result, dtype=pl.Float64)


def bin_decoding_expr(
    *,
    values: pl.Expr,
    **kwargs: object,
) -> pl.Expr:
    """
    Replace sentinel values by copying from paired expression inputs.

    Call contract:
      values=<Expr>,
      filter_value_1=<scalar>, copy_from_1=<Expr>,
      filter_value_2=<scalar>, copy_from_2=<Expr>,
      ...

    Rule indices must start at 1, be contiguous, form complete pairs, and
    use unique `filter_value_N` sentinel values. No other kwargs are accepted.
    """
    pairs = _parse_bin_decoding_pairs(kwargs)

    result = values
    for _, filter_value, copy_from in pairs:
        if np.isnan(filter_value):
            mask = values.is_nan()
        else:
            mask = values == filter_value
        result = pl.when(mask).then(copy_from).otherwise(result)

    return result


def _random_single_imputation_from_arrays(
    biomarker_np: np.ndarray,
    lod_np: np.ndarray,
    loq_np: np.ndarray,
    min_unique_values: int = 0,
    min_observed_percentage: int = 0,
    seed: int | None = None,
) -> np.ndarray:
    biomarker_filled = np.where(np.isnan(biomarker_np), -1.0, biomarker_np)

    # perform configurable data checks
    checks_failed = False

    # check: at least [min_observed_percentage] % of the values are above LOD/LOQ
    if not checks_failed:
        count_above_lod_loq = np.count_nonzero(
            biomarker_filled > np.where(~np.isnan(lod_np), lod_np, loq_np)
        )
        if (
            count_above_lod_loq
            < biomarker_np.size / 100.0 * min_observed_percentage
        ):
            checks_failed = True

    # check: at least [min_unique_values] unique values are observed above LOD/LOQ
    if not checks_failed:
        above_lod_loq = biomarker_filled > np.where(
            ~np.isnan(lod_np), lod_np, loq_np
        )
        count_unique_values_above_lod_loq = np.unique(
            biomarker_filled[above_lod_loq]
        ).size
        if count_unique_values_above_lod_loq < min_unique_values:
            checks_failed = True

    if checks_failed:
        result = np.full(biomarker_np.size, np.nan)
    else:
        censored = biomarker_filled < 0
        values_np = np.where(censored, lod_np, biomarker_filled)

        dist = fit_censored_lognorm(values_np, censored)
        rng = np.random.default_rng(seed=seed)

        lower = np.zeros_like(biomarker_filled, dtype=float)
        upper = np.zeros_like(biomarker_filled, dtype=float)

        cat_below_lod = biomarker_filled == -1
        cat_between = biomarker_filled == -2
        cat_below_loq = biomarker_filled == -3

        lower[cat_below_lod] = 0
        upper[cat_below_lod] = lod_np[cat_below_lod]

        lower[cat_between] = lod_np[cat_between]
        upper[cat_between] = loq_np[cat_between]

        lower[cat_below_loq] = 0
        upper[cat_below_loq] = loq_np[cat_below_loq]

        cdf_lo = dist.cdf(lower)
        cdf_hi = dist.cdf(upper)

        u = rng.uniform(cdf_lo, cdf_hi)
        imputed = dist.ppf(u)

        result = biomarker_filled.copy()
        result[censored] = imputed[censored]

    return result


def random_single_imputation_scalar_input_kernel(
    biomarker: pl.Series,
    lod: float,
    loq: float,
    min_unique_values: int = 0,
    min_observed_percentage: int = 0,
    seed: int | None = None,
) -> pl.Series:
    _validate_scalar_thresholds(loq, lod)

    biomarker_np = biomarker.cast(pl.Float64).to_numpy()
    result = _random_single_imputation_from_arrays(
        biomarker_np=biomarker_np,
        lod_np=np.full(biomarker_np.size, lod, dtype=float),
        loq_np=np.full(biomarker_np.size, loq, dtype=float),
        min_unique_values=min_unique_values,
        min_observed_percentage=min_observed_percentage,
        seed=seed,
    )

    return pl.Series(result, dtype=pl.Float64)


def random_single_imputation_scalar_input_expr(
    biomarker: pl.Expr,
    lod: float,
    loq: float,
    min_unique_values: int = 0,
    min_observed_percentage: int = 0,
    seed: int | None = None,
) -> pl.Expr:
    _validate_scalar_thresholds(loq, lod)

    return pl.struct([biomarker.alias("_biomarker")]).map_batches(
        lambda s: random_single_imputation_scalar_input_kernel(
            s.struct.field("_biomarker"),
            lod=lod,
            loq=loq,
            min_unique_values=min_unique_values,
            min_observed_percentage=min_observed_percentage,
            seed=seed,
        ),
        return_dtype=pl.Float64,
    )


def random_single_imputation_kernel(
    biomarker: pl.Series,
    lod: pl.Series,
    loq: pl.Series,
    min_unique_values: int = 0,
    min_observed_percentage: int = 0,
    seed: int | None = None,
) -> pl.Series:
    length = len(biomarker)
    if len(lod) != length:
        raise ValueError("biomarker and lod must have the same length")
    if len(loq) != length:
        raise ValueError("biomarker and loq must have the same length")

    biomarker_np = biomarker.cast(pl.Float64).to_numpy()
    lod_np = lod.cast(pl.Float64).to_numpy()
    loq_np = loq.cast(pl.Float64).to_numpy()

    if np.any(np.isnan(lod_np)) or np.any(np.isnan(loq_np)):
        raise ValueError("lod and loq values must not be NaN")

    invalid_thresholds = (
        (~np.isnan(lod_np) & (lod_np <= 0))
        | (~np.isnan(loq_np) & (loq_np <= 0))
        | (~np.isnan(lod_np) & ~np.isnan(loq_np) & (lod_np >= loq_np))
    )
    if np.any(invalid_thresholds):
        raise ValueError("lod values must be > 0 and < loq values")

    result = _random_single_imputation_from_arrays(
        biomarker_np=biomarker_np,
        lod_np=lod_np,
        loq_np=loq_np,
        min_unique_values=min_unique_values,
        min_observed_percentage=min_observed_percentage,
        seed=seed,
    )

    return pl.Series(result, dtype=pl.Float64)


def random_single_imputation_expr(
    biomarker: pl.Expr,
    lod: pl.Expr,
    loq: pl.Expr,
    min_unique_values: int = 0,
    min_observed_percentage: int = 0,
    seed: int | None = None,
) -> pl.Expr:
    return pl.struct(
        [
            biomarker.alias("_biomarker"),
            lod.alias("_lod"),
            loq.alias("_loq"),
        ]
    ).map_batches(
        lambda s: random_single_imputation_kernel(
            biomarker=s.struct.field("_biomarker"),
            lod=s.struct.field("_lod"),
            loq=s.struct.field("_loq"),
            min_unique_values=min_unique_values,
            min_observed_percentage=min_observed_percentage,
            seed=seed,
        ),
        return_dtype=pl.Float64,
    )


FUNCTION_SPECS = [
    DerivedFunctionSpec(
        name="lab_sensitivity_dichotomization",
        kernel=lab_sensitivity_dichotomization_kernel,
        expr_builder=lab_sensitivity_dichotomization_expr,
    ),
    DerivedFunctionSpec(
        name="medium_bound_imputation_scalar_input",
        kernel=medium_bound_imputation_scalar_input_kernel,
        expr_builder=medium_bound_imputation_scalar_input_expr,
    ),
    DerivedFunctionSpec(
        name="medium_bound_imputation",
        kernel=medium_bound_imputation_kernel,
        expr_builder=medium_bound_imputation_expr,
    ),
    DerivedFunctionSpec(
        name="bin_decoding",
        kernel=bin_decoding_kernel,
        expr_builder=bin_decoding_expr,
    ),
    DerivedFunctionSpec(
        name="random_single_imputation_scalar_input",
        kernel=random_single_imputation_scalar_input_kernel,
        expr_builder=random_single_imputation_scalar_input_expr,
    ),
    DerivedFunctionSpec(
        name="random_single_imputation",
        kernel=random_single_imputation_kernel,
        expr_builder=random_single_imputation_expr,
    ),
]
