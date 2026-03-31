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


def random_single_imputation_kernel(
    biomarker: pl.Series,
    lod: float,
    loq: float,
    seed: int | None = None,
) -> pl.Series:
    _validate_scalar_thresholds(loq, lod)

    biomarker_np = biomarker.cast(pl.Float64).to_numpy()
    biomarker_filled = np.where(np.isnan(biomarker_np), -1.0, biomarker_np)

    censored = biomarker_filled < 0
    values_np = np.where(censored, lod, biomarker_filled)

    dist = fit_censored_lognorm(values_np, censored)
    rng = np.random.default_rng(seed=seed)

    lower = np.zeros_like(biomarker_filled, dtype=float)
    upper = np.zeros_like(biomarker_filled, dtype=float)

    cat_below_lod = biomarker_filled == -1
    cat_between = biomarker_filled == -2
    cat_below_loq = biomarker_filled == -3

    lower[cat_below_lod] = 0
    upper[cat_below_lod] = lod

    lower[cat_between] = lod
    upper[cat_between] = loq

    lower[cat_below_loq] = 0
    upper[cat_below_loq] = loq

    cdf_lo = dist.cdf(lower)
    cdf_hi = dist.cdf(upper)

    u = rng.uniform(cdf_lo, cdf_hi)
    imputed = dist.ppf(u)

    result = biomarker_filled.copy()
    result[censored] = imputed[censored]

    return pl.Series(result, dtype=pl.Float64)


def random_single_imputation_expr(
    biomarker: pl.Expr,
    lod: float,
    loq: float,
    seed: int | None = None,
) -> pl.Expr:
    _validate_scalar_thresholds(loq, lod)

    return pl.struct([biomarker.alias("_biomarker")]).map_batches(
        lambda s: random_single_imputation_kernel(
            s.struct.field("_biomarker"),
            lod=lod,
            loq=loq,
            seed=seed,
        ),
        return_dtype=pl.Float64,
    )


FUNCTION_SPECS = [
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
        name="random_single_imputation",
        kernel=random_single_imputation_kernel,
        expr_builder=random_single_imputation_expr,
    ),
]
