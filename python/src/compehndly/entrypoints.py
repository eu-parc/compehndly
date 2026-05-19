from __future__ import annotations

import polars as pl

from compehndly.api import get_map_fn

# Stable, path-addressable entrypoints for config-based loaders.
# These wrappers are intentionally thin, but their public signatures mirror
# the kernel kwargs that config-based loaders pass explicitly.

_SUMMATION = get_map_fn("summation", all_required=True)
_SUMMATION_ALLOW_PARTIAL = get_map_fn("summation", all_required=False)
_STANDARDIZE = get_map_fn("standardize")
_STANDARDIZE_CREATININE = get_map_fn("standardize_creatinine")
_NORMALIZE_SPECIFIC_GRAVITY = get_map_fn("normalize_specific_gravity")
_TOTAL_LIPID_CONCENTRATION = get_map_fn("total_lipid_concentration")
_STANDARDIZE_LIPID = get_map_fn("standardize_lipid")
_MEDIUM_BOUND_IMPUTATION_SCALAR_INPUT = get_map_fn(
    "medium_bound_imputation_scalar_input"
)
_MEDIUM_BOUND_IMPUTATION = get_map_fn("medium_bound_imputation")
_RANDOM_SINGLE_IMPUTATION_SCALAR_INPUT = get_map_fn(
    "random_single_imputation_scalar_input"
)
_RANDOM_SINGLE_IMPUTATION = get_map_fn("random_single_imputation")


def summation(**series_by_name: pl.Series) -> pl.Series:
    return _SUMMATION(**series_by_name)


def summation_allow_partial(**series_by_name: pl.Series) -> pl.Series:
    return _SUMMATION_ALLOW_PARTIAL(**series_by_name)


def standardize(measured: pl.Series, standard: pl.Series) -> pl.Series:
    return _STANDARDIZE(measured=measured, standard=standard)


def standardize_creatinine(measured: pl.Series, crt: pl.Series) -> pl.Series:
    return _STANDARDIZE_CREATININE(measured=measured, crt=crt)


def normalize_specific_gravity(
    measured: pl.Series,
    sg_measured: pl.Series,
    sg_ref: float,
) -> pl.Series:
    return _NORMALIZE_SPECIFIC_GRAVITY(
        measured=measured,
        sg_measured=sg_measured,
        sg_ref=sg_ref,
    )


def total_lipid_concentration(chol: pl.Series, trigl: pl.Series) -> pl.Series:
    return _TOTAL_LIPID_CONCENTRATION(chol=chol, trigl=trigl)


def standardize_lipid(
    measured: pl.Series,
    lipid_value: pl.Series,
) -> pl.Series:
    return _STANDARDIZE_LIPID(measured=measured, lipid_value=lipid_value)


def medium_bound_imputation_scalar_input(
    measurement: pl.Series,
    loq: float,
    lod: float | None = None,
) -> pl.Series:
    return _MEDIUM_BOUND_IMPUTATION_SCALAR_INPUT(
        measurement=measurement,
        loq=loq,
        lod=lod,
    )


def medium_bound_imputation(
    measurement: pl.Series,
    loq: pl.Series,
    lod: pl.Series | None = None,
) -> pl.Series:
    return _MEDIUM_BOUND_IMPUTATION(
        measurement=measurement,
        loq=loq,
        lod=lod,
    )


def random_single_imputation(
    biomarker: pl.Series,
    lod: pl.Series,
    loq: pl.Series,
    min_unique_values: int = 0,
    min_observed_percentage: int = 0,
    seed: int | None = None,
) -> pl.Series:
    return _RANDOM_SINGLE_IMPUTATION(
        biomarker=biomarker,
        lod=lod,
        loq=loq,
        min_unique_values=min_unique_values,
        min_observed_percentage=min_observed_percentage,
        seed=seed,
    )


def random_single_imputation_scalar_input(
    biomarker: pl.Series,
    lod: float,
    loq: float,
    min_unique_values: int = 0,
    min_observed_percentage: int = 0,
    seed: int | None = None,
) -> pl.Series:
    return _RANDOM_SINGLE_IMPUTATION_SCALAR_INPUT(
        biomarker=biomarker,
        lod=lod,
        loq=loq,
        min_unique_values=min_unique_values,
        min_observed_percentage=min_observed_percentage,
        seed=seed,
    )


__all__ = [
    "summation",
    "summation_allow_partial",
    "standardize",
    "standardize_creatinine",
    "normalize_specific_gravity",
    "total_lipid_concentration",
    "standardize_lipid",
    "medium_bound_imputation_scalar_input",
    "medium_bound_imputation",
    "random_single_imputation_scalar_input",
    "random_single_imputation",
]
