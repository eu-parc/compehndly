from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from compehndly.api import get_map_fn

# Stable, path-addressable entrypoints for config-based loaders.
# These wrappers are intentionally thin, but their public signatures mirror
# the kernel kwargs that config-based loaders pass explicitly.

_LAB_SENSITIVITY_DICHOTOMIZATION = get_map_fn(
    "lab_sensitivity_dichotomization"
)
_SUMMATION = get_map_fn("summation", all_required=True)
_SUMMATION_ALLOW_PARTIAL = get_map_fn("summation", all_required=False)
_MULTIPLY_BY_GROUP = get_map_fn("multiply_by_group")
_WEIGHTED_SUMMATION = get_map_fn("weighted_summation")
_STANDARDIZE = get_map_fn("standardize")
_STANDARDIZE_CREATININE = get_map_fn("standardize_creatinine")
_NORMALIZE_SPECIFIC_GRAVITY = get_map_fn("normalize_specific_gravity")
_TOTAL_LIPID_CONCENTRATION = get_map_fn("total_lipid_concentration")
_CONSOLIDATE_LIPID_VALUE = get_map_fn("consolidate_lipid_value")
_STANDARDIZE_LIPID = get_map_fn("standardize_lipid")
_MEDIUM_BOUND_IMPUTATION_SCALAR_INPUT = get_map_fn(
    "medium_bound_imputation_scalar_input"
)
_MEDIUM_BOUND_IMPUTATION = get_map_fn("medium_bound_imputation")
_BIN_DECODING = get_map_fn("bin_decoding")
_RANDOM_SINGLE_IMPUTATION_SCALAR_INPUT = get_map_fn(
    "random_single_imputation_scalar_input"
)
_RANDOM_SINGLE_IMPUTATION = get_map_fn("random_single_imputation")


def lab_sensitivity_dichotomization(**series_by_name: pl.Series) -> pl.Series:
    return _LAB_SENSITIVITY_DICHOTOMIZATION(**series_by_name)


def summation(
    cutoff: float | None = None,
    **series_by_name: pl.Series,
) -> pl.Series:
    if cutoff is None:
        return _SUMMATION(**series_by_name)
    return get_map_fn(
        "summation",
        all_required=True,
        cutoff=cutoff,
    )(**series_by_name)


def summation_allow_partial(
    cutoff: float | None = None,
    **series_by_name: pl.Series,
) -> pl.Series:
    if cutoff is None:
        return _SUMMATION_ALLOW_PARTIAL(**series_by_name)
    return get_map_fn(
        "summation",
        all_required=False,
        cutoff=cutoff,
    )(**series_by_name)


def multiply_by_group(
    *,
    scalar_factor: float | None = None,
    **kwargs: pl.Series | bool,
) -> pl.Series:
    return _MULTIPLY_BY_GROUP(
        scalar_factor=scalar_factor,
        **kwargs,
    )


def weighted_summation(
    all_required: bool = True,
    cutoff: float | None = None,
    **values_by_name: pl.Series | float,
) -> pl.Series:
    return _WEIGHTED_SUMMATION(
        all_required=all_required,
        cutoff=cutoff,
        **values_by_name,
    )


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


def coalesce_by_priority(
    priority: Sequence[str],
    **series_by_name: pl.Series,
) -> pl.Series:
    return get_map_fn("coalesce_by_priority", priority=priority)(
        **series_by_name
    )


def consolidate_lipid_value(
    lipid_enz_harm: pl.Series, lipid_enz_imp: pl.Series, lipid_imp: pl.Series
) -> pl.Series:
    return _CONSOLIDATE_LIPID_VALUE(
        lipid_enz_harm=lipid_enz_harm,
        lipid_enz_imp=lipid_enz_imp,
        lipid_imp=lipid_imp,
    )


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


def bin_decoding(
    *,
    values: pl.Series,
    **kwargs: pl.Series | float,
) -> pl.Series:
    return _BIN_DECODING(
        values=values,
        **kwargs,
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
    "lab_sensitivity_dichotomization",
    "summation",
    "summation_allow_partial",
    "multiply_by_group",
    "weighted_summation",
    "standardize",
    "standardize_creatinine",
    "normalize_specific_gravity",
    "total_lipid_concentration",
    "coalesce_by_priority",
    "consolidate_lipid_value",
    "standardize_lipid",
    "medium_bound_imputation_scalar_input",
    "medium_bound_imputation",
    "bin_decoding",
    "random_single_imputation_scalar_input",
    "random_single_imputation",
]
