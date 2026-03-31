from __future__ import annotations

import polars as pl

from compehndly.api import get_map_fn

# Stable, path-addressable entrypoints for config-based loaders.
# These wrappers intentionally have the map_batches contract:
#   (**series_by_name) -> pl.Series

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
_RANDOM_SINGLE_IMPUTATION = get_map_fn("random_single_imputation")


def summation(**series_by_name: pl.Series) -> pl.Series:
    return _SUMMATION(**series_by_name)


def summation_allow_partial(**series_by_name: pl.Series) -> pl.Series:
    return _SUMMATION_ALLOW_PARTIAL(**series_by_name)


def standardize(**series_by_name: pl.Series) -> pl.Series:
    return _STANDARDIZE(**series_by_name)


def standardize_creatinine(**series_by_name: pl.Series) -> pl.Series:
    return _STANDARDIZE_CREATININE(**series_by_name)


def normalize_specific_gravity(**series_by_name: pl.Series) -> pl.Series:
    return _NORMALIZE_SPECIFIC_GRAVITY(**series_by_name)


def total_lipid_concentration(**series_by_name: pl.Series) -> pl.Series:
    return _TOTAL_LIPID_CONCENTRATION(**series_by_name)


def standardize_lipid(**series_by_name: pl.Series) -> pl.Series:
    return _STANDARDIZE_LIPID(**series_by_name)


def medium_bound_imputation_scalar_input(
    **series_by_name: pl.Series,
) -> pl.Series:
    return _MEDIUM_BOUND_IMPUTATION_SCALAR_INPUT(**series_by_name)


def medium_bound_imputation(**series_by_name: pl.Series) -> pl.Series:
    return _MEDIUM_BOUND_IMPUTATION(**series_by_name)


def random_single_imputation(**series_by_name: pl.Series) -> pl.Series:
    return _RANDOM_SINGLE_IMPUTATION(**series_by_name)


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
    "random_single_imputation",
]
