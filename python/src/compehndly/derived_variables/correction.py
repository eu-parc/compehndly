from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

from compehndly.polars.kernels import DerivedFunctionSpec


def standardize_kernel(measured: pl.Series, standard: pl.Series) -> pl.Series:
    return (measured * 100) / standard


def standardize_expr(measured: pl.Expr, standard: pl.Expr) -> pl.Expr:
    return (measured * 100) / standard


def standardize_creatinine_kernel(
    measured: pl.Series, crt: pl.Series
) -> pl.Series:
    return standardize_kernel(measured, crt)


def standardize_creatinine_expr(measured: pl.Expr, crt: pl.Expr) -> pl.Expr:
    return standardize_expr(measured, crt)


def normalize_specific_gravity_kernel(
    measured: pl.Series,
    sg_measured: pl.Series,
    sg_ref: float,
) -> pl.Series:
    return (measured * (sg_ref - 1)) / sg_measured


def normalize_specific_gravity_expr(
    measured: pl.Expr,
    sg_measured: pl.Expr,
    sg_ref: float,
) -> pl.Expr:
    return (measured * (sg_ref - 1)) / sg_measured


def total_lipid_concentration_kernel(
    chol: pl.Series, trigl: pl.Series
) -> pl.Series:
    return (chol * 2.27) + trigl + 62.3


def total_lipid_concentration_expr(chol: pl.Expr, trigl: pl.Expr) -> pl.Expr:
    return (chol * 2.27) + trigl + 62.3


def _validate_priority(
    priority: Sequence[str],
    values_by_name: Mapping[str, object],
) -> tuple[str, ...]:
    if isinstance(priority, str):
        raise TypeError("priority must be a sequence of input names")
    if not priority:
        raise ValueError("priority must contain at least one input name")

    names: list[str] = []
    seen: set[str] = set()
    for name in priority:
        if not isinstance(name, str):
            raise TypeError("priority must contain only input names")
        if not name:
            raise ValueError("priority must not contain empty input names")
        if name in seen:
            raise ValueError(f"priority contains duplicate input name: {name}")
        names.append(name)
        seen.add(name)

    value_names = set(values_by_name)
    missing = [name for name in names if name not in value_names]
    if missing:
        missing_names = ", ".join(missing)
        raise ValueError(
            f"priority references unknown inputs: {missing_names}"
        )

    extra = sorted(value_names - seen)
    if extra:
        extra_names = ", ".join(extra)
        raise ValueError(f"inputs are not listed in priority: {extra_names}")

    return tuple(names)


def coalesce_by_priority_kernel(
    priority: Sequence[str],
    **series_by_name: pl.Series,
) -> pl.Series:
    priority_names = _validate_priority(priority, series_by_name)

    lengths = {len(series_by_name[name]) for name in priority_names}
    if len(lengths) != 1:
        raise ValueError("All input series must have the same length")

    result = series_by_name[priority_names[0]]
    for name in priority_names[1:]:
        result = result.fill_null(series_by_name[name])

    return result


def coalesce_by_priority_expr(
    priority: Sequence[str],
    **exprs_by_name: pl.Expr,
) -> pl.Expr:
    priority_names = _validate_priority(priority, exprs_by_name)
    return pl.coalesce([exprs_by_name[name] for name in priority_names])


def consolidate_lipid_value_kernel(
    lipid_enz_harm: pl.Series, lipid_enz_imp: pl.Series, lipid_imp: pl.Series
) -> pl.Series:
    return coalesce_by_priority_kernel(
        priority=("lipid_enz_harm", "lipid_enz_imp", "lipid_imp"),
        lipid_enz_harm=lipid_enz_harm,
        lipid_enz_imp=lipid_enz_imp,
        lipid_imp=lipid_imp,
    )


def consolidate_lipid_value_expr(
    lipid_enz_harm: pl.Expr, lipid_enz_imp: pl.Expr, lipid_imp: pl.Expr
) -> pl.Expr:
    return coalesce_by_priority_expr(
        priority=("lipid_enz_harm", "lipid_enz_imp", "lipid_imp"),
        lipid_enz_harm=lipid_enz_harm,
        lipid_enz_imp=lipid_enz_imp,
        lipid_imp=lipid_imp,
    )


def standardize_lipid_kernel(
    measured: pl.Series, lipid_value: pl.Series
) -> pl.Series:
    return standardize_kernel(measured, lipid_value)


def standardize_lipid_expr(measured: pl.Expr, lipid_value: pl.Expr) -> pl.Expr:
    return standardize_expr(measured, lipid_value)


FUNCTION_SPECS = [
    DerivedFunctionSpec(
        name="standardize",
        kernel=standardize_kernel,
        expr_builder=standardize_expr,
    ),
    DerivedFunctionSpec(
        name="standardize_creatinine",
        kernel=standardize_creatinine_kernel,
        expr_builder=standardize_creatinine_expr,
    ),
    DerivedFunctionSpec(
        name="normalize_specific_gravity",
        kernel=normalize_specific_gravity_kernel,
        expr_builder=normalize_specific_gravity_expr,
    ),
    DerivedFunctionSpec(
        name="total_lipid_concentration",
        kernel=total_lipid_concentration_kernel,
        expr_builder=total_lipid_concentration_expr,
    ),
    DerivedFunctionSpec(
        name="coalesce_by_priority",
        kernel=coalesce_by_priority_kernel,
        expr_builder=coalesce_by_priority_expr,
    ),
    DerivedFunctionSpec(
        name="consolidate_lipid_value",
        kernel=consolidate_lipid_value_kernel,
        expr_builder=consolidate_lipid_value_expr,
    ),
    DerivedFunctionSpec(
        name="standardize_lipid",
        kernel=standardize_lipid_kernel,
        expr_builder=standardize_lipid_expr,
    ),
]
