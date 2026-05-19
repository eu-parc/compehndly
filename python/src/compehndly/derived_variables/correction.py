from __future__ import annotations

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


def consolidate_lipid_value_kernel(
    lipid_enz_harm: pl.Series, lipid_enz_imp: pl.Series, lipid_imp: pl.Series
) -> pl.Series:
    length = len(lipid_enz_harm)
    if len(lipid_enz_imp) != length:
        raise ValueError(
            "lipid_enz_harm and lipid_enz_imp must have the same length"
        )
    if len(lipid_imp) != length:
        raise ValueError(
            "lipid_enz_harm and lipid_imp must have the same length"
        )

    return lipid_enz_harm.fill_null(lipid_enz_imp).fill_null(lipid_imp)


def consolidate_lipid_value_expr(
    lipid_enz_harm: pl.Expr, lipid_enz_imp: pl.Expr, lipid_imp: pl.Expr
) -> pl.Expr:
    return lipid_enz_harm.fill_null(lipid_enz_imp).fill_null(lipid_imp)


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
