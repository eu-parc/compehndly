from __future__ import annotations

from functools import reduce
import operator

import polars as pl

from compehndly.polars.kernels import DerivedFunctionSpec


def _validate_cutoff(cutoff: float | None) -> None:
    if cutoff is None:
        return
    if not 0 <= cutoff <= 1:
        raise ValueError("cutoff must be between 0 and 1")


def summation_kernel(
    *series: pl.Series,
    all_required: bool = True,
    cutoff: float | None = None,
) -> pl.Series:
    if not series:
        raise ValueError("At least one input series is required")
    _validate_cutoff(cutoff)

    lengths = {len(s) for s in series}
    if len(lengths) != 1:
        raise ValueError("All input series must have the same length")

    length = len(series[0])
    if cutoff is not None:
        has_sufficient_values = any(
            length > 0 and (length - s.null_count()) / length >= cutoff
            for s in series
        )
        if not has_sufficient_values:
            return pl.Series(name=series[0].name, values=[None] * length)
    elif all_required and any(s.null_count() == length for s in series):
        return pl.Series([None] * length)

    result = series[0].fill_null(0)
    for s in series[1:]:
        result = result + s.fill_null(0)

    return result


def summation_expr(
    *exprs: pl.Expr,
    all_required: bool = True,
    cutoff: float | None = None,
) -> pl.Expr:
    if not exprs:
        raise ValueError("At least one input expression is required")
    _validate_cutoff(cutoff)

    result = exprs[0].fill_null(0)
    for expr in exprs[1:]:
        result = result + expr.fill_null(0)

    if cutoff is not None:
        sufficient_value_flags = [
            (expr.is_not_null().mean() >= cutoff).fill_null(False)
            for expr in exprs
        ]
        has_sufficient_values = reduce(operator.or_, sufficient_value_flags)
        return pl.when(has_sufficient_values).then(result).otherwise(None)

    if not all_required:
        return result

    entirely_null_flags = [expr.is_null().all() for expr in exprs]
    any_entirely_null = reduce(operator.or_, entirely_null_flags)
    return pl.when(~any_entirely_null).then(result).otherwise(None)


FUNCTION_SPEC = DerivedFunctionSpec(
    name="summation",
    kernel=summation_kernel,
    expr_builder=summation_expr,
)
