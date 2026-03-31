from __future__ import annotations

from functools import reduce
import operator

import polars as pl

from compehndly.polars.kernels import DerivedFunctionSpec


def summation_kernel(
    *series: pl.Series, all_required: bool = True
) -> pl.Series:
    if not series:
        raise ValueError("At least one input series is required")

    lengths = {len(s) for s in series}
    if len(lengths) != 1:
        raise ValueError("All input series must have the same length")

    length = len(series[0])
    if all_required and any(s.null_count() == length for s in series):
        return pl.Series([None] * length)

    result = series[0].fill_null(0)
    for s in series[1:]:
        result = result + s.fill_null(0)

    return result


def summation_expr(*exprs: pl.Expr, all_required: bool = True) -> pl.Expr:
    if not exprs:
        raise ValueError("At least one input expression is required")

    result = exprs[0].fill_null(0)
    for expr in exprs[1:]:
        result = result + expr.fill_null(0)

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
