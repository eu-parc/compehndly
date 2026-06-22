from __future__ import annotations

from functools import reduce
import operator
from numbers import Real

import polars as pl

from compehndly.polars.kernels import DerivedFunctionSpec

_WEIGHT_PREFIX = "weight__"


def _validate_cutoff(cutoff: float | None) -> None:
    """Validate that an optional non-null coverage cutoff is a fraction."""
    if cutoff is None:
        return
    if not 0 <= cutoff <= 1:
        raise ValueError("cutoff must be between 0 and 1")


def _split_named_weighted_inputs(
    values_by_name: dict[str, object],
    *,
    data_type: type[pl.Series] | type[pl.Expr],
) -> tuple[dict[str, pl.Series | pl.Expr], dict[str, float]]:
    data_by_name: dict[str, pl.Series | pl.Expr] = {}
    weights_by_name: dict[str, float] = {}
    unsupported_names: list[str] = []

    for name, value in values_by_name.items():
        if isinstance(value, data_type):
            data_by_name[name] = value
            continue

        if isinstance(value, Real) and not isinstance(value, bool):
            if not name.startswith(_WEIGHT_PREFIX):
                unsupported_names.append(name)
                continue
            input_name = name.removeprefix(_WEIGHT_PREFIX)
            if not input_name:
                unsupported_names.append(name)
                continue
            weights_by_name[input_name] = float(value)
            continue

        unsupported_names.append(name)

    if unsupported_names:
        raise TypeError(
            "weighted_summation accepts only series/expressions and numeric "
            f"weights named '{_WEIGHT_PREFIX}<input_name>'; unsupported "
            f"inputs: {', '.join(unsupported_names)}"
        )

    if not data_by_name:
        raise ValueError("At least one input series/expression is required")

    missing_weights = sorted(set(data_by_name) - set(weights_by_name))
    if missing_weights:
        raise ValueError(
            "inputs are missing weights: "
            + ", ".join(
                f"{name} expects {_WEIGHT_PREFIX}{name}"
                for name in missing_weights
            )
        )

    unknown_weights = sorted(set(weights_by_name) - set(data_by_name))
    if unknown_weights:
        raise ValueError(
            "weights reference unknown inputs: "
            + ", ".join(f"{_WEIGHT_PREFIX}{name}" for name in unknown_weights)
        )

    return data_by_name, weights_by_name


def summation_kernel(
    *series: pl.Series,
    all_required: bool = True,
    cutoff: float | None = None,
) -> pl.Series:
    """
    Sum eager Polars series while treating row-level nulls as zero.

    When `cutoff` is provided, each input series is checked for its fraction
    of non-null values. The sum is returned if at least one series has a
    non-null fraction greater than or equal to `cutoff`; otherwise an all-null
    result is returned. This cutoff rule takes precedence over `all_required`.

    When `cutoff` is not provided and `all_required` is true, an entirely-null
    input series makes the full result null. Otherwise, partial sums are
    allowed and null values contribute zero.
    """
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
    """
    Build a lazy Polars expression equivalent to `summation_kernel`.

    With `cutoff`, each expression is evaluated over the current Polars
    context for its non-null fraction. The summed expression is emitted when
    at least one expression meets or exceeds the cutoff; otherwise the result
    is null. This mirrors the eager kernel's "any input passes" behavior and
    overrides `all_required`.
    """
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


def weighted_summation_kernel(
    all_required: bool = True,
    cutoff: float | None = None,
    **values_by_name: pl.Series | float,
) -> pl.Series:
    """
    Sum eager Polars series after multiplying each by its paired weight.

    Callers provide all inputs as named kwargs. Series kwargs provide the data,
    and numeric weight kwargs must be named `weight__<series_name>`, for
    example `a=series_a` and `weight__a=0.2`. This keeps the pairing based on
    names, not keyword order. Row-level nulls contribute zero after weighting.

    The `all_required` and `cutoff` gates match `summation_kernel`: with a
    cutoff, at least one input series must have a non-null fraction greater
    than or equal to the cutoff; otherwise an all-null result is returned.
    Without a cutoff, `all_required=True` returns an all-null result if any
    input series is entirely null.
    """
    series_by_name, weights_by_name = _split_named_weighted_inputs(
        values_by_name,
        data_type=pl.Series,
    )
    series = tuple(series_by_name.values())

    lengths = {len(s) for s in series}
    if len(lengths) != 1:
        raise ValueError("All input series must have the same length")

    length = len(series[0])
    if cutoff is not None:
        _validate_cutoff(cutoff)
        has_sufficient_values = any(
            length > 0 and (length - s.null_count()) / length >= cutoff
            for s in series
        )
        if not has_sufficient_values:
            return pl.Series(name=series[0].name, values=[None] * length)
    elif all_required and any(s.null_count() == length for s in series):
        return pl.Series([None] * length)

    result = None
    for name, s in series_by_name.items():
        weighted = s.fill_null(0) * weights_by_name[name]
        result = weighted if result is None else result + weighted

    return result


def weighted_summation_expr(
    all_required: bool = True,
    cutoff: float | None = None,
    **values_by_name: pl.Expr | float,
) -> pl.Expr:
    """
    Build a lazy expression for named weighted summation.

    Expression kwargs provide the data, and numeric weight kwargs must be
    named `weight__<expression_name>`. The pairing is based on those names,
    not keyword order. `all_required` and `cutoff` have the same meaning as in
    `weighted_summation_kernel` and `summation_expr`.
    """
    exprs_by_name, weights_by_name = _split_named_weighted_inputs(
        values_by_name,
        data_type=pl.Expr,
    )
    exprs = tuple(exprs_by_name.values())
    _validate_cutoff(cutoff)

    result = None
    for name, expr in exprs_by_name.items():
        weighted = expr.fill_null(0) * weights_by_name[name]
        result = weighted if result is None else result + weighted

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


FUNCTION_SPECS = (
    DerivedFunctionSpec(
        name="summation",
        kernel=summation_kernel,
        expr_builder=summation_expr,
    ),
    DerivedFunctionSpec(
        name="weighted_summation",
        kernel=weighted_summation_kernel,
        expr_builder=weighted_summation_expr,
    ),
)
