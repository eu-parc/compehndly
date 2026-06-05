from __future__ import annotations

from typing import Any

import polars as pl

from compehndly.polars.kernels import DerivedFunctionSpec


def _parse_factor_index(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None

    suffix = name.removeprefix(prefix)
    if not suffix.isdigit():
        raise ValueError(f"{name} must use a numeric suffix")

    return int(suffix)


def _as_bool(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise TypeError(f"{name} must be a boolean scalar")


def _parse_multiply_by_group_factors(
    kwargs: dict[str, Any],
) -> list[tuple[int, pl.Series | pl.Expr, bool]]:
    factors_by_index: dict[int, pl.Series | pl.Expr] = {}
    invert_by_index: dict[int, bool] = {}
    invalid_names: list[str] = []

    for name, value in kwargs.items():
        factor_index = _parse_factor_index(name, "factor_")
        if factor_index is not None:
            if not isinstance(value, (pl.Series, pl.Expr)):
                raise TypeError(f"{name} must be a Polars Series or Expr")
            factors_by_index[factor_index] = value
            continue

        invert_index = _parse_factor_index(name, "invert_")
        if invert_index is not None:
            invert_by_index[invert_index] = _as_bool(value, name)
            continue

        invalid_names.append(name)

    if invalid_names:
        names = ", ".join(sorted(invalid_names))
        raise ValueError(
            "Unexpected arguments for multiply_by_group: "
            f"{names}. Use factor_N/invert_N arguments."
        )

    indices = set(factors_by_index) | set(invert_by_index)
    if not indices:
        raise ValueError("At least one factor_N argument is required")

    missing_factors = sorted(indices - set(factors_by_index))
    if missing_factors:
        raise ValueError(
            "missing factor_"
            + ", factor_".join(str(index) for index in missing_factors)
        )

    expected_indices = set(range(1, max(indices) + 1))
    if indices != expected_indices:
        missing_indices = sorted(expected_indices - indices)
        raise ValueError(
            "factor_N/invert_N indices must start at 1 and be contiguous; "
            "missing indices: "
            + ", ".join(str(index) for index in missing_indices)
        )

    return [
        (index, factors_by_index[index], invert_by_index.get(index, False))
        for index in sorted(indices)
    ]


def multiply_by_group_kernel(
    *,
    scalar_factor: float | None = None,
    **kwargs: Any,
) -> pl.Series:
    """
    Multiply and divide indexed factor Series.

    Call contract:
      factor_1=<Series>, invert_1=<bool scalar>,
      factor_2=<Series>, invert_2=<bool scalar>,
      ...

    `invert_N` is optional and defaults to `False`. Rule indices must start
    at 1, be contiguous, and every index must provide `factor_N`.
    """
    factors = _parse_multiply_by_group_factors(kwargs)

    lengths = {len(factor) for _, factor, _ in factors}
    if len(lengths) != 1:
        raise ValueError("All input series must have the same length")

    result = pl.Series([1.0] * next(iter(lengths)))
    for _, factor, invert in factors:
        if invert:
            result = result / factor
        else:
            result = result * factor

    if scalar_factor is not None:
        result = result * scalar_factor

    return result


def multiply_by_group_expr(
    *,
    scalar_factor: float | None = None,
    **kwargs: Any,
) -> pl.Expr:
    """
    Multiply and divide indexed factor expressions.

    Call contract:
      factor_1=<Expr>, invert_1=<bool scalar>,
      factor_2=<Expr>, invert_2=<bool scalar>,
      ...

    `invert_N` is optional and defaults to `False`. Rule indices must start
    at 1, be contiguous, and every index must provide `factor_N`.
    """
    factors = _parse_multiply_by_group_factors(kwargs)

    result = pl.lit(1.0)
    for _, factor, invert in factors:
        if invert:
            result = result / factor
        else:
            result = result * factor

    if scalar_factor is not None:
        result = result * scalar_factor

    return result


FUNCTION_SPEC = DerivedFunctionSpec(
    name="multiply_by_group",
    kernel=multiply_by_group_kernel,
    expr_builder=multiply_by_group_expr,
)
