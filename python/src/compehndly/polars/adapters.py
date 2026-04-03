from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

from compehndly.polars.kernels import DerivedFunctionSpec

Frame = pl.DataFrame | pl.LazyFrame


def apply_to_dataframe(
    frame: pl.DataFrame,
    *,
    spec: DerivedFunctionSpec,
    input_columns: Sequence[str] | Mapping[str, str],
    **params: Any,
) -> pl.Series:
    """
    Apply a function spec to an eager Polars DataFrame and return the derived series.
    """
    if not input_columns:
        raise ValueError("At least one input column is required")

    if isinstance(input_columns, Mapping):
        return spec.kernel(
            **{
                arg_name: frame[col_name]
                for arg_name, col_name in input_columns.items()
            },
            **params,
        )

    return spec.kernel(*[frame[col] for col in input_columns], **params)


def with_derived_column(
    frame: Frame,
    *,
    spec: DerivedFunctionSpec,
    output_column: str,
    input_columns: Sequence[str] | Mapping[str, str],
    **params: Any,
) -> Frame:
    """
    Add a derived column for Polars eager/lazy frames using the correct execution path.
    """
    if not input_columns:
        raise ValueError("At least one input column is required")

    if isinstance(frame, pl.LazyFrame):
        if isinstance(input_columns, Mapping):
            expr = spec.expr_builder(
                **{
                    arg_name: pl.col(col_name)
                    for arg_name, col_name in input_columns.items()
                },
                **params,
            ).alias(output_column)
        else:
            expr = spec.expr_builder(
                *[pl.col(c) for c in input_columns], **params
            ).alias(output_column)
        return frame.with_columns(expr)

    if isinstance(frame, pl.DataFrame):
        series = apply_to_dataframe(
            frame,
            spec=spec,
            input_columns=input_columns,
            **params,
        ).alias(output_column)
        return frame.with_columns(series)

    raise TypeError(
        "Unsupported frame type. Expected Polars DataFrame or Polars LazyFrame."
    )
