from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import polars as pl


def _flatten_mapping(
    groups: dict[Hashable, list[Hashable]],
) -> dict[Hashable, Hashable]:
    mapping: dict[Hashable, Hashable] = {}
    for new_label, old_values in groups.items():
        for old in old_values:
            mapping[old] = new_label
    return mapping


def bin_categorical(
    data: pl.Series | pl.Expr,
    groups: dict[Hashable, list[Hashable]],
    default: Hashable | None = None,
) -> pl.Series | pl.Expr:
    """
    Bin categorical values by grouping categories into new labels.

    Supports both eager `pl.Series` and lazy `pl.Expr` inputs.
    """
    mapping = _flatten_mapping(groups)

    if isinstance(data, pl.Series):
        out = [mapping.get(value, default) for value in data.to_list()]
        return pl.Series(name=data.name, values=out)

    if isinstance(data, pl.Expr):
        result: pl.Expr = pl.lit(default)
        for old_value, new_label in mapping.items():
            result = (
                pl.when(data == old_value)
                .then(pl.lit(new_label))
                .otherwise(result)
            )
        return result

    raise TypeError(
        "Unsupported input type. Expected Polars Series or Polars Expr."
    )


def bin_numeric(
    data: pl.Series | pl.Expr,
    boundaries: Sequence[float],
    labels: Sequence[Any],
    right_inclusive: bool = False,
) -> pl.Series | pl.Expr:
    """
    Bin numeric values into intervals defined by boundaries.

    Supports both eager `pl.Series` and lazy `pl.Expr` inputs.
    """
    if len(labels) != len(boundaries) - 1:
        raise ValueError("labels must have length len(boundaries) - 1")

    if isinstance(data, pl.Series):
        values = data.to_list()
        out = [None] * len(values)

        for idx, value in enumerate(values):
            if value is None:
                continue

            for i, label in enumerate(labels):
                lower = boundaries[i]
                upper = boundaries[i + 1]

                if right_inclusive:
                    in_bin = (value > lower) and (value <= upper)
                else:
                    in_bin = (value >= lower) and (value < upper)

                if in_bin:
                    out[idx] = label
                    break

        return pl.Series(name=data.name, values=out)

    if isinstance(data, pl.Expr):
        result: pl.Expr = pl.lit(None)

        for i, label in enumerate(labels):
            lower = boundaries[i]
            upper = boundaries[i + 1]

            if right_inclusive:
                mask = (data > lower) & (data <= upper)
            else:
                mask = (data >= lower) & (data < upper)

            result = pl.when(mask).then(pl.lit(label)).otherwise(result)

        return result

    raise TypeError(
        "Unsupported input type. Expected Polars Series or Polars Expr."
    )
