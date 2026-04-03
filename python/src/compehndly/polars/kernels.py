from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Callable

import polars as pl


SeriesKernel = Callable[..., pl.Series]
ExprBuilder = Callable[..., pl.Expr]
MapFn = Callable[..., pl.Series]


@dataclass(frozen=True)
class DerivedFunctionSpec:
    """
    A backend-ready derived-variable definition.

    Contributors should only provide two function implementations:
    - kernel: operates on one or more `pl.Series` (+ optional scalar params)
    - expr_builder: operates on one or more `pl.Expr` (+ optional scalar params)
    """

    name: str
    kernel: SeriesKernel
    expr_builder: ExprBuilder


def make_map_fn(spec: DerivedFunctionSpec, **params: Any) -> MapFn:
    """
    Build `map_fn(**series_by_name) -> pl.Series` for external map-batch APIs.
    """

    kernel_sig = inspect.signature(spec.kernel)
    has_var_positional = any(
        p.kind == inspect.Parameter.VAR_POSITIONAL
        for p in kernel_sig.parameters.values()
    )

    def _map_fn(**series_by_name: pl.Series) -> pl.Series:
        if not series_by_name:
            raise ValueError("At least one input series is required")
        if has_var_positional:
            return spec.kernel(*series_by_name.values(), **params)
        return spec.kernel(**series_by_name, **params)

    return _map_fn


def apply_spec(
    spec: DerivedFunctionSpec,
    *series_or_exprs: Any,
    **params: Any,
) -> Any:
    """
    Dispatch to expr or kernel path based on input type.
    """
    data_kwargs: dict[str, Any] = {}
    scalar_kwargs: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (pl.Series, pl.Expr)):
            data_kwargs[key] = value
        else:
            scalar_kwargs[key] = value

    if series_or_exprs and data_kwargs:
        raise ValueError(
            "Do not mix positional series/expr inputs with named series/expr inputs."
        )

    kernel_sig = inspect.signature(spec.kernel)
    expr_sig = inspect.signature(spec.expr_builder)
    kernel_has_var_positional = any(
        p.kind == inspect.Parameter.VAR_POSITIONAL
        for p in kernel_sig.parameters.values()
    )
    expr_has_var_positional = any(
        p.kind == inspect.Parameter.VAR_POSITIONAL
        for p in expr_sig.parameters.values()
    )

    if data_kwargs:
        values = list(data_kwargs.values())
        if all(isinstance(v, pl.Expr) for v in values):
            if expr_has_var_positional:
                return spec.expr_builder(*values, **scalar_kwargs)
            return spec.expr_builder(**data_kwargs, **scalar_kwargs)
        if all(isinstance(v, pl.Series) for v in values):
            if kernel_has_var_positional:
                return spec.kernel(*values, **scalar_kwargs)
            return spec.kernel(**data_kwargs, **scalar_kwargs)
        raise TypeError(
            "Named data inputs must be all `pl.Series` or all `pl.Expr`."
        )

    if not series_or_exprs:
        raise ValueError("At least one input is required")

    if all(isinstance(item, pl.Expr) for item in series_or_exprs):
        return spec.expr_builder(*series_or_exprs, **scalar_kwargs)

    normalized = [
        item if isinstance(item, pl.Series) else pl.Series(item)
        for item in series_or_exprs
    ]
    return spec.kernel(*normalized, **scalar_kwargs)
