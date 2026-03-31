from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
import importlib
import pkgutil
from typing import Any

import polars as pl

from compehndly.polars.kernels import (
    DerivedFunctionSpec,
    MapFn,
    apply_spec,
    make_map_fn,
)
from compehndly.polars.adapters import (
    with_derived_column as _with_derived_column,
)


@lru_cache(maxsize=1)
def _discover_specs() -> dict[str, DerivedFunctionSpec]:
    registry: dict[str, DerivedFunctionSpec] = {}
    package = importlib.import_module("compehndly.derived_variables")

    for module in pkgutil.iter_modules(
        package.__path__, package.__name__ + "."
    ):
        mod = importlib.import_module(module.name)
        single_spec = getattr(mod, "FUNCTION_SPEC", None)
        if single_spec is not None:
            if single_spec.name in registry:
                raise ValueError(
                    f"Duplicate function name discovered: {single_spec.name}"
                )
            registry[single_spec.name] = single_spec

        multiple_specs = getattr(mod, "FUNCTION_SPECS", None)
        if multiple_specs is not None:
            for spec in multiple_specs:
                if spec.name in registry:
                    raise ValueError(
                        f"Duplicate function name discovered: {spec.name}"
                    )
                registry[spec.name] = spec

    return registry


def list_functions() -> list[str]:
    return sorted(_discover_specs().keys())


def get_map_fn(function_name: str, **params: Any) -> MapFn:
    specs = _discover_specs()
    if function_name not in specs:
        available = ", ".join(sorted(specs.keys()))
        raise KeyError(
            f"Unknown function '{function_name}'. Available: {available}"
        )
    return make_map_fn(specs[function_name], **params)


def apply(
    function_name: str,
    *series_or_exprs: Any,
    **params: Any,
) -> Any:
    """
    Apply a function spec directly on one or more `pl.Series` or `pl.Expr` inputs.
    """
    specs = _discover_specs()
    if function_name not in specs:
        available = ", ".join(sorted(specs.keys()))
        raise KeyError(
            f"Unknown function '{function_name}'. Available: {available}"
        )

    return apply_spec(specs[function_name], *series_or_exprs, **params)


def with_derived_column(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    function_name: str,
    input_columns: Sequence[str] | Mapping[str, str],
    output_column: str,
    **params: Any,
) -> pl.DataFrame | pl.LazyFrame:
    specs = _discover_specs()
    if function_name not in specs:
        available = ", ".join(sorted(specs.keys()))
        raise KeyError(
            f"Unknown function '{function_name}'. Available: {available}"
        )

    return _with_derived_column(
        frame,
        spec=specs[function_name],
        output_column=output_column,
        input_columns=input_columns,
        **params,
    )
