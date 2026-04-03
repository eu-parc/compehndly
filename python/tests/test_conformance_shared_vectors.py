import builtins
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from compehndly import apply


@pytest.mark.conformance
class TestSharedConformanceVectors:
    @staticmethod
    def _load_cases():
        root = Path(__file__).resolve().parents[2]
        path = root / "shared" / "conformance" / "derived_variables_cases.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["cases"]

    @staticmethod
    def _series(values):
        return pl.Series(values)

    def _run_case(self, case):
        function_name = case["function"]
        params = case.get("params", {})
        input_spec = case.get("input", {})

        positional = input_spec.get("positional")
        named = input_spec.get("named")

        if positional is not None and named is not None:
            raise ValueError(
                f"Case {case['id']} mixes positional and named input specs"
            )

        def _invoke():
            if positional is not None:
                series = [self._series(values) for values in positional]
                return apply(function_name, *series, **params)

            if named is not None:
                named_series = {k: self._series(v) for k, v in named.items()}
                return apply(function_name, **named_series, **params)

            return apply(function_name, **params)

        expected_error = case.get("expect_error")
        if expected_error:
            err_type_name = expected_error["type"]
            err_type = getattr(builtins, err_type_name, ValueError)
            with pytest.raises(err_type):
                _invoke()
            return

        out = _invoke()
        assertions = case.get("assertions", {})

        if assertions.get("all_null"):
            assert out.null_count() == len(out), case["id"]

        expected = assertions.get("expected")
        if expected is not None:
            out_np = out.to_numpy()
            expected_np = np.array(
                [np.nan if x is None else x for x in expected], dtype=float
            )
            assert np.allclose(out_np, expected_np, equal_nan=True), case["id"]

        if assertions.get("non_negative"):
            out_np = out.to_numpy()
            assert np.all(out_np >= 0), case["id"]

        if assertions.get("no_nan"):
            out_np = out.to_numpy()
            assert not np.any(np.isnan(out_np)), case["id"]

        equals_at_indices = assertions.get("equals_at_indices", {})
        if equals_at_indices:
            out_np = out.to_numpy()
            for idx_str, value in equals_at_indices.items():
                idx = int(idx_str)
                assert out_np[idx] == value, case["id"]

        ranges_at_indices = assertions.get("ranges_at_indices", {})
        if ranges_at_indices:
            out_np = out.to_numpy()
            for idx_str, bounds in ranges_at_indices.items():
                idx = int(idx_str)
                lower, upper = bounds
                assert lower <= out_np[idx] <= upper, case["id"]

        min_value_from_index = assertions.get("min_value_from_index")
        if min_value_from_index:
            start = int(min_value_from_index["start"])
            value = float(min_value_from_index["value"])
            out_np = out.to_numpy()
            assert np.all(out_np[start:] >= value), case["id"]

    def test_all_shared_cases(self):
        for case in self._load_cases():
            self._run_case(case)
