import pytest
import numpy as np
import pyarrow as pa

# Import the functions to test
from compehndly.derived_variables.summation import (
    _summation_v0_0_1_reference,
    _summation_v0_0_1_arrow,
)


def arrow_value_to_py(arr, i):
    return None if arr.is_valid() is False else arr[i].as_py()


def run_reference_rowwise(*arrays, all_required):
    """
    Convert Arrow columns to Python scalars and feed them row-wise
    to the reference implementation.
    """
    length = len(arrays[0])
    out = []
    for i in range(length):
        row_vals = [arrow_value_to_py(arr, i) for arr in arrays]
        out.append(_summation_v0_0_1_reference(*row_vals, all_required=all_required))
    return out


@pytest.mark.core
class TestSummation:
    def test_summation_matches_reference(self):
        a = pa.array([1.0, None, 3.0])
        b = pa.array([None, 5.0, 1.0])
        c = pa.array([2.0, 2.0, None])

        out_arrow = _summation_v0_0_1_arrow(a, b, c, all_required=True)
        out_ref = run_reference_rowwise(a, b, c, all_required=True)

        assert len(out_arrow) == len(out_ref)

        # Compare row-by-row, converting Arrow → Python
        for i in range(len(out_ref)):
            v_arrow = arrow_value_to_py(out_arrow, i)
            v_ref = out_ref[i]
            assert (v_arrow is None and v_ref is None) or np.isclose(v_arrow, v_ref)

    def test_entire_null_array_produces_all_null(self):
        a = pa.array([1.0, 2.0, 3.0])
        b = pa.array([None, None, None])  # entirely null
        c = pa.array([1.0, 1.0, 1.0])

        out = _summation_v0_0_1_arrow(a, b, c, all_required=True)

        # Should be completely null
        assert out.null_count == len(out)
        assert not all(out.is_valid())

    def test_nulls_treated_as_zero_when_not_all_required(self):
        a = pa.array([1.0, None, 3.0])
        b = pa.array([None, 2.0, None])

        out_arrow = _summation_v0_0_1_arrow(a, b, all_required=False)
        out_np = np.array([v if v is not None else np.nan for v in out_arrow.to_pylist()])

        # Manual expected:
        # Row 0: 1 + 0 = 1
        # Row 1: 0 + 2 = 2
        # Row 2: 3 + 0 = 3
        expected = np.array([1.0, 2.0, 3.0])

        assert np.allclose(out_np, expected, equal_nan=False)

    def test_length_mismatch_raises(self):
        a = pa.array([1.0, 2.0])
        b = pa.array([1.0])

        with pytest.raises(ValueError):
            _summation_v0_0_1_arrow(a, b)

    def test_no_input_raises(self):
        with pytest.raises(ValueError):
            _summation_v0_0_1_arrow()
