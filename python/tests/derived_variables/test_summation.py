import numpy as np
import polars as pl
import pytest

from compehndly import apply


@pytest.mark.derived
class TestSummation:
    def test_summation_all_required_true(self):
        a = pl.Series([1.0, None, 3.0])
        b = pl.Series([None, 5.0, 1.0])
        c = pl.Series([2.0, 2.0, None])

        out = apply("summation", a, b, c, all_required=True)
        expected = np.array([3.0, 7.0, 4.0])
        assert np.allclose(out.to_numpy(), expected, equal_nan=True)

    def test_entire_null_array_produces_all_null(self):
        a = pl.Series([1.0, 2.0, 3.0])
        b = pl.Series([None, None, None])
        c = pl.Series([1.0, 1.0, 1.0])

        out = apply("summation", a, b, c, all_required=True)
        assert out.null_count() == len(out)

    def test_nulls_treated_as_zero_when_not_all_required(self):
        a = pl.Series([1.0, None, 3.0])
        b = pl.Series([None, 2.0, None])

        out = apply("summation", a, b, all_required=False)
        expected = np.array([1.0, 2.0, 3.0])
        assert np.allclose(out.to_numpy(), expected, equal_nan=True)

    def test_length_mismatch_raises(self):
        a = pl.Series([1.0, 2.0])
        b = pl.Series([1.0])

        with pytest.raises(ValueError):
            apply("summation", a, b)

    def test_no_input_raises(self):
        with pytest.raises(ValueError):
            apply("summation")
