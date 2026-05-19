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

    def test_cutoff_allows_sum_when_one_series_has_enough_values(self):
        a = pl.Series([1.0, None, None, None, None])
        b = pl.Series([None, 2.0, 3.0, 4.0, None])
        c = pl.Series([1.0, 1.0, None, None, None])

        out = apply("summation", a, b, c, cutoff=0.6)
        expected = np.array([2.0, 3.0, 3.0, 4.0, 0.0])
        assert np.allclose(out.to_numpy(), expected, equal_nan=True)

    def test_cutoff_produces_all_null_when_no_series_has_enough_values(self):
        a = pl.Series([1.0, None, None, None, None])
        b = pl.Series([None, 2.0, 3.0, None, None])

        out = apply("summation", a, b, cutoff=0.6)
        assert out.null_count() == len(out)

    def test_cutoff_applies_when_partial_sums_are_allowed(self):
        a = pl.Series([1.0, None, None, None, None])
        b = pl.Series([None, 2.0, 3.0, None, None])

        out = apply("summation", a, b, all_required=False, cutoff=0.6)
        assert out.null_count() == len(out)

    def test_cutoff_supports_lazy_expressions(self):
        df = pl.DataFrame(
            {
                "a": [1.0, None, None, None, None],
                "b": [None, 2.0, 3.0, 4.0, None],
            }
        )

        out = df.lazy().select(
            apply(
                "summation",
                pl.col("a"),
                pl.col("b"),
                cutoff=0.6,
            ).alias("sum_col")
        )

        assert out.collect()["sum_col"].to_list() == [
            1.0,
            2.0,
            3.0,
            4.0,
            0.0,
        ]

    @pytest.mark.parametrize("cutoff", [-0.1, 1.1])
    def test_invalid_cutoff_raises(self, cutoff):
        with pytest.raises(ValueError, match="cutoff"):
            apply("summation", pl.Series([1.0]), cutoff=cutoff)

    def test_length_mismatch_raises(self):
        a = pl.Series([1.0, 2.0])
        b = pl.Series([1.0])

        with pytest.raises(ValueError):
            apply("summation", a, b)

    def test_no_input_raises(self):
        with pytest.raises(ValueError):
            apply("summation")
