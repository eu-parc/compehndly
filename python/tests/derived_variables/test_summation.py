import numpy as np
import polars as pl
import pytest

from compehndly import apply
from compehndly.entrypoints import weighted_summation


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


@pytest.mark.derived
class TestWeightedSummation:
    def test_weighted_summation_multiplies_named_inputs_by_weights(self):
        out = apply(
            "weighted_summation",
            a=pl.Series([1.0, None, 3.0]),
            b=pl.Series([10.0, 20.0, None]),
            weight__a=2.0,
            weight__b=0.5,
        )

        expected = np.array([7.0, 10.0, 6.0])
        assert np.allclose(out.to_numpy(), expected, equal_nan=True)

    def test_weighted_summation_entrypoint_path(self):
        out = weighted_summation(
            a=pl.Series([1.0, None, 3.0]),
            b=pl.Series([10.0, 20.0, None]),
            weight__a=2.0,
            weight__b=0.5,
        )

        assert out.to_list() == [7.0, 10.0, 6.0]

    def test_weighted_summation_respects_all_required(self):
        out = apply(
            "weighted_summation",
            a=pl.Series([1.0, 2.0]),
            b=pl.Series([None, None]),
            weight__a=1.0,
            weight__b=1.0,
            all_required=True,
        )

        assert out.null_count() == len(out)

    def test_weighted_summation_pairs_weights_by_name_not_order(self):
        out = apply(
            "weighted_summation",
            weight__b=0.5,
            a=pl.Series([1.0, None, 3.0]),
            weight__a=2.0,
            b=pl.Series([10.0, 20.0, None]),
        )

        expected = np.array([7.0, 10.0, 6.0])
        assert np.allclose(out.to_numpy(), expected, equal_nan=True)

    def test_weighted_summation_cutoff_uses_any_input_series(self):
        out = apply(
            "weighted_summation",
            a=pl.Series([1.0, None, None, None, None]),
            b=pl.Series([None, 2.0, 3.0, 4.0, None]),
            weight__a=10.0,
            weight__b=1.0,
            cutoff=0.6,
        )

        expected = np.array([10.0, 2.0, 3.0, 4.0, 0.0])
        assert np.allclose(out.to_numpy(), expected, equal_nan=True)

    def test_weighted_summation_cutoff_returns_null_when_no_input_passes(self):
        out = apply(
            "weighted_summation",
            a=pl.Series([1.0, None, None, None, None]),
            b=pl.Series([None, 2.0, 3.0, None, None]),
            weight__a=10.0,
            weight__b=1.0,
            cutoff=0.6,
        )

        assert out.null_count() == len(out)

    def test_weighted_summation_supports_lazy_expressions(self):
        df = pl.DataFrame(
            {
                "a": [1.0, None, 3.0],
                "b": [10.0, 20.0, None],
            }
        )

        out = df.lazy().select(
            apply(
                "weighted_summation",
                a=pl.col("a"),
                b=pl.col("b"),
                weight__a=2.0,
                weight__b=0.5,
            ).alias("weighted")
        )

        assert out.collect()["weighted"].to_list() == [7.0, 10.0, 6.0]

    def test_weighted_summation_rejects_missing_weight(self):
        with pytest.raises(ValueError, match="missing weights"):
            apply(
                "weighted_summation",
                a=pl.Series([1.0]),
                b=pl.Series([2.0]),
                weight__a=1.0,
            )

    def test_weighted_summation_rejects_unknown_weight(self):
        with pytest.raises(ValueError, match="unknown inputs"):
            apply(
                "weighted_summation",
                a=pl.Series([1.0]),
                weight__a=1.0,
                weight__missing=1.0,
            )

    def test_weighted_summation_rejects_weight_without_prefix(self):
        with pytest.raises(TypeError, match="weight__"):
            apply(
                "weighted_summation",
                a=pl.Series([1.0]),
                weight_a=1.0,
            )

    def test_weighted_summation_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            apply(
                "weighted_summation",
                a=pl.Series([1.0, 2.0]),
                b=pl.Series([1.0]),
                weight__a=1.0,
                weight__b=1.0,
            )
