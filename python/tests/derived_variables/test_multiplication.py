import numpy as np
import polars as pl
import pytest

from compehndly import apply
from compehndly.entrypoints import multiply_by_group


@pytest.mark.derived
class TestMultiplication:
    def test_multiply_by_group_multiplies_and_divides_named_inputs(self):
        a = pl.Series([2.0, 4.0, 6.0])
        b = pl.Series([3.0, 5.0, 7.0])
        c = pl.Series([2.0, 10.0, 3.0])

        out = apply(
            "multiply_by_group",
            factor_1=a,
            factor_2=b,
            factor_3=c,
            invert_3=True,
        )

        expected = np.array([3.0, 2.0, 14.0])
        assert np.allclose(out.to_numpy(), expected, equal_nan=True)

    def test_multiply_by_group_allows_only_inverted_factors(self):
        divisor = pl.Series([2.0, 4.0])

        out = apply(
            "multiply_by_group",
            factor_1=divisor,
            invert_1=True,
        )

        expected = np.array([0.5, 0.25])
        assert np.allclose(out.to_numpy(), expected, equal_nan=True)

    def test_multiply_by_group_applies_scalar_factor(self):
        out = apply(
            "multiply_by_group",
            factor_1=pl.Series([2.0, 4.0]),
            factor_2=pl.Series([4.0, 2.0]),
            invert_2=True,
            scalar_factor=100.0,
        )

        expected = np.array([50.0, 200.0])
        assert np.allclose(out.to_numpy(), expected, equal_nan=True)

    def test_multiply_by_group_ignores_none_scalar_factor(self):
        out = apply(
            "multiply_by_group",
            factor_1=pl.Series([2.0, 4.0]),
            factor_2=pl.Series([4.0, 2.0]),
            invert_2=True,
            scalar_factor=None,
        )

        expected = np.array([0.5, 2.0])
        assert np.allclose(out.to_numpy(), expected, equal_nan=True)

    def test_multiply_by_group_interleaves_inverse_factors(self):
        out = apply(
            "multiply_by_group",
            factor_1=pl.Series([1e200]),
            factor_2=pl.Series([1e200]),
            factor_3=pl.Series([1e200]),
            invert_2=True,
        )

        assert np.isfinite(out[0])
        assert out[0] == 1e200

    def test_multiply_by_group_supports_lazy_expressions(self):
        df = pl.DataFrame(
            {
                "a": [2.0, 4.0],
                "b": [3.0, 5.0],
                "c": [2.0, 10.0],
            }
        )

        out = df.lazy().select(
            apply(
                "multiply_by_group",
                factor_1=pl.col("a"),
                factor_2=pl.col("b"),
                factor_3=pl.col("c"),
                invert_3=True,
                scalar_factor=10.0,
            ).alias("multiplied")
        )

        assert out.collect()["multiplied"].to_list() == [30.0, 20.0]

    def test_multiply_by_group_entrypoint_path(self):
        out = multiply_by_group(
            factor_1=pl.Series([2.0, 4.0]),
            factor_2=pl.Series([3.0, 5.0]),
            factor_3=pl.Series([2.0, 10.0]),
            invert_3=True,
            scalar_factor=10.0,
        )

        assert out.to_list() == [30.0, 20.0]

    def test_multiply_by_group_rejects_unexpected_arguments(self):
        with pytest.raises(ValueError, match="Unexpected arguments"):
            apply(
                "multiply_by_group",
                factor_1=pl.Series([1.0]),
                factors=("factor_1",),
            )

    def test_multiply_by_group_rejects_invert_without_factor(self):
        with pytest.raises(ValueError, match="missing factor_2"):
            apply(
                "multiply_by_group",
                factor_1=pl.Series([1.0]),
                invert_2=True,
            )

    def test_multiply_by_group_rejects_non_contiguous_indices(self):
        with pytest.raises(ValueError, match="contiguous"):
            apply(
                "multiply_by_group",
                factor_2=pl.Series([1.0]),
            )

    def test_multiply_by_group_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            apply(
                "multiply_by_group",
                factor_1=pl.Series([1.0, 2.0]),
                factor_2=pl.Series([1.0]),
            )

    def test_multiply_by_group_requires_at_least_one_factor(self):
        with pytest.raises(ValueError, match="At least one factor_N"):
            multiply_by_group(
                scalar_factor=10.0,
            )
