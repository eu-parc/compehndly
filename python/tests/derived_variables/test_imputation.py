import numpy as np
import polars as pl
import pytest

from compehndly import apply


@pytest.mark.derived
class TestImputation:
    def test_lab_sensitivity_dichotomization_basic(self):
        df = pl.DataFrame(
            {
                "measurement": [-1.0, -2.0, -3.0, 1.0, 6.0, 7.0, 8.0],
                "lod": [2.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                "loq": [4.0, 5.0, 6.0, 4.0, 4.0, 4.0, 4.0],
            }
        )

        expr = apply(
            "lab_sensitivity_dichotomization",
            measurement=pl.col("measurement"),
            lod=pl.col("lod"),
            loq=pl.col("loq"),
        )
        out = df.lazy().select(expr.alias("imputed")).collect()["imputed"]

        out_np = out.to_numpy()
        assert out_np.tolist() == [True, True, True, True, False, False, False]

    def test_random_single_imputation_basic(self):
        biomarker = pl.Series([5.0, -1.0, -2.0, 10.0, -3.0, 8.0])
        lod = 2.0
        loq = 4.0

        out = apply(
            "random_single_imputation_scalar_input",
            biomarker,
            lod=lod,
            loq=loq,
            min_unique_values=3,
            min_observed_percentage=50,
            seed=123,
        )

        out_np = out.to_numpy()
        assert not np.any(
            out_np < 0
        ), "Censored values were not properly imputed."
        assert out_np[0] == 5.0
        assert out_np[3] == 10.0
        assert out_np[5] == 8.0

    def test_random_single_imputation_bounds_respected(self):
        lod = 2.0
        loq = 4.0

        rng = np.random.default_rng(7)
        above_loq = rng.lognormal(size=100) + loq
        biomarker = above_loq.copy()
        biomarker[0:3] = np.array([-1.0, -2.0, -3.0])

        out = apply(
            "random_single_imputation_scalar_input",
            pl.Series(biomarker),
            lod=lod,
            loq=loq,
            min_unique_values=1,
            min_observed_percentage=30,
            seed=42,
        )

        out_np = out.to_numpy()
        imputed = out_np[:3]

        assert not np.any(np.isnan(imputed)), "Imputation produced NaNs."
        assert 0 <= imputed[0] <= 2.0
        assert 2.0 <= imputed[1] <= 4.0
        assert 0 <= imputed[2] <= 4.0
        assert np.all(out_np[3:] >= loq)

    def test_random_single_imputation_accepts_lod_loq_series(self):
        rng = np.random.default_rng(7)
        biomarker = rng.lognormal(size=100) + 5.0
        biomarker[0:3] = np.array([-1.0, -2.0, -3.0])

        lod = np.full(100, 2.0)
        loq = np.full(100, 4.0)
        lod[1] = 3.0
        loq[1] = 5.0
        loq[2] = 6.0

        out = apply(
            "random_single_imputation",
            biomarker=pl.Series(biomarker),
            lod=pl.Series(lod),
            loq=pl.Series(loq),
            min_unique_values=1,
            min_observed_percentage=30,
            seed=42,
        )

        out_np = out.to_numpy()
        imputed = out_np[:3]

        assert not np.any(np.isnan(imputed)), "Imputation produced NaNs."
        assert 0 <= imputed[0] <= 2.0
        assert 3.0 <= imputed[1] <= 5.0
        assert 0 <= imputed[2] <= 6.0
        assert np.all(out_np[3:] >= 5.0)

    def test_random_single_imputation_expr_accepts_lod_loq_series(self):
        df = pl.DataFrame(
            {
                "biomarker": [-1.0, -2.0, -3.0, 5.0, 6.0, 7.0, 8.0],
                "lod": [2.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                "loq": [4.0, 5.0, 6.0, 4.0, 4.0, 4.0, 4.0],
            }
        )

        expr = apply(
            "random_single_imputation",
            biomarker=pl.col("biomarker"),
            lod=pl.col("lod"),
            loq=pl.col("loq"),
            seed=42,
        )
        out = df.lazy().select(expr.alias("imputed")).collect()["imputed"]

        out_np = out.to_numpy()
        assert 0 <= out_np[0] <= 2.0
        assert 3.0 <= out_np[1] <= 5.0
        assert 0 <= out_np[2] <= 6.0
        assert out_np[3:].tolist() == [5.0, 6.0, 7.0, 8.0]

    def test_random_single_imputation_insufficient_observed_values(self):
        biomarker = pl.Series([5.0, -1.0, -2.0, -1.0, -3.0, -2.0])
        lod = 2.0
        loq = 4.0

        out = apply(
            "random_single_imputation_scalar_input",
            biomarker,
            lod=lod,
            loq=loq,
            min_observed_percentage=30,
            seed=123,
        )

        out_np = out.to_numpy()
        assert np.all(
            np.isnan(out_np)
        ), "Failing check: all values should be NaN."

    def test_random_single_imputation_insufficient_unique_values(self):
        biomarker = pl.Series([5.0, -1.0, -2.0, 8.0, -3.0, 8.0])
        lod = 2.0
        loq = 4.0

        out = apply(
            "random_single_imputation_scalar_input",
            biomarker,
            lod=lod,
            loq=loq,
            min_unique_values=3,
            seed=123,
        )

        out_np = out.to_numpy()
        assert np.all(
            np.isnan(out_np)
        ), "Failing check: all values should be NaN."

    def test_random_single_imputation_requires_matching_threshold_lengths(
        self,
    ):
        with pytest.raises(ValueError, match="same length"):
            apply(
                "random_single_imputation",
                biomarker=pl.Series([5.0, -1.0]),
                lod=pl.Series([2.0]),
                loq=pl.Series([4.0, 4.0]),
            )

    def test_random_single_imputation_rejects_invalid_series_thresholds(self):
        with pytest.raises(ValueError, match="lod values must be > 0"):
            apply(
                "random_single_imputation",
                biomarker=pl.Series([5.0, -1.0]),
                lod=pl.Series([2.0, 4.0]),
                loq=pl.Series([4.0, 4.0]),
            )

    def test_medium_bound_imputation_scalar_input(self):
        measurement = pl.Series([0.2, 1.2, 2.5])

        out = apply(
            "medium_bound_imputation_scalar_input",
            measurement,
            loq=2.0,
            lod=1.0,
        )

        expected = np.array([0.5, 1.5, 2.5])
        assert np.allclose(out.to_numpy(), expected, equal_nan=True)

    def test_bin_decoding_series_replaces_filter_values(self):
        out = apply(
            "bin_decoding",
            values=pl.Series("values", [-10.0, 1.25, -3.0, 4.5, -2.0]),
            copy_from_1=pl.Series("copy_a", [10.0, 20.0, 30.0, 40.0, 50.0]),
            filter_value_1=-10.0,
            copy_from_2=pl.Series("copy_b", [60.0, 70.0, 80.0, 90.0, 100.0]),
            filter_value_2=-3.0,
        )

        assert out.to_list() == [10.0, 1.25, 80.0, 4.5, -2.0]

    def test_bin_decoding_expr_accepts_variable_filter_count(
        self,
    ):
        df = pl.DataFrame(
            {
                "values": [-10.0, 1.25, -3.0, 4.5, -2.0, -1.0],
                "copy_a": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "copy_b": [70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
                "copy_c": [130.0, 140.0, 150.0, 160.0, 170.0, 180.0],
                "copy_d": [190.0, 200.0, 210.0, 220.0, 230.0, 240.0],
            }
        )

        expr = apply(
            "bin_decoding",
            values=pl.col("values"),
            copy_from_1=pl.col("copy_a"),
            filter_value_1=-10.0,
            copy_from_2=pl.col("copy_b"),
            filter_value_2=-3.0,
            copy_from_3=pl.col("copy_c"),
            filter_value_3=-2.0,
            copy_from_4=pl.col("copy_d"),
            filter_value_4=-1.0,
        )
        out = df.lazy().select(expr.alias("imputed")).collect()["imputed"]

        assert out.to_list() == [10.0, 1.25, 90.0, 4.5, 170.0, 240.0]

    def test_bin_decoding_requires_kwargs(self):
        with pytest.raises(TypeError):
            apply(
                "bin_decoding",
                pl.Series([-10.0]),
                pl.Series([10.0]),
                filter_value_1=-10.0,
            )

    def test_bin_decoding_requires_complete_pairs(self):
        with pytest.raises(ValueError, match="missing copy_from_1"):
            apply(
                "bin_decoding",
                values=pl.Series([-10.0]),
                filter_value_1=-10.0,
            )

    def test_bin_decoding_requires_contiguous_indices(self):
        with pytest.raises(ValueError, match="contiguous"):
            apply(
                "bin_decoding",
                values=pl.Series([-10.0]),
                copy_from_2=pl.Series([10.0]),
                filter_value_2=-10.0,
            )

    def test_bin_decoding_rejects_duplicate_filter_values(self):
        with pytest.raises(ValueError, match="unique"):
            apply(
                "bin_decoding",
                values=pl.Series([-10.0]),
                copy_from_1=pl.Series([10.0]),
                filter_value_1=-10.0,
                copy_from_2=pl.Series([20.0]),
                filter_value_2=-10.0,
            )
