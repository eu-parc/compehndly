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
