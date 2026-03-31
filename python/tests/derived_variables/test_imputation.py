import numpy as np
import polars as pl
import pytest

from compehndly import apply


@pytest.mark.derived
class TestImputation:
    def test_random_single_imputation_basic(self):
        biomarker = pl.Series([5.0, -1.0, -2.0, 10.0, -3.0, 8.0])
        lod = 2.0
        loq = 4.0

        out = apply(
            "random_single_imputation", biomarker, lod=lod, loq=loq, seed=123
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
            "random_single_imputation",
            pl.Series(biomarker),
            lod=lod,
            loq=loq,
            seed=42,
        )

        out_np = out.to_numpy()
        imputed = out_np[:3]

        assert not np.any(np.isnan(imputed)), "Imputation produced NaNs."
        assert 0 <= imputed[0] <= 2.0
        assert 2.0 <= imputed[1] <= 4.0
        assert 0 <= imputed[2] <= 4.0
        assert np.all(out_np[3:] >= loq)

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
