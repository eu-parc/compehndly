import numpy as np
import pyarrow as pa
import pytest

from compehndly.derived_variables.imputation import (
    _random_single_imputation_arrow_v0_0_1,
)


@pytest.mark.core
class TestImputation:
    def test_imputation_basic(self):
        # A small dataset with mixed censor categories
        biomarker = pa.array([5.0, -1, -2, 10.0, -3, 8.0])
        lod = 2.0
        loq = 4.0

        out = _random_single_imputation_arrow_v0_0_1(biomarker, lod, loq, seed=123)

        assert isinstance(out, pa.Array)
        assert len(out) == len(biomarker)
        out_np = out.to_numpy()
        assert not np.any(out_np < 0), "Censored values were not properly imputed."
        assert out_np[0] == 5.0
        assert out_np[3] == 10.0
        assert out_np[5] == 8.0

    @pytest.mark.core
    def test_imputation_bounds_respected(self):
        lod = 2.0
        loq = 4.0

        rng = np.random.default_rng()
        above_loq = rng.lognormal(size=100) + loq
        biomarker = above_loq.copy()
        biomarker[0:3] = np.array([-1.0, -2.0, -3.0])
        biomarker_pa = pa.array(biomarker)

        out = _random_single_imputation_arrow_v0_0_1(biomarker_pa, lod, loq, seed=42)
        out_np = out.to_numpy()
        imputed = out_np[:3]

        assert not np.any(np.isnan(imputed)), "Imputation produced NaNs."
        # Check correct ranges:
        # -1 → [0, LOD]
        assert 0 <= imputed[0] <= 2.0
        # -2 → [LOD, LOQ]
        assert 2.0 <= imputed[1] <= 4.0
        # -3 → [0, LOQ]
        assert 0 <= imputed[2] <= 4.0
        # Ensure uncensored is unchanged
        assert np.all(out_np[3:] >= loq)
