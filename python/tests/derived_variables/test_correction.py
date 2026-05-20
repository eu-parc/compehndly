import numpy as np
import polars as pl
import pytest

from compehndly import apply


@pytest.mark.derived
class TestCorrection:
    def test_standardize_basic(self):
        measured = pl.Series([50.0, 100.0, 75.0, 0.0])
        standard = pl.Series([25.0, 50.0, 25.0, 10.0])

        out = apply("standardize", measured, standard)
        expected = np.array([200.0, 200.0, 300.0, 0.0])
        assert np.allclose(out.to_numpy(), expected, equal_nan=True)

    def test_standardize_with_nulls(self):
        measured = pl.Series([50.0, None, 75.0])
        standard = pl.Series([25.0, 50.0, None])

        out = apply("standardize", measured, standard)
        assert out[0] == 200.0
        assert out[1] is None
        assert out[2] is None

    def test_standardize_creatinine_basic(self):
        measured = pl.Series([50.0, 100.0, 75.0])
        crt = pl.Series([100.0, 80.0, 120.0])

        out = apply("standardize_creatinine", measured, crt)
        expected = (measured * 100 / crt).to_numpy()
        assert np.allclose(out.to_numpy(), expected, equal_nan=True)

    def test_normalize_specific_gravity_basic(self):
        measured = pl.Series([50.0, 100.0, 75.0])
        sg_measured = pl.Series([1.020, 1.015, 1.025])
        sg_ref = 1.024

        out = apply(
            "normalize_specific_gravity", measured, sg_measured, sg_ref=sg_ref
        )
        expected = (measured * (sg_ref - 1) / sg_measured).to_numpy()
        assert np.allclose(out.to_numpy(), expected, equal_nan=True)

    def test_normalize_specific_gravity_named_series_kwargs(self):
        measured = pl.Series([50.0, 100.0, 75.0])
        sg_measured = pl.Series([1.020, 1.015, 1.025])
        sg_ref = 1.024

        out = apply(
            "normalize_specific_gravity",
            sg_measured=sg_measured,
            measured=measured,
            sg_ref=sg_ref,
        )
        expected = (measured * (sg_ref - 1) / sg_measured).to_numpy()
        assert np.allclose(out.to_numpy(), expected, equal_nan=True)

    def test_total_lipid_concentration_basic(self):
        chol = pl.Series([200.0, 180.0, 220.0, 190.0])
        trigl = pl.Series([150.0, 120.0, 180.0, 140.0])

        out = apply("total_lipid_concentration", chol, trigl)
        expected = (chol * 2.27 + trigl + 62.3).to_numpy()
        assert np.allclose(out.to_numpy(), expected, equal_nan=True)

    def test_consolidate_lipid_value_basic(self):
        lipid_enz_harm = pl.Series([50.0, None, None])
        lipid_enz_imp = pl.Series([1.0, 100.0, None])
        lipid_imp = pl.Series([1.0, 1.0, 75.0])

        out = apply(
            "consolidate_lipid_value", lipid_enz_harm, lipid_enz_imp, lipid_imp
        )
        out_list = out.to_numpy().tolist()
        expected = [50.0, 100.0, 75.0]
        assert expected == out_list

    def test_coalesce_by_priority_basic(self):
        out = apply(
            "coalesce_by_priority",
            primary=pl.Series([None, 2.0, None, None]),
            secondary=pl.Series([1.0, 20.0, None, None]),
            fallback=pl.Series([10.0, 200.0, 30.0, None]),
            priority=("primary", "secondary", "fallback"),
        )

        assert out.to_list() == [1.0, 2.0, 30.0, None]

    def test_coalesce_by_priority_respects_priority_order(self):
        out = apply(
            "coalesce_by_priority",
            primary=pl.Series([None, 2.0, None]),
            secondary=pl.Series([1.0, 20.0, None]),
            fallback=pl.Series([10.0, 200.0, 30.0]),
            priority=("fallback", "secondary", "primary"),
        )

        assert out.to_list() == [10.0, 200.0, 30.0]

    def test_coalesce_by_priority_rejects_unknown_priority_name(self):
        with pytest.raises(ValueError, match="unknown inputs"):
            apply(
                "coalesce_by_priority",
                primary=pl.Series([None]),
                priority=("primary", "missing"),
            )

    def test_coalesce_by_priority_rejects_inputs_not_in_priority(self):
        with pytest.raises(ValueError, match="not listed in priority"):
            apply(
                "coalesce_by_priority",
                primary=pl.Series([None]),
                fallback=pl.Series([1.0]),
                priority=("primary",),
            )

    def test_standardize_lipid_basic(self):
        measured = pl.Series([50.0, 100.0, 75.0])
        lipid_value = pl.Series([666.3, 590.9, 724.3])

        out = apply("standardize_lipid", measured, lipid_value)
        expected = (measured * 100 / lipid_value).to_numpy()
        assert np.allclose(out.to_numpy(), expected, equal_nan=True)
