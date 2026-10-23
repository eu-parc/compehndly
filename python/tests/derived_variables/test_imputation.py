import numpy as np
import polars as pl
import pytest

from compehndly import apply


@pytest.mark.derived
class TestImputation:
    @staticmethod
    def _assert_same_values(left: pl.Series, right: pl.Series):
        assert left.null_count() == right.null_count()
        assert len(left) == len(right)
        for left_value, right_value in zip(
            left.to_list(), right.to_list(), strict=True
        ):
            if (
                isinstance(left_value, float)
                and isinstance(right_value, float)
                and np.isnan(left_value)
                and np.isnan(right_value)
            ):
                continue
            assert left_value == right_value

    @staticmethod
    def _assert_null_or_between(
        value: float | None,
        lower: float,
        upper: float,
    ):
        assert value is not None
        assert lower <= value <= upper

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

        assert out.to_list() == [False, False, None, False, True, True, True]

    def test_lab_sensitivity_dichotomization_nulls_match_expr(self):
        df = pl.DataFrame(
            {
                "measurement": [None, 0.5, 0.5, 0.5],
                "lod": [1.0, None, 1.0, None],
                "loq": [2.0, 2.0, None, None],
            }
        )

        series_out = apply(
            "lab_sensitivity_dichotomization",
            measurement=df["measurement"],
            lod=df["lod"],
            loq=df["loq"],
        )
        expr = apply(
            "lab_sensitivity_dichotomization",
            measurement=pl.col("measurement"),
            lod=pl.col("lod"),
            loq=pl.col("loq"),
        )
        expr_out = df.lazy().select(expr.alias("out")).collect()["out"]

        assert series_out.to_list() == [None, False, False, None]
        self._assert_same_values(series_out, expr_out)

    def test_lab_sensitivity_dichotomization_decision_table(self):
        df = pl.DataFrame(
            {
                "measurement": [
                    -10.0,
                    -3.0,
                    -2.0,
                    -1.0,
                    0.5,
                    1.5,
                    3.0,
                    -10.0,
                    -3.0,
                    -2.0,
                    -1.0,
                    0.5,
                    1.5,
                    3.0,
                    -10.0,
                    -3.0,
                    -2.0,
                    -1.0,
                    0.5,
                    1.5,
                    3.0,
                    -10.0,
                    -3.0,
                    -2.0,
                    -1.0,
                    0.5,
                    1.5,
                    3.0,
                ],
                "lod": [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                "loq": [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                ],
            }
        )
        expected = [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            False,
            False,
            True,
            True,
            None,
            False,
            None,
            None,
            False,
            False,
            True,
            None,
            None,
            False,
            False,
            False,
            False,
            True,
        ]

        series_out = apply(
            "lab_sensitivity_dichotomization",
            measurement=df["measurement"],
            lod=df["lod"],
            loq=df["loq"],
        )
        expr = apply(
            "lab_sensitivity_dichotomization",
            measurement=pl.col("measurement"),
            lod=pl.col("lod"),
            loq=pl.col("loq"),
        )
        expr_out = df.lazy().select(expr.alias("out")).collect()["out"]

        assert series_out.to_list() == expected
        self._assert_same_values(series_out, expr_out)

    def test_random_single_imputation_basic(self):
        biomarker = pl.Series([5.0, -1.0, -2.0, 10.0, 0.5, 8.0])
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
        assert not np.any(out_np < 0), (
            "Censored values were not properly imputed."
        )
        assert out_np[0] == 5.0
        assert out_np[3] == 10.0
        assert out_np[5] == 8.0

    def test_random_single_imputation_bounds_respected(self):
        lod = 2.0
        loq = 4.0

        rng = np.random.default_rng(7)
        above_loq = rng.lognormal(size=100) + loq
        biomarker = above_loq.copy()
        biomarker[0:3] = np.array([-1.0, -2.0, 0.5])

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
        assert 0 <= imputed[2] <= 2.0
        assert np.all(out_np[3:] >= loq)

    def test_random_single_imputation_accepts_lod_loq_series(self):
        rng = np.random.default_rng(7)
        biomarker = rng.lognormal(size=100) + 5.0
        biomarker[0:3] = np.array([-1.0, -2.0, 0.5])

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
        assert 0 <= imputed[2] <= 2.0
        assert np.all(out_np[3:] >= 5.0)

    def test_random_single_imputation_expr_accepts_lod_loq_series(self):
        df = pl.DataFrame(
            {
                "biomarker": [-1.0, -2.0, 0.5, 5.0, 6.0, 7.0, 8.0],
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
        assert 0 <= out_np[2] <= 2.0
        assert out_np[3:].tolist() == [5.0, 6.0, 7.0, 8.0]

    def test_random_single_imputation_nulls_and_boundaries_match_expr(self):
        df = pl.DataFrame(
            {
                "biomarker": [None, -1.0, -2.0, -3.0, 5.0, 6.0, 7.0],
                "lod": [2.0, None, 2.0, 2.0, 2.0, 2.0, 2.0],
                "loq": [4.0, 4.0, None, None, 4.0, 4.0, 4.0],
            }
        )

        series_out = apply(
            "random_single_imputation",
            biomarker=df["biomarker"],
            lod=df["lod"],
            loq=df["loq"],
            seed=42,
        )
        expr = apply(
            "random_single_imputation",
            biomarker=pl.col("biomarker"),
            lod=pl.col("lod"),
            loq=pl.col("loq"),
            seed=42,
        )
        expr_out = df.lazy().select(expr.alias("out")).collect()["out"]

        assert series_out[:4].to_list() == [None, None, None, None]
        assert series_out[4:].to_list() == [5.0, 6.0, 7.0]
        self._assert_same_values(series_out, expr_out)

    def test_random_single_imputation_decision_table(self):
        measurements = [-10.0, -3.0, -2.0, -1.0, 0.5, 1.5, 3.0]
        df = pl.DataFrame(
            {
                "biomarker": measurements * 4,
                "lod": ([None] * 7 + [1.0] * 7 + [None] * 7 + [1.0] * 7),
                "loq": ([None] * 7 + [None] * 7 + [2.0] * 7 + [2.0] * 7),
            }
        )

        series_out = apply(
            "random_single_imputation",
            biomarker=df["biomarker"],
            lod=df["lod"],
            loq=df["loq"],
            seed=42,
        )
        expr = apply(
            "random_single_imputation",
            biomarker=pl.col("biomarker"),
            lod=pl.col("lod"),
            loq=pl.col("loq"),
            seed=42,
        )
        expr_out = df.lazy().select(expr.alias("out")).collect()["out"]
        values = series_out.to_list()

        assert values[:7] == [None] * 7
        assert values[7:10] == [None, None, None]
        self._assert_null_or_between(values[10], 0.0, 1.0)
        self._assert_null_or_between(values[11], 0.0, 1.0)
        assert values[12:14] == [1.5, 3.0]
        assert values[14] is None
        self._assert_null_or_between(values[15], 0.0, 2.0)
        assert values[16:18] == [None, None]
        self._assert_null_or_between(values[18], 0.0, 2.0)
        self._assert_null_or_between(values[19], 0.0, 2.0)
        assert values[20] == 3.0
        assert values[21:23] == [None, None]
        self._assert_null_or_between(values[23], 1.0, 2.0)
        self._assert_null_or_between(values[24], 0.0, 1.0)
        self._assert_null_or_between(values[25], 0.0, 1.0)
        self._assert_null_or_between(values[26], 1.0, 2.0)
        assert values[27] == 3.0
        self._assert_same_values(series_out, expr_out)

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
        assert np.all(np.isnan(out_np)), (
            "Failing check: all values should be NaN."
        )

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
        assert np.all(np.isnan(out_np)), (
            "Failing check: all values should be NaN."
        )

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

    def test_medium_bound_imputation_scalar_input_null_matches_expr(self):
        df = pl.DataFrame({"measurement": [0.2, None, 2.5]})

        series_out = apply(
            "medium_bound_imputation_scalar_input",
            df["measurement"],
            loq=2.0,
            lod=1.0,
        )
        expr = apply(
            "medium_bound_imputation_scalar_input",
            pl.col("measurement"),
            loq=2.0,
            lod=1.0,
        )
        expr_out = df.lazy().select(expr.alias("out")).collect()["out"]

        assert series_out.to_list() == [0.5, None, 2.5]
        self._assert_same_values(series_out, expr_out)

    def test_medium_bound_imputation_null_thresholds_match_expr(self):
        df = pl.DataFrame(
            {
                "measurement": [0.2, None, 0.2, 0.2],
                "lod": [1.0, 1.0, None, None],
                "loq": [2.0, 2.0, 2.0, None],
            }
        )

        series_out = apply(
            "medium_bound_imputation",
            measurement=df["measurement"],
            lod=df["lod"],
            loq=df["loq"],
        )
        expr = apply(
            "medium_bound_imputation",
            measurement=pl.col("measurement"),
            lod=pl.col("lod"),
            loq=pl.col("loq"),
        )
        expr_out = df.lazy().select(expr.alias("out")).collect()["out"]

        assert series_out.to_list() == [0.5, None, 1.0, None]
        self._assert_same_values(series_out, expr_out)

    def test_medium_bound_imputation_decision_table(self):
        measurements = [-10.0, -3.0, -2.0, -1.0, 0.5, 1.5, 3.0]
        df = pl.DataFrame(
            {
                "measurement": measurements * 4,
                "lod": ([None] * 7 + [1.0] * 7 + [None] * 7 + [1.0] * 7),
                "loq": ([None] * 7 + [None] * 7 + [2.0] * 7 + [2.0] * 7),
            }
        )
        expected = (
            [None] * 7
            + [None, None, None, 0.5, 0.5, 1.5, 3.0]
            + [None, 1.0, None, None, 1.0, 1.0, 3.0]
            + [None, None, 1.5, 0.5, 0.5, 1.5, 3.0]
        )

        series_out = apply(
            "medium_bound_imputation",
            measurement=df["measurement"],
            lod=df["lod"],
            loq=df["loq"],
        )
        expr = apply(
            "medium_bound_imputation",
            measurement=pl.col("measurement"),
            lod=pl.col("lod"),
            loq=pl.col("loq"),
        )
        expr_out = df.lazy().select(expr.alias("out")).collect()["out"]

        assert series_out.to_list() == expected
        self._assert_same_values(series_out, expr_out)

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

    def test_bin_decoding_nulls_match_expr(self):
        df = pl.DataFrame(
            {
                "values": [None, -1.0, -1.0, float("nan")],
                "copy": [10.0, None, 20.0, 30.0],
            }
        )

        series_out = apply(
            "bin_decoding",
            values=df["values"],
            filter_value_1=-1.0,
            copy_from_1=df["copy"],
        )
        expr = apply(
            "bin_decoding",
            values=pl.col("values"),
            filter_value_1=-1.0,
            copy_from_1=pl.col("copy"),
        )
        expr_out = df.lazy().select(expr.alias("out")).collect()["out"]

        assert series_out[:3].to_list() == [None, None, 20.0]
        assert np.isnan(series_out[3])
        self._assert_same_values(series_out, expr_out)

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
