import importlib
import polars as pl
import pytest

from compehndly import apply, get_map_fn, list_functions, with_derived_column


@pytest.mark.integration
class TestPolarsIntegration:
    @staticmethod
    def _extract_callable(path: str):
        assert "." in path
        module_name, func_name = path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, func_name)

    def test_list_functions_exposes_registered_specs(self):
        names = set(list_functions())
        assert "summation" in names
        assert "standardize" in names
        assert "random_single_imputation" in names

    def test_get_map_fn_works_with_external_map_batches_pattern(self):
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        map_fn = get_map_fn("summation", all_required=False)

        mapped = pl.struct(pl.col("a"), pl.col("b")).map_batches(
            lambda s: map_fn(a=s.struct.field("a"), b=s.struct.field("b")),
            return_dtype=pl.Float64,
        )

        out = df.lazy().select(mapped.alias("sum_col")).collect()
        assert out.columns == ["sum_col"]
        assert out.height == 2

    def test_apply_dispatch_supports_series_and_expr_inputs(self):
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

        out_series = apply("summation", df["a"], df["b"], all_required=False)
        out_expr = apply(
            "summation", pl.col("a"), pl.col("b"), all_required=False
        )
        out_lazy = (
            df.lazy().select(out_expr.alias("sum_col")).collect()["sum_col"]
        )

        assert out_series.to_list() == out_lazy.to_list()

    def test_with_derived_column_works_for_dataframe_and_lazyframe(self):
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

        eager = with_derived_column(
            frame=df,
            function_name="summation",
            input_columns=["a", "b"],
            output_column="sum_col",
            all_required=False,
        )
        lazy = with_derived_column(
            frame=df.lazy(),
            function_name="summation",
            input_columns=["a", "b"],
            output_column="sum_col",
            all_required=False,
        ).collect()

        assert "sum_col" in eager.columns
        assert eager["sum_col"].to_list() == lazy["sum_col"].to_list()

    def test_with_derived_column_supports_named_input_mapping(self):
        df = pl.DataFrame({"m": [50.0, 100.0], "sg": [1.02, 1.01]})
        out = with_derived_column(
            frame=df,
            function_name="normalize_specific_gravity",
            input_columns={"measured": "m", "sg_measured": "sg"},
            output_column="normalized",
            sg_ref=1.024,
        )
        expected = (df["m"] * (1.024 - 1) / df["sg"]).to_list()
        assert out["normalized"].to_list() == expected

    def test_apply_rejects_mixed_positional_and_named_data_inputs(self):
        with pytest.raises(ValueError):
            apply(
                "normalize_specific_gravity",
                pl.Series([50.0]),
                sg_measured=pl.Series([1.02]),
                sg_ref=1.024,
            )

    def test_unknown_function_name_raises_keyerror(self):
        with pytest.raises(KeyError):
            get_map_fn("does_not_exist")

    def test_entrypoint_path_extraction_for_summation(self):
        map_fn = self._extract_callable(
            "compehndly.entrypoints.summation_allow_partial"
        )
        df = pl.DataFrame({"a": [1.0, None], "b": [3.0, 4.0]})

        mapped = pl.struct(pl.col("a"), pl.col("b")).map_batches(
            lambda s: map_fn(a=s.struct.field("a"), b=s.struct.field("b")),
            return_dtype=pl.Float64,
        )
        out = df.lazy().select(mapped.alias("sum_col")).collect()
        assert out["sum_col"].to_list() == [4.0, 4.0]

    def test_entrypoint_named_args_for_non_commutative_function(self):
        map_fn = self._extract_callable(
            "compehndly.entrypoints.normalize_specific_gravity"
        )
        df = pl.DataFrame({"m": [50.0, 100.0], "sg": [1.02, 1.01]})

        mapped = pl.struct(
            pl.col("m").alias("measured"),
            pl.col("sg").alias("sg_measured"),
            pl.lit(1.024).alias("sg_ref"),
        ).map_batches(
            lambda s: map_fn(
                sg_measured=s.struct.field("sg_measured"),
                measured=s.struct.field("measured"),
                sg_ref=s.struct.field("sg_ref"),
            ),
            return_dtype=pl.Float64,
        )
        out = df.lazy().select(mapped.alias("normalized")).collect()
        expected = (df["m"] * (1.024 - 1) / df["sg"]).to_list()
        assert out["normalized"].to_list() == expected
