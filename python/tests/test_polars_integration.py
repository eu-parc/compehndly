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

    def _run_entrypoint_map_batches(
        self,
        *,
        path: str,
        frame: pl.DataFrame,
        struct_exprs,
        call,
        output_column: str,
        return_dtype=pl.Float64,
    ) -> pl.DataFrame:
        """
        Exercise the integration pattern used by config-based callers.

        The caller resolves a stable entrypoint path, packs input columns into
        a Polars struct, and calls the entrypoint from `map_batches` with
        explicit named fields and scalar kwargs.
        """
        map_fn = self._extract_callable(path)
        mapped = pl.struct(*struct_exprs).map_batches(
            lambda s: call(map_fn, s),
            return_dtype=return_dtype,
        )
        return frame.lazy().select(mapped.alias(output_column)).collect()

    def test_list_functions_exposes_registered_specs(self):
        names = set(list_functions())
        assert "summation" in names
        assert "multiply_by_group" in names
        assert "weighted_summation" in names
        assert "standardize" in names
        assert "random_single_imputation_scalar_input" in names
        assert "random_single_imputation" in names
        assert "bin_decoding" in names

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

    def test_with_derived_column_supports_priority_coalesce_lazy(self):
        df = pl.DataFrame(
            {
                "lab_a": [None, 2.0, None, None],
                "lab_b": [1.0, 20.0, None, None],
                "lab_c": [10.0, 200.0, 30.0, None],
            }
        )

        out = with_derived_column(
            frame=df.lazy(),
            function_name="coalesce_by_priority",
            input_columns={
                "primary": "lab_a",
                "secondary": "lab_b",
                "fallback": "lab_c",
            },
            output_column="coalesced",
            priority=("primary", "secondary", "fallback"),
        ).collect()

        assert out["coalesced"].to_list() == [1.0, 2.0, 30.0, None]

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

    def test_entrypoint_path_extraction_for_summation_cutoff(self):
        map_fn = self._extract_callable("compehndly.entrypoints.summation")
        df = pl.DataFrame(
            {
                "a": [1.0, None, None, None, None],
                "b": [None, 2.0, 3.0, None, None],
            }
        )

        mapped = pl.struct(pl.col("a"), pl.col("b")).map_batches(
            lambda s: map_fn(
                cutoff=0.6,
                a=s.struct.field("a"),
                b=s.struct.field("b"),
            ),
            return_dtype=pl.Float64,
        )
        out = df.lazy().select(mapped.alias("sum_col")).collect()
        assert out["sum_col"].null_count() == out.height

    def test_entrypoint_path_extraction_for_weighted_summation(self):
        df = pl.DataFrame(
            {
                "a": [1.0, None, 3.0],
                "b": [10.0, 20.0, None],
            }
        )

        out = self._run_entrypoint_map_batches(
            path="compehndly.entrypoints.weighted_summation",
            frame=df,
            struct_exprs=[
                pl.col("a"),
                pl.col("b"),
            ],
            call=lambda map_fn, s: map_fn(
                weight__b=0.5,
                a=s.struct.field("a"),
                weight__a=2.0,
                b=s.struct.field("b"),
            ),
            output_column="weighted",
        )

        assert out["weighted"].to_list() == [7.0, 10.0, 6.0]

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

    def test_entrypoint_named_args_for_standardize_creatinine(self):
        map_fn = self._extract_callable(
            "compehndly.entrypoints.standardize_creatinine"
        )
        df = pl.DataFrame({"m": [50.0, 100.0], "crt": [100.0, 80.0]})

        mapped = pl.struct(
            pl.col("m").alias("measured"),
            pl.col("crt").alias("crt"),
        ).map_batches(
            lambda s: map_fn(
                crt=s.struct.field("crt"),
                measured=s.struct.field("measured"),
            ),
            return_dtype=pl.Float64,
        )
        out = df.lazy().select(mapped.alias("standardized")).collect()
        expected = (df["m"] * 100 / df["crt"]).to_list()
        assert out["standardized"].to_list() == expected

    def test_entrypoint_named_args_for_total_lipid_concentration(self):
        map_fn = self._extract_callable(
            "compehndly.entrypoints.total_lipid_concentration"
        )
        df = pl.DataFrame({"chol": [200.0, 180.0], "trigl": [150.0, 120.0]})

        mapped = pl.struct(pl.col("chol"), pl.col("trigl")).map_batches(
            lambda s: map_fn(
                trigl=s.struct.field("trigl"),
                chol=s.struct.field("chol"),
            ),
            return_dtype=pl.Float64,
        )
        out = df.lazy().select(mapped.alias("lipid")).collect()
        expected = (df["chol"] * 2.27 + df["trigl"] + 62.3).to_list()
        assert out["lipid"].to_list() == expected

    def test_entrypoint_named_args_for_priority_coalesce(self):
        map_fn = self._extract_callable(
            "compehndly.entrypoints.coalesce_by_priority"
        )
        df = pl.DataFrame(
            {
                "lab_a": [None, 2.0, None, None],
                "lab_b": [1.0, 20.0, None, None],
                "lab_c": [10.0, 200.0, 30.0, None],
            }
        )

        mapped = pl.struct(
            pl.col("lab_a").alias("primary"),
            pl.col("lab_b").alias("secondary"),
            pl.col("lab_c").alias("fallback"),
        ).map_batches(
            lambda s: map_fn(
                priority=("primary", "secondary", "fallback"),
                primary=s.struct.field("primary"),
                secondary=s.struct.field("secondary"),
                fallback=s.struct.field("fallback"),
            ),
            return_dtype=pl.Float64,
        )
        out = df.lazy().select(mapped.alias("coalesced")).collect()
        assert out["coalesced"].to_list() == [1.0, 2.0, 30.0, None]

    def test_entrypoint_scalar_kwargs_for_medium_bound_imputation(self):
        map_fn = self._extract_callable(
            "compehndly.entrypoints.medium_bound_imputation_scalar_input"
        )
        df = pl.DataFrame({"measurement": [0.2, 1.2, 2.5]})

        mapped = pl.struct(pl.col("measurement")).map_batches(
            lambda s: map_fn(
                measurement=s.struct.field("measurement"),
                loq=2.0,
                lod=1.0,
            ),
            return_dtype=pl.Float64,
        )
        out = df.lazy().select(mapped.alias("imputed")).collect()
        assert out["imputed"].to_list() == [0.5, 1.5, 2.5]

    def test_entrypoint_kwargs_for_bin_decoding(self):
        map_fn = self._extract_callable("compehndly.entrypoints.bin_decoding")
        df = pl.DataFrame(
            {
                "values": [-10.0, 1.25, -3.0, 4.5],
                "copy_a": [10.0, 20.0, 30.0, 40.0],
                "copy_b": [50.0, 60.0, 70.0, 80.0],
            }
        )

        mapped = pl.struct(
            pl.col("values"),
            pl.col("copy_a"),
            pl.col("copy_b"),
        ).map_batches(
            lambda s: map_fn(
                values=s.struct.field("values"),
                copy_from_1=s.struct.field("copy_a"),
                filter_value_1=-10.0,
                copy_from_2=s.struct.field("copy_b"),
                filter_value_2=-3.0,
            ),
            return_dtype=pl.Float64,
        )
        out = df.lazy().select(mapped.alias("imputed")).collect()
        assert out["imputed"].to_list() == [10.0, 1.25, 70.0, 4.5]
