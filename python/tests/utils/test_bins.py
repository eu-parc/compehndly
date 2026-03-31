import polars as pl
import pytest

from compehndly.utils.bins import bin_categorical, bin_numeric


class TestBinCategorical:
    def test_series_input(self):
        data = pl.Series("cat", ["a", "b", "c", None])
        groups = {"x": ["a", "b"], "y": ["c"]}

        out = bin_categorical(data, groups, default="other")
        assert out.to_list() == ["x", "x", "y", "other"]

    def test_expr_input(self):
        df = pl.DataFrame({"cat": ["a", "b", "c", None]})
        groups = {"x": ["a", "b"], "y": ["c"]}

        out = df.select(
            bin_categorical(pl.col("cat"), groups, default="other").alias(
                "binned"
            )
        )
        assert out["binned"].to_list() == ["x", "x", "y", "other"]


class TestBinNumeric:
    def test_series_left_closed(self):
        data = pl.Series("val", [0.5, 1.5, 2.5, 3.5, None])
        out = bin_numeric(
            data, boundaries=[0.0, 2.0, 4.0], labels=["low", "high"]
        )
        assert out.to_list() == ["low", "low", "high", "high", None]

    def test_series_right_closed(self):
        data = pl.Series("val", [0.0, 2.0, 4.0])
        out = bin_numeric(
            data,
            boundaries=[0.0, 2.0, 4.0],
            labels=["low", "high"],
            right_inclusive=True,
        )
        assert out.to_list() == [None, "low", "high"]

    def test_expr_input(self):
        df = pl.DataFrame({"val": [0.5, 1.5, 2.5, 3.5]})
        out = df.select(
            bin_numeric(
                pl.col("val"),
                boundaries=[0.0, 2.0, 4.0],
                labels=["low", "high"],
            ).alias("binned")
        )
        assert out["binned"].to_list() == ["low", "low", "high", "high"]

    def test_invalid_labels_raises(self):
        with pytest.raises(ValueError):
            bin_numeric(
                pl.Series([1.0]), boundaries=[0.0, 1.0, 2.0], labels=["a"]
            )
