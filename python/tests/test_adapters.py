import pytest

import compehndly


@pytest.mark.polars
class TestPolarsAdapter:
    @pytest.fixture(scope="class")
    def polars_module(self):
        import polars

        return polars

    def test_lazy_frame(self, polars_module):
        # Input data as a LazyFrame
        pl = polars_module
        lf = pl.LazyFrame(
            {
                "measured": [1.0, 2.0, 3.0],
                "sg_measured": [1.01, 1.02, 1.03],
            }
        )

        # Call the registered function on the LazyFrame columns
        result_lf = lf.with_columns(
            compehndly.normalize_specific_gravity(
                pl.col("measured"),
                pl.col("sg_measured"),
                sg_ref=1.024,
            ).alias("normalized")
        )

        # Collect to eager DataFrame
        result = result_lf.collect()

        # Expected values computed manually
        expected = pl.DataFrame(
            {
                "measured": [1.0, 2.0, 3.0],
                "sg_measured": [1.01, 1.02, 1.03],
                "normalized": [
                    (1.0 * (1.024 - 1)) / 1.01,
                    (2.0 * (1.024 - 1)) / 1.02,
                    (3.0 * (1.024 - 1)) / 1.03,
                ],
            }
        )

        # Polars has a built-in frame comparison
        assert result.frame_equal(expected, null_equal=True)
