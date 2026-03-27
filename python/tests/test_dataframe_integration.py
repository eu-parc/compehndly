import pyarrow as pa
import pytest

import compehndly


@pytest.mark.core
class TestIntegration:
    @pytest.fixture(scope="module")
    def registry(self):
        to_register = ["tests.utils"]
        return compehndly.FunctionRegistry.build_registry(to_register)

    def test_normalize_basic(self, registry):
        biomarker = pa.array([1.0, 2.0, 3.0], type=pa.float64())
        sg = pa.array([1.01, 1.02, 1.03], type=pa.float64())
        normalize = registry.get("normalize")
        result = normalize(biomarker, sg, sg_ref=1.024)

        expected = pa.array(
            [
                (1.0 * (1.024 - 1)) / (1.01 - 1),
                (2.0 * (1.024 - 1)) / (1.02 - 1),
                (3.0 * (1.024 - 1)) / (1.03 - 1),
            ],
            type=pa.float64(),
        )
        assert result.equals(expected)

    def test_normalize_with_nulls(self, registry):
        biomarker = pa.array([1.0, None, 3.0], type=pa.float64())
        sg = pa.array([1.01, 1.02, None], type=pa.float64())
        normalize = registry.get("normalize")
        result = normalize(biomarker, sg, sg_ref=1.024)

        expected = pa.array([(1.0 * (1.024 - 1)) / (1.01 - 1), None, None], type=pa.float64())

        assert result.equals(expected)

    def test_normalize_zero_denominator(self, registry):
        biomarker = pa.array([1.0])
        sg = pa.array([1.0])  # denominator = 0
        normalize = registry.get("normalize")
        result = normalize(biomarker, sg, sg_ref=1.024)
        expected = pa.array([None], type=pa.float64())
        assert result.equals(expected)


@pytest.mark.polars
class TestPolarsIntegration:
    @pytest.fixture(scope="module")
    def registry(self):
        import polars as ps

        to_register = ["tests.utils"]
        return compehndly.FunctionRegistry.build_registry(to_register, adapter="polars")

    def test_normalize_basic(self, registry):
        import polars as ps

        biomarker = ps.Series([1.0, 2.0, 3.0], dtype=ps.Float64)
        sg = ps.Series([1.01, 1.02, 1.03], dtype=ps.Float64)
        normalize = registry.get("normalize")
        result = normalize(biomarker, sg, sg_ref=1.024)

        expected = ps.Series(
            [
                (1.0 * (1.024 - 1)) / (1.01 - 1),
                (2.0 * (1.024 - 1)) / (1.02 - 1),
                (3.0 * (1.024 - 1)) / (1.03 - 1),
            ],
            dtype=ps.Float64,
        )
        assert result.equals(expected)


@pytest.mark.pandas
class TestPandasIntegration:
    @pytest.fixture(scope="module")
    def registry(self):
        import pandas as ps

        to_register = ["tests.utils"]
        return compehndly.FunctionRegistry.build_registry(to_register, adapter="pandas")

    def test_normalize_basic(self, registry):
        import pandas as ps

        biomarker = ps.Series([1.0, 2.0, 3.0], dtype=float)
        sg = ps.Series([1.01, 1.02, 1.03], dtype=float)
        normalize = registry.get("normalize")
        result = normalize(biomarker, sg, sg_ref=1.024)

        expected = ps.Series(
            [
                (1.0 * (1.024 - 1)) / (1.01 - 1),
                (2.0 * (1.024 - 1)) / (1.02 - 1),
                (3.0 * (1.024 - 1)) / (1.03 - 1),
            ],
            dtype=float,
        )
        assert result.equals(expected)
