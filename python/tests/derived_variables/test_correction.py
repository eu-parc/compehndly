import pytest
import pyarrow as pa
import pyarrow.compute as pc

from compehndly.derived_variables.correction import (
    _standardize_v0_0_1_reference,
    _standardize_v0_0_1_arrow,
    _standardize_creatinine_v0_0_1_reference,
    _standardize_creatinine_v0_0_1_arrow,
    _normalize_specific_gravity_v0_0_1_reference,
    _normalize_specific_gravity_v0_0_1_arrow,
    _total_lipid_concentration_v0_0_1_reference,
    _total_lipid_concentration_v0_0_1_arrow,
    _standardize_lipid_v0_0_1_reference,
    _standardize_lipid_v0_0_1_arrow,
)


@pytest.fixture
def sample_measurements():
    return {
        "measured": 50.0,
        "standard": 25.0,
        "crt": 100.0,
        "sg_measured": 1.020,
        "sg_ref": 1.024,
        "chol": 200.0,
        "trigl": 150.0,
    }


@pytest.fixture
def sample_arrays():
    return {
        "measured": pa.array([50.0, 100.0, 75.0, 0.0], type=pa.float64()),
        "standard": pa.array([25.0, 50.0, 25.0, 10.0], type=pa.float64()),
        "crt": pa.array([100.0, 80.0, 120.0, 90.0], type=pa.float64()),
        "sg_measured": pa.array([1.020, 1.015, 1.025, 1.010], type=pa.float64()),
        "chol": pa.array([200.0, 180.0, 220.0, 190.0], type=pa.float64()),
        "trigl": pa.array([150.0, 120.0, 180.0, 140.0], type=pa.float64()),
    }


class TestStandardize:
    def test_standardize_reference_basic(self, sample_measurements):
        result = _standardize_v0_0_1_reference(sample_measurements["measured"], sample_measurements["standard"])

        expected = 100 * sample_measurements["measured"] / sample_measurements["standard"]

        assert result == expected

    def test_standardize_reference_zero_standard(self):
        with pytest.raises(ZeroDivisionError):
            _standardize_v0_0_1_reference(50.0, 0.0)

    def test_standardize_reference_zero_measured(self, sample_measurements):
        result = _standardize_v0_0_1_reference(0.0, sample_measurements["standard"])
        assert result == 0.0

    def test_standardize_arrow_basic(self, sample_arrays):
        result = _standardize_v0_0_1_arrow(sample_arrays["measured"], sample_arrays["standard"])

        expected = map(
            _standardize_v0_0_1_reference, sample_arrays["measured"].to_pylist(), sample_arrays["standard"].to_pylist()
        )
        expected = pa.array(list(expected), type=pa.float64())

        assert result.equals(expected)
        assert result.type == pa.float64()

    def test_standardize_arrow_with_nulls(self):
        measured = pa.array([50.0, None, 75.0], type=pa.float64())
        standard = pa.array([25.0, 50.0, None], type=pa.float64())
        result = _standardize_v0_0_1_arrow(measured, standard)
        assert result[0].as_py() == 200.0
        assert result[1].is_valid is False
        assert result[2].is_valid is False

    def test_standardize_consistency(self, sample_measurements, sample_arrays):
        ref_result = _standardize_v0_0_1_reference(sample_measurements["measured"], sample_measurements["standard"])
        arrow_result = _standardize_v0_0_1_arrow(
            pa.array([sample_measurements["measured"]]),
            pa.array([sample_measurements["standard"]]),
        )

        assert arrow_result[0].as_py() == ref_result


class TestStandardizeCreatinine:
    def test_standardize_creatinine_reference_basic(self, sample_measurements):
        result = _standardize_creatinine_v0_0_1_reference(sample_measurements["measured"], sample_measurements["crt"])

        expected = _standardize_v0_0_1_reference(sample_measurements["measured"], sample_measurements["crt"])

        assert result == expected

    def test_standardize_creatinine_arrow_basic(self, sample_arrays):
        result = _standardize_creatinine_v0_0_1_arrow(sample_arrays["measured"], sample_arrays["crt"])
        expected = _standardize_v0_0_1_arrow(sample_arrays["measured"], sample_arrays["crt"])
        assert result.equals(expected)


class TestNormalizeSpecificGravity:
    def test_normalize_specific_gravity_reference_basic(self, sample_measurements):
        result = _normalize_specific_gravity_v0_0_1_reference(
            sample_measurements["measured"],
            sample_measurements["sg_measured"],
            sample_measurements["sg_ref"],
        )

        expected = (
            sample_measurements["measured"] * (sample_measurements["sg_ref"] - 1) / sample_measurements["sg_measured"]
        )
        assert result == expected

    def test_normalize_specific_gravity_arrow_basic(self, sample_arrays):
        sg_ref = 1.024
        result = _normalize_specific_gravity_v0_0_1_arrow(
            sample_arrays["measured"], sample_arrays["sg_measured"], sg_ref
        )

        expected = map(
            _normalize_specific_gravity_v0_0_1_reference,
            sample_arrays["measured"].to_pylist(),
            sample_arrays["sg_measured"].to_pylist(),
            [sg_ref] * len(sample_arrays["measured"]),
        )
        expected = pa.array(list(expected), type=pa.float64())
        assert result == expected

    def test_normalize_specific_gravity_arrow_with_nulls(self):
        measured = pa.array([50.0, None, 75.0], type=pa.float64())
        sg_measured = pa.array([1.020, 1.015, None], type=pa.float64())
        sg_ref = 1.024
        result = _normalize_specific_gravity_v0_0_1_arrow(measured, sg_measured, sg_ref)
        assert result[0].is_valid is True
        assert result[1].is_valid is False
        assert result[2].is_valid is False

    def test_normalize_specific_gravity_consistency(self, sample_measurements):
        ref_result = _normalize_specific_gravity_v0_0_1_reference(
            sample_measurements["measured"],
            sample_measurements["sg_measured"],
            sample_measurements["sg_ref"],
        )
        arrow_result = _normalize_specific_gravity_v0_0_1_arrow(
            pa.array([sample_measurements["measured"]]),
            pa.array([sample_measurements["sg_measured"]]),
            sample_measurements["sg_ref"],
        )
        assert pytest.approx(arrow_result[0].as_py(), rel=1e-6) == ref_result


class TestTotalLipidConcentration:
    def test_total_lipid_concentration_reference_basic(self, sample_measurements):
        result = _total_lipid_concentration_v0_0_1_reference(sample_measurements["chol"], sample_measurements["trigl"])

        expected = 2.27 * sample_measurements["chol"] + sample_measurements["trigl"] + 62.3
        assert result == expected

    def test_total_lipid_concentration_reference_zero_values(self):
        result = _total_lipid_concentration_v0_0_1_reference(0.0, 0.0)

        expected = 2.27 * 0.0 + 0.0 + 62.3
        assert result == expected

    def test_total_lipid_concentration_arrow_basic(self, sample_arrays):
        result = _total_lipid_concentration_v0_0_1_arrow(sample_arrays["chol"], sample_arrays["trigl"])
        result = pc.round(result, 4)  # Round to avoid floating-point precision issues

        expected = map(
            _total_lipid_concentration_v0_0_1_reference,
            sample_arrays["chol"].to_pylist(),
            sample_arrays["trigl"].to_pylist(),
        )
        expected = pa.array(list(expected), type=pa.float64())
        expected = pc.round(expected, 4)

        assert result.equals(expected)
        assert result.type == pa.float64()

    def test_total_lipid_concentration_arrow_with_nulls(self):
        chol = pa.array([200.0, None, 220.0], type=pa.float64())
        trigl = pa.array([150.0, 120.0, None], type=pa.float64())
        result = _total_lipid_concentration_v0_0_1_arrow(chol, trigl)
        assert result[0].is_valid is True
        assert result[1].is_valid is False
        assert result[2].is_valid is False

    def test_total_lipid_concentration_consistency(self, sample_measurements):
        ref_result = _total_lipid_concentration_v0_0_1_reference(
            sample_measurements["chol"], sample_measurements["trigl"]
        )
        arrow_result = _total_lipid_concentration_v0_0_1_arrow(
            pa.array([sample_measurements["chol"]]),
            pa.array([sample_measurements["trigl"]]),
        )
        assert pytest.approx(arrow_result[0].as_py(), rel=1e-6) == ref_result


class TestStandardizeLipid:
    def test_standardize_lipid_reference_basic(self, sample_measurements):
        lipid_value = 666.3
        result = _standardize_lipid_v0_0_1_reference(sample_measurements["measured"], lipid_value)
        expected = 100 * sample_measurements["measured"] / lipid_value

        assert result == expected

    def test_standardize_lipid_reference_zero_lipid(self):
        with pytest.raises(ZeroDivisionError):
            _standardize_lipid_v0_0_1_reference(50.0, 0.0)

    def test_standardize_lipid_arrow_basic(self):
        measured = pa.array([50.0, 100.0, 75.0], type=pa.float64())
        lipid_value = pa.array([666.3, 590.9, 724.3], type=pa.float64())
        result = _standardize_lipid_v0_0_1_arrow(measured, lipid_value)

        expected = map(_standardize_lipid_v0_0_1_reference, measured.to_pylist(), lipid_value.to_pylist())
        expected = pa.array(list(expected), type=pa.float64())

        assert result.equals(expected)
        assert result.type == pa.float64()

    def test_standardize_lipid_arrow_with_nulls(self):
        measured = pa.array([50.0, None, 75.0], type=pa.float64())
        lipid_value = pa.array([666.3, 590.9, None], type=pa.float64())
        result = _standardize_lipid_v0_0_1_arrow(measured, lipid_value)
        assert result[0].is_valid is True
        assert result[1].is_valid is False
        assert result[2].is_valid is False

    def test_standardize_lipid_consistency(self, sample_measurements):
        lipid_value = 666.3
        ref_result = _standardize_lipid_v0_0_1_reference(sample_measurements["measured"], lipid_value)
        arrow_result = _standardize_lipid_v0_0_1_arrow(
            pa.array([sample_measurements["measured"]]),
            pa.array([lipid_value]),
        )
        assert pytest.approx(arrow_result[0].as_py(), rel=1e-6) == ref_result
