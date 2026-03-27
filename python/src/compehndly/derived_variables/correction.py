import pyarrow as pa
import pyarrow.compute as pc

__registrations__ = []


# TODO: move decorator for joint use
def register(registry_name, name, version):
    def decorator(func):
        __registrations__.append((registry_name, name, version, func))
        return func

    return decorator


# ---------------- STANDARDIZE ----------------------


def _standardize_v0_0_1_reference(
    measured: float,
    standard: float,
) -> float:
    return 100 * measured / standard


@register(registry_name="default", name="standardize", version="0.0.1")
def _standardize_v0_0_1_arrow(measured: pa.Array, standard: pa.Array) -> pa.Array:
    return pc.divide(pc.multiply(measured, 100), standard)


# ---------------- URINARY BIOMARKERS ----------------------


def _standardize_creatinine_v0_0_1_reference(
    measured: float,
    crt: float,
) -> float:
    "measured in micrograms/L, crt in mg/dL"
    return _standardize_v0_0_1_reference(measured, crt)


@register(registry_name="default", name="standardize_creatinine", version="0.0.1")
def _standardize_creatinine_v0_0_1_arrow(measured: pa.Array, crt: pa.Array) -> pa.Array:
    return _standardize_v0_0_1_arrow(measured, crt)


def _normalize_specific_gravity_v0_0_1_reference(
    measured: float,
    sg_measured: float,
    sg_ref: float,
) -> float:
    ret = measured * (sg_ref - 1) / sg_measured

    return ret


@register(registry_name="default", name="normalize_specific_gravity", version="0.0.1")
def _normalize_specific_gravity_v0_0_1_arrow(measured: pa.Array, sg_measured: pa.Array, sg_ref: float) -> pa.Array:
    # Compute (sg_ref - 1) as a scalar
    sg_factor = pa.scalar(sg_ref - 1, type=pa.float64())

    # measured * (sg_ref - 1)
    numerator = pc.multiply(measured, sg_factor)

    # (measured * (sg_ref - 1)) / sg_measured
    ret = pc.divide(numerator, sg_measured)

    return ret


# ---------------- LIPID SOLUBLE BIOMARKERS ----------------------


def _total_lipid_concentration_v0_0_1_reference(chol: float, trigl: float) -> float:
    """The total blood lipid content was calculated using the formula proposed by
    (Bernert et al., 2007; Phillips et al., 1989) as advised by the analytical experts of HBM4EU.
    Total lipids (mg/dL) = 2.27 * cholesterol (mg/dL) + triglycerides (mg/dL) + 62.3
    """
    return 2.27 * chol + trigl + 62.3


@register(registry_name="default", name="total_lipid_concentration", version="0.0.1")
def _total_lipid_concentration_v0_0_1_arrow(chol: pa.Array, trigl: pa.Array) -> pa.Array:
    return pc.add(pc.multiply(chol, 2.27), pc.add(trigl, 62.3))


def _standardize_lipid_v0_0_1_reference(
    measured: float,
    lipid_value: float,
) -> float:
    return _standardize_v0_0_1_reference(measured, lipid_value)

@register(registry_name="default", name="standardize_lipid", version="0.0.1")
def _standardize_lipid_v0_0_1_arrow(measured: pa.Array, lipid_value: pa.Array) -> pa.Array:
    return _standardize_v0_0_1_arrow(measured, lipid_value)