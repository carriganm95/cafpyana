"""makedf.flux: flux integration / active-volume / xsec-unit helpers hoisted
out of analysis_village/nueNp0Pi/utils.py.
"""
import inspect

import pytest

from makedf.flux import get_active_volume, get_xsec_unit


@pytest.mark.parametrize(
    "detector,expected_cm3",
    [
        ("SBND", 380 * 380 * 440),
        ("SBND_nohighyz", 380 * 380 * 440 - 380 * (190 - 100) * (450 - 250)),
        ("SBND_face", 380 * 380 * 50),
        ("SBND_face_yzcut", 380 * 380 * 50 - 380 * (190 - 100) * 50),
        ("SBND_end", 380 * 380 * 50),
    ],
)
def test_get_active_volume_known_detectors(detector, expected_cm3):
    assert get_active_volume(detector) == expected_cm3


def test_get_active_volume_unknown_detector_raises_value_error():
    # Regression guard: this used to silently fall through and raise
    # UnboundLocalError instead of a clear, actionable error.
    with pytest.raises(ValueError):
        get_active_volume("not_a_real_detector")


def test_get_xsec_unit_fluxfile_has_no_hardcoded_default():
    # Regression guard: fluxfile used to default to a specific physicist's
    # personal /exp/sbnd/... path. It must now be required.
    sig = inspect.signature(get_xsec_unit)
    assert sig.parameters["fluxfile"].default is inspect.Parameter.empty


def test_get_xsec_unit_missing_fluxfile_raises_typeerror():
    with pytest.raises(TypeError):
        get_xsec_unit(1e20)  # fluxfile omitted entirely
