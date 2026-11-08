import numpy as np
import pytest

from src.world_model.m3w_shared_correction import shared_correction
from src.evaluation.m3w_risk_mixture import gap


def test_shared_motion_and_norm_control():
    v = np.array([[2., 1.], [2., 1.], [2., 1.], [1., 0.], [0., 1.], [1., -2.]])
    selected = np.array([True, False, True, True, True, True])
    a, b, c = shared_correction(v, selected, [0, 3, 5, 6])
    np.testing.assert_array_equal(a[:3], 0.)
    np.testing.assert_array_equal(b[:3], 0.)
    np.testing.assert_allclose(a[-1], .5*v[-1])
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1))
    np.testing.assert_array_equal(c, [2, 2, 2, 1, 1, -1])
    assert a[3, 1] < 0. and b[3, 1] == 0.


def test_empty_query_rejected():
    with pytest.raises(ValueError, match="query-grouped"):
        shared_correction([[0., 1.]], [True], [0, 0, 1])


def test_risk_gap_separates_mixture_and_rate():
    source = {"low": dict(harm=1, reference=10), "high": dict(harm=3, reference=10)}
    mixture_only = {"low": dict(harm=2, reference=20), "high": dict(harm=3, reference=10)}
    changed = gap(source, mixture_only)
    assert changed["within"] == pytest.approx(0.)
    assert changed["composition"] == pytest.approx(changed["gap"])
    swapped = gap(mixture_only, source)
    assert swapped["composition"] == pytest.approx(-changed["composition"])
    assert gap(source, {"low": dict(harm=0, reference=0), "high": mixture_only["high"]})["reference_coverage"] == [.5, 1.]
