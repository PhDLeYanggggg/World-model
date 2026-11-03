import numpy as np
import pytest

from src.evaluation.m3w_eqmotion_amplitude import attenuate
from src.evaluation.m3w_selection_forecast_readout import errors


@pytest.mark.parametrize('alpha', [.001, .01, .25, .5])
def test_available_label_ADE_convexity_and_harm_upper_bound(alpha):
    rng = np.random.default_rng(41)
    b, p, y = [rng.normal(size=(101, 12, 2)) for _ in range(3)]
    valid = rng.random((101, 12)) > .3; valid[:, 0] = True
    base, _ = errors(b, y, valid); full, _ = errors(p, y, valid)
    partial, _ = errors(attenuate(b, p, alpha), y, valid)
    assert np.all(partial <= (1-alpha)*base+alpha*full+1e-12)
    assert np.all(np.maximum(partial-base, 0) <= alpha*np.maximum(full-base, 0)+1e-12)
    assert np.all(partial-base <= alpha*(full-base)+1e-12)
    d = np.linalg.norm(p-b, axis=-1).max(1)
    np.testing.assert_allclose(np.linalg.norm(attenuate(b, p, alpha)-b, axis=-1).max(1), alpha*d)
