import numpy as np
import pytest

from scripts import run_m3w_av2_temporal_menu as run
from src.world_model import m3w_av2_temporal_menu as model


def test_temporal_bank_keeps_exact_peaks_and_zero_offsets():
    cv = np.arange(2*12*2, dtype=float).reshape(2, 12, 2)
    delta = np.zeros_like(cv)
    delta[..., 0] = 2.
    bank = np.stack((cv+delta, cv), axis=1)
    out = model.expanded(cv, bank)
    np.testing.assert_array_equal(out[:, :2], bank)
    np.testing.assert_array_equal(out[:, 1::2], np.broadcast_to(cv[:, None], out[:, 1::2].shape))
    peak = np.linalg.norm(out-cv[:, None], axis=-1).max(-1)
    np.testing.assert_allclose(peak[:, ::2], 2.)
    np.testing.assert_allclose(out[:, 2, 0, 0]-cv[:, 0, 0], 2.)
    np.testing.assert_allclose(out[:, 4, 0, 0]-cv[:, 0, 0], .5)


def test_membership_tracks_same_size_banks_and_cross_bank_options():
    out = model.membership(np.array([[0, 6], [2, 8], [4, 10], [0, 8]]), 2)
    for k, ix in zip(model.MODES[:-1], range(3), strict=True):
        np.testing.assert_array_equal(np.flatnonzero(out[k]), [ix])
    assert out["union"].all()


def test_pair_search_does_not_count_an_equal_reference_or_known_only_gain():
    a = np.zeros((3, 12, 2))
    y = a.copy()
    y[1, :, 0], y[2, :, 0] = 2., 4.
    valid = np.ones((3, 12), dtype=bool)
    assert run.exact_pair(np.stack((a, y)), a, y, valid, 72.) == 1
    p = y.copy()
    p[:, -1, 0] = [0., 200., -200.]
    valid[:, -1] = False
    assert run.exact_pair(np.stack((a, p)), a, y, valid, 66.) is None


def test_temporal_bank_rejects_nonfinite_paths():
    x = np.zeros((2, 12, 2))
    with pytest.raises(ValueError):
        model.expanded(x, x[:, None]+np.nan)
