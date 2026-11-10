import numpy as np
import torch

from scripts import run_m3w_opendd_nested_swap_release as run


def test_nonpositive_magnitude_abstains_without_changing_sign_candidates():
    h = np.arange(-7., 1.)[None, :, None]*np.array([[[1., 0.]], [[0., 1.]], [[-1., 0.]]])
    cv = np.zeros((3, 12, 2))
    anchor = np.zeros((1, 2, 12, 2))
    anchor[..., 0] = .5
    feature = dict(history=h, offsets=np.zeros((3, 2)), x=np.ones((3, 5)), counts=[3])
    packet = dict(cv=cv, bounds=[[0, 3]], additive_rows=np.array([[0, 1]]), additive_paths=anchor)

    class Negative(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((len(x), 2, 4))

    x = run.propose(packet, feature, Negative())
    assert x['sign'][0, 0] >= 0
    assert x['magnitude'][0, 0] == -1
    _, forecasts = run.candidates(packet, feature, x)
    np.testing.assert_array_equal(forecasts['magnitude'], forecasts['additive'])
    assert np.count_nonzero(np.any(forecasts['sign'] != cv, axis=(1, 2))) == 2


def test_calibration_excludes_tied_failure_and_empty_is_not_release():
    p = run.base.calibration
    assert p(np.array([.8, .7, .7, .1]), np.array([True, True, False, True])) == (.8, 1)
    assert p(np.array([.8, .8]), np.array([True, False])) == (None, 0)
    assert p(np.array([]), np.array([], dtype=bool)) == (None, 0)
