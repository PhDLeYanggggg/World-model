import numpy as np
import torch

from scripts import run_m3w_opendd_conditional_pair_gain as run


def test_additive_reference_uses_only_explicit_training_sites():
    a = np.zeros((2, 2, 45), np.float32)
    a[1, :, 0] = 1.
    x = torch.from_numpy(a)
    truth = torch.tensor([[0., 1.], [1., 1.]])
    data = dict(train=(x, None, truth), excluded=(x, None, 1.-truth), empty=None)
    coeff, sites = run.ridge_reference(data, ["train", "empty"])
    assert sites == ["train"]
    np.testing.assert_array_equal(run.linear_features(a), run.linear_features(a[:, ::-1]))
    scores = run.linear_features(a)@coeff
    assert scores[1] > scores[0]


def test_gain_targets_equal_full_grid_missing_pair_lower_bound():
    ref = np.zeros((3, 12, 2))
    y = ref.copy()
    y[0, :, 0] = 1.
    z = np.stack((y, -y))
    valid = np.ones(y.shape[:2], bool)
    valid[0, -3:] = False
    unit = 2.
    label = run.relative.labels(z, ref, y, valid, unit)
    score = run.target.frontier.pair_reduction(z, ref, y, valid)[:, 1]
    np.testing.assert_allclose((label[:, 0]-label[:, 1])*unit, score)
    assert label[0, 1] > 0.
