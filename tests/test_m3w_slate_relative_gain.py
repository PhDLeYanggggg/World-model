import numpy as np
import torch

from scripts.run_m3w_opendd_slate_relative_gain import assemble, legal
from src.evaluation.m3w_interaction_frontier import pair_reduction
from src.world_model.m3w_slate_relative_gain import RelativeGain, features, labels


def inputs():
    rng = np.random.default_rng(52)
    return (rng.normal(size=(3, 8, 2)), rng.normal(size=(3, 12, 2)),
            rng.normal(size=(3, 12, 2)), rng.normal(size=(2, 3, 12, 2)))


def test_slate_labels_equal_grid_bound_and_include_unchanged_agent():
    h, cv, _, candidate = inputs()
    ref = cv.copy()
    candidate[:, 2] = ref[2]
    valid = np.ones((3, 12), dtype=bool)
    valid[2, 5:] = False
    target = cv.copy()
    target[~valid] = np.nan
    y = labels(candidate, ref, target, valid, 1.)
    np.testing.assert_allclose(y[:, 0]-y[:, 1],
                               pair_reduction(candidate, ref, target, valid)[:, 1], atol=1e-7)
    assert (y[:, 1] > 0.).all()
    without_neighbor = labels(candidate[:, :2], ref[:2], target[:2], valid[:2], 1.)
    np.testing.assert_array_equal(without_neighbor[:, 1], 0.)
    missing = labels(candidate, ref, target, np.zeros_like(valid), 1.)
    np.testing.assert_array_equal(missing[:, 0], 0.)
    assert (missing[:, 1] > 0.).all()


def test_score_symmetry_and_causal_identical_anchor():
    h, cv, ols, candidate = inputs()
    net = RelativeGain()
    ref = candidate[1]
    x, xr, bound, _, _ = features(h, cv, ols, candidate, ref)
    a, b, u = (torch.from_numpy(v) for v in (x, xr, bound))
    out, reverse = net(a, b, u), net(b, a, u)
    torch.testing.assert_close(out[:, 0], -reverse[:, 0])
    torch.testing.assert_close(out[:, 1], reverse[:, 1])
    torch.testing.assert_close(out[1], torch.zeros_like(out[1]))
    assert torch.all(out[:, 0].abs() <= u) and torch.all((out[:, 1] >= 0.) & (out[:, 1] <= u))
    perm = [2, 0, 1]
    new = features(h[perm]+14., cv[perm], ols[perm], candidate[:, perm], ref[perm])
    for a, b in zip((x, xr, bound), new[:3], strict=True):
        np.testing.assert_allclose(a, b, atol=1e-7)


def test_legal_two_edit_path_retains_distinct_recipients_and_peaks():
    _, cv, _, p = inputs()
    bank = p.swapaxes(0, 1)
    paths, row, arm, opts = legal(cv, bank, np.array([.1, .3]))
    candidates = assemble(cv, paths, row, arm, opts)
    assert np.all(row[opts[:, 0]] != row[opts[:, 1]])
    peak = np.linalg.norm(candidates - cv, axis=-1).max(-1)
    np.testing.assert_allclose(np.sort(peak, axis=1), np.tile([0., .1, .3], (len(peak), 1)), atol=1e-12)
