import numpy as np

from scripts.run_m3w_causal_menu_control import ARMS, LEARNED, choose, matched, remap, run


def test_remap_does_not_replace_fallback_or_choose_another_family():
    np.testing.assert_array_equal(remap(np.array([1, -1, 0]), np.array([2, 7])), [7, -1, 2])


def test_count_mismatch_abstains_for_both():
    a, b = matched(np.array([0, -1]), np.array([0, 1]))
    assert (a == -1).all() and (b == -1).all()


def test_same_caps_and_own_vs_common_screen():
    n, k = 4, len(run.NAMES)
    h = np.zeros((n, 8, 2))
    h[:, :, 0] = np.arange(n)[:, None]+np.arange(8)[None, :]
    cv = np.zeros((n, 12, 2))
    bank = cv[:, None]+np.arange(1, k+1)[None, :, None, None]
    crowd = np.full(n, n)
    x, _ = run.model.descriptors(h, bank, cv, crowd)
    coefficient = np.zeros((k, x.shape[-1], 6))
    coefficient[:, 0, [0, 3]] = [.01, 2.]
    coefficient[LEARNED, 0, 0] = .2
    coefficient[LEARNED, 0, 4] = 1.
    values, mismatch = choose(h, bank, cv, crowd, coefficient)
    got = dict(zip(ARMS, values))
    assert mismatch == 0
    assert (got['mixed_joint'] >= 0).sum() == (got['causal_joint'] >= 0).sum() == 2
    assert np.isin(got['mixed_joint'], LEARNED).sum() == 2
    assert not np.isin(got['causal_joint'], LEARNED).any()
    assert (got['mixed_screen'] >= 0).sum() == 2
    assert (got['causal_screen'] == -1).all()
    assert (got['mixed_common_screen'] == -1).all()
