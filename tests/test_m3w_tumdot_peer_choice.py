import numpy as np

from src.world_model import m3w_tumdot_peer_choice as m


def test_peer_features_are_query_local_and_translation_invariant():
    rng = np.random.default_rng(42)
    h = rng.normal(size=(5, 8, 2))
    b = rng.normal(size=(5, 6, 12, 2))
    offsets = np.array([0, 2, 5])
    x, perm = m.contexts(h, offsets, b, 1)
    np.testing.assert_array_equal(perm, [1, 0, 4, 2, 3])
    assert x.shape == (5, 6, 11)
    h[2:] += 10.
    b[2:] += 20.
    y, _ = m.contexts(h + 30., offsets, b, 1)
    np.testing.assert_allclose(x[:2], y[:2], rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(x[:, 1, -2:], 0.)


def test_pair_error_uses_only_shared_valid_steps():
    h = np.zeros((2, 8, 2))
    p = np.zeros((2, 3, 2))
    y = p.copy()
    y[1, :, 0] = [3., 3., 100.]
    valid = np.array([[True, True, True], [True, True, False]])
    assert m.pair_error(p, y, valid, h, [0, 2]) == (6., 2)


def test_matching_uses_realized_changes_not_candidate_ids():
    from scripts.audit_m3w_tumdot_peer_budget import INC, match

    bank = np.zeros((4, 6, 2, 2))
    alt = (INC+1) % bank.shape[1]
    bank[0, alt] = 1.
    bank[2:, alt] = 1.
    choices = dict(a=np.array([alt, alt, alt, alt]), b=np.array([INC, alt, alt, INC]))
    scores = {k: np.zeros((4, 6, 3)) for k in choices}
    scores["a"][3, alt, 0] = 1.
    x = match(bank, scores, choices, np.array([0, 2, 4]))
    np.testing.assert_array_equal(x["a"], [INC, INC, INC, alt])
    np.testing.assert_array_equal(x["b"], [INC, INC, alt, INC])
