import numpy as np
import pytest

from src.world_model import m3w_donor_motion_shape as model
from src.world_model.m3w_sustained_turn import choose as previous


def sample():
    rng = np.random.default_rng(12)
    h = rng.normal(size=(20, 8, 2)).cumsum(1)
    y = rng.normal(size=(20, 12, 2)).cumsum(1)
    return h, y


def test_reject_held_site_and_use_only_donor_complete_labels():
    h, y = sample()
    valid = np.ones(y.shape[:2], bool)
    valid[-1, -1] = False
    y[-1] = np.nan
    donors = {"A": dict(h=h, y=y, valid=valid)}
    with pytest.raises(ValueError, match="disjoint"):
        model.Library.fit(donors, "A", 8)
    assert model.Library.fit(donors, "B", 8).tree.n == len(h)-1
    a = model.Library.fit(donors, "B", 4).predict(h[:, :4])[0]
    b = model.Library.fit({"A": {"h": h}}, "B", 4).predict(h[:, :4])[0]
    np.testing.assert_array_equal(a, b)


def test_null_preserves_residuals_and_retrieval_is_rotation_equivariant():
    h, y = sample()
    lib = model.Library.fit({"A": dict(h=h, y=y, valid=np.ones(y.shape[:2], bool))}, "B", 8)
    np.testing.assert_array_equal(np.sort(lib.offsets[:, 0, 0]), np.sort(lib.null_offsets[:, 0, 0]))
    p, d = lib.predict(h)
    rot = np.array([[0., 1.], [-1., 0.]])
    q, e = lib.predict(h @ rot + [50., 13.])
    np.testing.assert_allclose(q, p @ rot, atol=1e-11)
    np.testing.assert_allclose(d, e, atol=1e-11)


def test_prefix_gate_replays_legacy_anchor_and_matches_radius():
    h, y = sample()
    lib = {k: model.Library.fit({"A": dict(h=h, y=y, valid=np.ones(y.shape[:2], bool))}, "B", k)
           for k in (*model.PREFIXES, 8)}
    amp = np.full(len(h), .5)
    offsets = {k: v.predict(h[:, :k])[0] for k, v in lib.items()}
    cv, anchor, candidates, used, mask = model.choose(h, amp, offsets)
    np.testing.assert_array_equal(anchor, previous(h, amp)[0])
    np.testing.assert_array_equal(used[~mask], np.broadcast_to(anchor[:, None], used.shape)[~mask])
    radius = model.peak(anchor-cv)
    for bank in (candidates, used):
        np.testing.assert_allclose(model.peak(bank-cv[:, None]), np.broadcast_to(radius[:, None], mask.shape))
