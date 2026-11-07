import numpy as np
import pytest

from src.world_model import m3w_feasible_slate as model
from src.world_model import m3w_native_support_retrieval as prior


def test_exact_options_cover_both_registered_controls():
    r = np.random.default_rng(18)
    for n in (2, 5, 20):
        legal = np.ones((n, 4), bool)
        legal[0] = False
        original = np.full(n, -1)
        original[-min(n-1, 2):] = 1
        u, scale, support = r.normal(size=legal.shape), r.uniform(1, 4, n), r.random(n)
        slate, rows = model.options(original, u, scale, legal)
        gain, sup = prior.rerank(original, u, scale, legal, support)
        np.testing.assert_array_equal(slate[0], gain)
        np.testing.assert_array_equal(slate[np.argmax(support[rows])], sup)
        assert ((slate >= 0).sum(1) == (original >= 0).sum()).all()
        assert len(slate) <= 8


def test_matched_gate_keeps_query_mask_and_respects_same_eligibility():
    p = np.array([.9, .6, .4])
    mask = np.array([False, True, True])
    assert model.choose(p, mask, True) == (0, 1)
    assert model.choose(p, mask, False) == (0, None)
    with pytest.raises(ValueError):
        model.choose(p, ~np.ones(3, dtype=bool), True)
