import numpy as np
import pytest
from scripts.verify_m3w_eqmotion_cost_refit import expected_choices
from src.world_model.m3w_bounded_cost_head import ARMS


def test_separate_rules_use_frozen_anchor_and_stable_id_ties():
    scores = {f+'_'+a:np.tile([1., 0.], (4, 1)) for f in ('frozen', 'refit') for a in ARMS}
    scores['frozen_bounded_fraction'][1:, 1] = 2
    allowed = np.array([True, True, True, False]); ids = np.array([9, 2, 5, 1])
    out = expected_choices(scores, allowed, ids)
    assert len(out) == 21
    np.testing.assert_array_equal(out['refit_direct_native_strict_stop'], allowed)
    np.testing.assert_array_equal(out['refit_direct_native_matched_count'], [False, True, False, False])
    assert all(v.sum() == 1 for k, v in out.items() if k.endswith('matched_count'))


def test_separate_rule_rejects_nonfinite_costs():
    with pytest.raises(ValueError): expected_choices({'frozen_bounded_fraction':np.full((2, 2), np.nan)},
                                                      np.ones(2, bool), np.arange(2))
