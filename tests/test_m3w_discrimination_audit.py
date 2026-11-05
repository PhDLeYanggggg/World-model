import numpy as np
import pytest
from src.evaluation.m3w_discrimination_audit import audit, summarize


def test_ranking_and_boundary_are_different():
    r = summarize([1, 2, 3, 4], [-1, -2, 1, 2], [1, 1, 1, 1])
    assert r['query_weighted_AUROC'] == 1
    assert r['beneficial_mass_rejected_fraction'] == 1
    assert r['harmful_mass_admitted_fraction'] == 0
    assert r['no_threshold_search']


def test_zero_support_does_not_create_rank_evidence():
    r = summarize([0, 1], [0, 1], [1, 1])
    assert r['query_weighted_AUROC'] is None
    assert r['zero_target_rows'] == 1
    assert r['beneficial_rows'] == 0


def test_unknown_outcomes_and_unproposed_actions_do_not_enter():
    risk = np.zeros((4, 7, 2)); decision = risk.copy(); target = risk.copy()
    target[:, 1] = np.array([-1, 1, -1, 50])[:, None]
    allowed = np.zeros((4, 7), bool); allowed[:, 1] = True
    observed = np.array([True, True, True, False]); w = np.ones(4); sites = np.array(['a']*4)
    result = audit(risk, decision, target, allowed, observed, w, sites)
    target[3] = -1e9; target[:, 5] = 1e9
    assert audit(risk, decision, target, allowed, observed, w, sites) == result
    assert result['total']['combined']['nonzero_supported_rows'] == 3
    assert result['unknown_outcomes_excluded'] == 1


def test_rejects_multiple_actions_and_invalid_weights():
    with pytest.raises(ValueError):
        summarize([1, 2], [-1, 1], [-1, 1])
    r = np.zeros((1, 7, 2))
    with pytest.raises(ValueError):
        audit(r, r, r, np.ones((1, 7), bool), np.ones(1, bool), np.ones(1), np.array(['a']))
