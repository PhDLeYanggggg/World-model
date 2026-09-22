from argparse import Namespace
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.run_m3w_tempered_cost import validate_args, validate_config, causal_scores, empirical_gate
from scripts.run_m3w_bounded_cost import selections
from scripts.verify_m3w_tempered_cost import expected_choices, check_contrast
from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast


def args(**kw):
    return Namespace(**(dict(audit_only=False, resume=False, evaluate=False, verify=False, view=None, stop_at=None)|kw))


@pytest.mark.parametrize('changes', [dict(evaluate=True, resume=True), dict(audit_only=True, view='coupa_seed17'),
    dict(verify=True, stop_at=100), dict(evaluate=True, verify=True), dict(stop_at=100), dict(view='x', stop_at=0)])
def test_phase_separation(changes):
    with pytest.raises(ValueError): validate_args(args(**changes))


def test_allowed_phases_and_fixed_contract():
    for kw in (dict(view='coupa_seed17', stop_at=100), dict(resume=True), dict(evaluate=True), dict(verify=True)):
        validate_args(args(**kw))
    cfg = json.loads(Path('configs/m3w_tempered_cost_v1.json').read_text())
    parent = json.loads(Path('configs/m3w_eqmotion_cost_refit_v1.json').read_text())
    validate_config(cfg, parent)
    for field, value in (('loss_exponent', 2), ('threshold_search', True), ('closed_role_readout', True),
                         ('primary_reference', 'best_test_arm'), ('bootstrap_resamples', 100)):
        with pytest.raises(ValueError): validate_config(cfg|{field:value}, parent)


@pytest.mark.parametrize('key', ['target', 'valid', 'future_endpoint', 'complete_future'])
def test_inference_rejects_outcome_arrays(key):
    with pytest.raises(ValueError): causal_scores({key:np.ones(1)}, None, None, None, None)


def test_choices_match_independent_reconstruction_with_frozen_count():
    rng = np.random.default_rng(42); n = 103
    score = rng.uniform(0, 5, (n, 2)); ids = rng.permutation(n)
    past = rng.normal(size=(n, 8, 2)); d = rng.uniform(.1, 5, n)
    past[:3, -1] = past[:3, -2]; d[3] = 0
    score[4:7] = [2, 0]; ids[4:7] = sorted(ids[4:7], reverse=True)
    allowed = np.any(past[:, -1] != past[:, -2], axis=1) & (d > 0)
    anchor = np.zeros(n, bool); anchor[4:15] = True
    a = selections(score, past, d, anchor, ids); b = expected_choices(score, allowed, ids, anchor)
    for key in a: np.testing.assert_array_equal(a[key], b[key])
    assert a['matched_count'].sum() == anchor.sum() and not a['matched_count'][:4].any()
    bad = anchor.copy(); bad[0] = True
    with pytest.raises(ValueError): expected_choices(score, allowed, ids, bad)


def test_primary_cannot_ignore_one_seed_or_zero_reference_harm():
    seed = dict(zero_CV_harmed=0, subsets={'positive_easy':{'equal_scene_gain_percent':-1.}},
                ADE={'equal_scene_gain_percent':2.})
    p = dict(seeds={str(k):copy.deepcopy(seed) for k in (17, 29, 43)})
    assert all(empirical_gate(p, {'ci95_pp':[.01, 1.]}).values())
    p['seeds']['29']['zero_CV_harmed'] = 1
    p['seeds']['43']['subsets']['positive_easy']['equal_scene_gain_percent'] = -2.00001
    g = empirical_gate(p, {'ci95_pp':[0., 1.]})
    assert not g['exact_zero'] and not g['easy'] and not g['positive_primary_ci']


def test_separate_paired_scene_bootstrap_check_detects_changed_interval():
    a = dict(a={'gain_percent':1.}, b={'gain_percent':4.}, c={'gain_percent':2.})
    b = {k:{'gain_percent':.5} for k in a}
    r = paired_scene_contrast([1., 4., 2.], [.5, .5, .5])
    check_contrast(a, b, list(a), r)
    with pytest.raises(AssertionError): check_contrast(a, b, list(a), r|{'ci95_pp':[0., 0.]})
