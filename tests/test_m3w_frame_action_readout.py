import copy
import numpy as np
import pytest
from src.evaluation import m3w_frame_action_readout as api
from tests.test_m3w_active_bank_readout import fixture


def group(source='s0', missing=False):
    packet = fixture(source, missing)[0]['new_neural']
    choices = {a: np.full(2, 6 if a != 'independent' else -1) for a in api.allocation.ARMS}
    ranks = {a: np.ones(2) for a in choices}
    info = dict(queries=2, multiagent_queries=0, nonoptimal_queries={}, changed_queries={'joint_matched': 2})
    oracle = np.array([6, -1 if missing else 6])
    return api.readout(packet, choices, ranks, info, 1.5, np.zeros((2, 2)), oracle)


def test_positive_known_result_cannot_hide_missing_risk_or_count_mismatch():
    r = group(missing=True)
    assert r['policies']['joint_matched']['known_ADE_gain_floor_percent'] == pytest.approx(60.)
    assert r['comparisons']['joint_matched_minus_independent']['full']['lower_pp'] == pytest.approx(0.)
    assert r['comparisons']['joint_matched_minus_independent']['matched']['known_pp'] is None
    assert not r['exact_count_per_query']
    assert not r['policies']['joint_matched']['policy']['budget_status']['supported']
    assert r['oracle_frame_check']['defined_easy_support'] == 1
    assert r['oracle_frame_check']['algebraic_budget_failures'] == 0
    assert r['oracle_frame_check']['label_aware_not_deployable']
    assert r['pair_proxy']['mean_positive_distance_contraction_over_frozen_scale']['floor'] is None


def test_full_roster_verification_keeps_independent_and_deployment_claims_false():
    contexts = {f'c{i}': [f's{4*(i%3)+j}' for j in range(4)] for i in range(6)}
    rows = [dict(context=c, source=s, seed=seed, control_exact=True, record_oracle_exact=True,
        result=copy.deepcopy(group(s, True))) for c, ss in contexts.items() for s in ss for seed in (17, 29, 43)]
    cfg = dict(seeds=[17, 29, 43], bootstrap_seed=4, bootstrap_resamples=32)
    result = api.summarize(rows, contexts, cfg)
    assert result['new_training_updates'] == 0 and result['cached_verified_cost_heads'] == 18
    assert not result['fitting_gates']['same_intervention_count_every_query']
    assert not result['fitting_advancement_gate'] and not result['independent_confirmation']
    assert not result['deployment_changed'] and not result['submission_ready']
    assert not result['stage5c_executed'] and not result['smc_enabled']
    with pytest.raises(ValueError):
        api.summarize(rows[:-1], contexts, cfg)
