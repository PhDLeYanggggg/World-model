import copy
import numpy as np
import pytest
from src.evaluation import m3w_shared_count_readout as api
from tests.test_m3w_active_bank_readout import fixture


def group(source='s0', missing=False):
    packet = fixture(source, missing)[0]['new_neural']
    packet['allowed'][:, 0] = True
    packet['rollouts'][:, 0, :, 0] = 1.2
    known = np.isfinite(packet['floor_ade'])
    packet['ade'][known, 0] = packet['fde'][known, 0] = 1.2
    packet['y'][known, 0, 1] = packet['y'][known, 0, 4] = .2
    packet['envelope'][:, 0] = .2
    p = np.zeros((2, 7, 5)); p[..., 2:4] = 1.
    p[:, 0, 0] = .1; p[:, 6, 0] = .2; p[..., 1] = p[..., 4] = .001
    choices, ranks, info = api.allocation.allocate(p, packet['allowed'], packet['recordings'],
        packet['frames'], packet['ids'], seconds=2.)
    return api.readout(packet, choices, ranks, info, np.full(2, 6), np.full(2, 6), 1.5, np.zeros((2, 2)))


def test_four_policy_bound_cancels_shared_unknown_effects():
    packet = fixture(missing=True)[0]['new_neural']
    choices = dict(physical_independent=np.full(2, -1), physical_joint=np.full(2, -1),
        neural_trimmed=np.full(2, 6), neural_joint=np.full(2, 6))
    result = api.interaction_contrast(packet, choices)
    assert result['known_pp'] == 0 and result['lower_pp'] == 0 and result['unknown_disagreement_mass'] == 0
    choices['neural_trimmed'][:] = -1
    result = api.interaction_contrast(packet, choices)
    assert result['known_pp'] == pytest.approx(60.)
    assert result['lower_pp'] == pytest.approx(0.)
    assert result['own_four_policy_bound'] and result['not_difference_of_lower_bounds']
    assert not result['causal_identification']


def test_reoptimized_primary_and_unchanged_full_risk_contract():
    r = group(missing=True)
    assert r['same_count_every_query'] and r['full_cohort_preserved']
    assert r['policies']['physical_joint']['known_ADE_gain_floor_percent'] == pytest.approx(-20.)
    assert r['policies']['neural_joint']['known_ADE_gain_floor_percent'] == pytest.approx(60.)
    assert not r['policies']['neural_joint']['policy']['budget_status']['supported']
    assert r['common_count_not_retroactive_deployment_repair']


def test_complete_fitting_gate_never_becomes_independent_or_deployment_claim():
    contexts = {f'c{i}': [f's{4*(i%3)+j}' for j in range(4)] for i in range(6)}
    rows = [dict(context=c, source=s, seed=seed, legacy_controls_exact=True,
        result=copy.deepcopy(group(s))) for c, ss in contexts.items() for s in ss for seed in (17, 29, 43)]
    cfg = dict(seeds=[17, 29, 43], bootstrap_seed=4, bootstrap_resamples=32)
    s = api.summarize(rows, contexts, cfg)
    assert s['fitting_advancement_gate']
    assert s['new_training_updates'] == 0 and not s['independent_confirmation']
    assert not s['deployment_changed'] and not s['submission_ready']
    assert not s['stage5c_executed'] and not s['smc_enabled']
    with pytest.raises(ValueError):
        api.summarize(rows[:-1], contexts, cfg)
