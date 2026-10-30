import copy
import numpy as np
import pytest
from src.evaluation import m3w_selected_query_critic_readout as api
from tests.test_m3w_active_bank_readout import fixture


def group(source='s0', missing=False):
    packet = fixture(source, missing)[0]['new_neural']
    p = np.broadcast_to([.1, .001, 1., .5, .0001], (2, 7, 5)).copy()
    choices = {a: np.full(2, 6) for a in api.decoding.NAMES}
    info = dict(queries=2, same_count_controls_every_query=True, nonoptimal_queries={})
    return api.readout(packet, choices, {a: np.ones(2) for a in choices}, info,
        dict(row_local=p, selected_set=p), p, np.full(2, 6), np.full(2, 6),
        1.5, np.zeros((2, 2)), np.ones(5))


def test_known_improvement_is_not_safety_and_unknown_queries_stay_unknown():
    r = group(missing=True)
    assert r['policies']['set_joint']['known_ADE_gain_floor_percent'] == pytest.approx(60.)
    assert not r['policies']['set_joint']['policy']['budget_status']['supported']
    assert r['calibration']['selected_set']['complete_frozen_selected_queries'] == 1
    assert r['calibration']['selected_set']['selected_unknown_queries'] == 1
    assert not r['inference_label_filter'] and not r['unknown_targets_imputed']


def test_full_roster_and_no_false_promotion():
    contexts = {f'c{i}': [f's{4*(i%3)+j}' for j in range(4)] for i in range(6)}
    rows = [dict(context=c, source=s, seed=seed, result=copy.deepcopy(group(s, True)),
        parent_controls_exact=True, paired_training_verified=True)
        for c, sources in contexts.items() for s in sources for seed in (17, 29, 43)]
    result = api.summarize(rows, contexts, dict(seeds=[17, 29, 43], bootstrap_seed=3, bootstrap_resamples=32))
    assert result['fresh_critic_fits'] == 36 and result['critic_training_updates'] == 72000
    assert not result['fitting_advancement_gate'] and not result['independent_confirmation']
    assert not result['deployment_changed'] and not result['submission_ready']
    assert not result['stage5c_executed'] and not result['smc_enabled']
    with pytest.raises(ValueError):
        api.summarize(rows[:-1], contexts, dict(seeds=[17, 29, 43]))
