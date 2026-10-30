import copy
import numpy as np
import pytest

from src.evaluation import m3w_budget_score_readout as api
from tests.test_m3w_active_bank_readout import fixture


def values(missing=False):
    packets, selected, ranks, prediction = fixture(missing=missing)
    choices = {o: copy.deepcopy(selected) for o in api.OBJECTIVES}
    for v in api.bank.VARIANTS:
        choices['original'][v][:] = -1
        choices['symmetric_score'][v][:] = -1
    return api.objective_readouts(packets, {o: prediction for o in api.OBJECTIVES}, choices,
        {o: ranks for o in api.OBJECTIVES}, 1.5, np.ones(5))


def test_objective_pairing_retains_original_core_and_empty_utility_status():
    result = values()
    baseline = result['original']['result']
    for v in api.bank.VARIANTS:
        assert baseline['policies'][v]['policy']['selected_count'] == 0
    assert result['upper_score']['against_original']['new_neural']['full']['known_pp'] == pytest.approx(60.)
    assert result['upper_score']['against_original']['new_neural']['matched']['known_pp'] is None
    assert result['upper_score']['budget_diagnostics']['new_neural']['known_selected_views'] == 2
    assert not result['upper_score']['budget_diagnostics']['new_neural']['probability_calibration_or_risk_guarantee']


def records(missing=False):
    contexts = {f'c{i}': [f's{4 * (i % 3) + j}' for j in range(4)] for i in range(6)}
    shared = values(missing)
    rows = [dict(context=c, source=s, seed=seed, objective_arm=o,
        original_readout_exact=o == 'original', **copy.deepcopy(shared[o]))
        for c, sources in contexts.items() for s in sources for seed in (17, 29, 43) for o in api.OBJECTIVES]
    return rows, contexts, dict(seeds=[17, 29, 43], bootstrap_seed=4, bootstrap_resamples=16)


def test_all216groups_and_passing_fitting_gates_never_promote_independent_or_deployed_status():
    rows, contexts, cfg = records()
    result = api.summarize(rows, contexts, cfg, dict(count=162, exact_original_controls=54))
    assert result['fitting_advancement_gate']
    assert result['objectives']['upper_score']['against_original']['new_neural']['full_known_pp']['point'] == pytest.approx(60.)
    assert not result['independent_confirmation'] and not result['deployment_changed']
    assert not result['submission_ready'] and not result['stage5c_executed'] and not result['smc_enabled']
    for bad in (rows[:-1], rows[:-1] + rows[:1]):
        with pytest.raises(ValueError):
            api.summarize(bad, contexts, cfg, dict(count=162, exact_original_controls=54))


def test_unknown_support_keeps_conservative_gate_failed_despite_known_gain():
    rows, contexts, cfg = records(missing=True)
    result = api.summarize(rows, contexts, cfg, dict(count=162, exact_original_controls=54))
    assert result['objectives']['upper_score']['accuracy']['new_neural']['point'] > 0
    assert not result['fitting_gates']['complete_primary_recording_risk']
    assert not result['fitting_advancement_gate']


def test_exact_harm_prediction_with_inflated_reference_can_still_create_false_safe_switch():
    truth = np.tile(np.array([.2, .1, 1., .5, .05]), (1, 7, 1))
    predicted = truth.copy()
    predicted[..., 2:4] *= 10
    allowed = np.zeros((1, 7), bool)
    allowed[:, 6] = True
    choice, _ = api.bank.cost.select_actions(predicted, allowed, arm='physical_neural')
    assert choice.tolist() == [6]
    assert truth[0, 6, 1] / truth[0, 6, 2] > .02
    assert truth[0, 6, 4] / truth[0, 6, 3] > .02
    p = dict(y=truth, allowed=allowed, scales=np.ones(1))
    d = api.score_diagnostics(p, predicted, choice, 'new_neural', np.ones(5))
    assert d['selected_score_underprediction_views'] == [1, 1]
    assert all(v < 0 for v in d['selected_normalized_score_bias'])
