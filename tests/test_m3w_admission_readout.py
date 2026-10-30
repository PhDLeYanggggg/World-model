import copy
import numpy as np
import pytest
from src.evaluation import m3w_admission_readout as api
from tests.test_m3w_active_bank_readout import fixture


def group(source='s0', missing=False):
    packets, _, _, prediction = fixture(source, missing)
    packet = packets['new_neural']
    choices = {o: np.full(2, 6 if o == 'admitted_score' else -1) for o in api.OBJECTIVES}
    return api.readout(packet, {o: prediction['new_neural'] for o in api.OBJECTIVES}, choices,
        {o: np.ones(2) for o in api.OBJECTIVES}, 1.5, np.ones(5))


def test_no_policy_core_changed_and_empty_match_is_not_estimable():
    result = group()
    assert result['policies']['admitted_score']['known_ADE_gain_floor_percent'] == pytest.approx(60.)
    assert result['comparisons']['admitted_minus_upper']['full']['known_pp'] == pytest.approx(60.)
    assert result['comparisons']['admitted_minus_upper']['matched']['known_pp'] is None
    assert result['comparisons']['admitted_minus_upper']['matched_role'] == 'utility_only_not_safety'
    assert result['attribution']['admitted_score']['selected_known_rows'] == 2
    assert result['attribution']['admitted_score']['future_informed_diagnostic_only']


def rows(missing=False):
    contexts = {f'c{i}': [f's{4*(i%3)+j}' for j in range(4)] for i in range(6)}
    values = [dict(context=c, source=s, seed=seed, controls_exact=True, result=copy.deepcopy(group(s, missing)))
        for c, sources in contexts.items() for s in sources for seed in (17, 29, 43)]
    cfg = dict(seeds=[17, 29, 43], bootstrap_seed=4, bootstrap_resamples=32)
    return values, contexts, cfg


def test_complete_fitting_gate_never_implies_independent_or_deployed_success():
    values, contexts, cfg = rows()
    result = api.summarize(values, contexts, cfg, dict(count=54, exact_parent_controls=36))
    assert result['fitting_advancement_gate']
    assert result['accuracy']['admitted_score']['point'] == pytest.approx(60.)
    assert not result['independent_confirmation'] and not result['deployment_changed']
    assert not result['submission_ready'] and not result['stage5c_executed'] and not result['smc_enabled']
    for bad in (values[:-1], values[:-1] + values[:1]):
        with pytest.raises(ValueError):
            api.summarize(bad, contexts, cfg, dict(count=54, exact_parent_controls=36))


def test_unknown_outcomes_and_missing_fde_cannot_be_hidden_by_positive_known_utility():
    values, contexts, cfg = rows(missing=True)
    values[0]['result']['policies']['admitted_score']['known_FDE_gain_floor_percent'] = None
    result = api.summarize(values, contexts, cfg, dict(count=54, exact_parent_controls=36))
    assert result['accuracy']['admitted_score']['point'] > 0
    assert not result['fitting_gates']['complete_recording_risk']
    assert not result['fitting_advancement_gate']
    assert result['FDE']['admitted_score']['point'] is None
