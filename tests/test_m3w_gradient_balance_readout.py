import copy
import numpy as np
import pytest
from src.evaluation import m3w_gradient_balance_readout as api
from tests.test_m3w_active_bank_readout import fixture


def group(source='s0', missing=False):
    packets, _, _, p = fixture(source, missing)
    choices = {o: np.full(2, 6 if o == 'balanced_score' else -1) for o in api.OBJECTIVES}
    return api.readout(packets['new_neural'], {o: p['new_neural'] for o in api.OBJECTIVES}, choices,
        {o: np.ones(2) for o in api.OBJECTIVES}, 1.5, np.ones(5))


def records(missing=False):
    contexts = {f'c{i}': [f's{4*(i%3)+j}' for j in range(4)] for i in range(6)}
    rows = [dict(context=c, source=s, seed=seed, controls_exact=True, result=copy.deepcopy(group(s, missing)))
        for c, sources in contexts.items() for s in sources for seed in (17, 29, 43)]
    return rows, contexts, dict(seeds=[17, 29, 43], bootstrap_seed=4, bootstrap_resamples=32)


def test_known_gain_and_passing_fitting_gate_never_promote_deployment_or_confirmation():
    rows, contexts, cfg = records()
    result = api.summarize(rows, contexts, cfg, dict(count=18, cached_verified_controls=[None]*36))
    assert result['fitting_advancement_gate']
    assert result['accuracy']['balanced_score']['point'] == pytest.approx(60.)
    assert not result['independent_confirmation'] and not result['deployment_changed']
    assert result['comparisons']['balanced_minus_admitted']['matched_known_pp']['point'] is None
    for bad in (rows[:-1], rows[:-1]+rows[:1]):
        with pytest.raises(ValueError):
            api.summarize(bad, contexts, cfg, dict(count=18, cached_verified_controls=[None]*36))


def test_unknown_sensitivity_and_missing_fde_are_not_hidden_by_positive_known_gain():
    rows, contexts, cfg = records(missing=True)
    rows[0]['result']['policies']['balanced_score']['known_FDE_gain_floor_percent'] = None
    result = api.summarize(rows, contexts, cfg, dict(count=18, cached_verified_controls=[None]*36))
    assert result['accuracy']['balanced_score']['point'] > 0
    assert not result['fitting_gates']['complete_recording_risk']
    assert not result['fitting_advancement_gate'] and result['FDE']['balanced_score']['point'] is None
    assert not result['stage5c_executed'] and not result['smc_enabled']
