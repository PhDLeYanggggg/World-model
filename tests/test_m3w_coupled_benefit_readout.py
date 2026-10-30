import copy
import numpy as np
import pytest
from src.evaluation import m3w_coupled_benefit_risk as api
from scripts import report_m3w_coupled_benefit_risk as reporting
from tests.test_m3w_active_bank_readout import fixture


def group(source='s0', missing=True):
    packet = fixture(source, missing)[0]['new_neural']
    p = np.broadcast_to([.1, .001, 1., .5, .0001], (2, 7, 5)).copy()
    choices = {a: np.full(2, 6) for a in api.decoder.NAMES}
    return api.readout(packet, choices, {a: np.ones(2) for a in choices},
        dict(queries=2, same_count_controls_every_query=True),
        {a: p for a in api.decoder.MODES}, 1.5, np.zeros((2, 2)), np.ones(5))


def summary():
    contexts = {f'c{i}': [f's{4*(i%3)+j}' for j in range(4)] for i in range(6)}
    rows = [dict(context=c, source=s, seed=seed, result=copy.deepcopy(group(s)),
        old_control_exact=True, paired_training_verified=True)
        for c, ss in contexts.items() for s in ss for seed in (17, 29, 43)]
    cfg = dict(seeds=[17, 29, 43], bootstrap_seed=3, bootstrap_resamples=32)
    return rows, contexts, cfg, api.summarize(rows, contexts, cfg)


def test_unknown_improvement_does_not_become_safe_or_independent():
    r = group()
    assert r['policies']['released_joint']['known_ADE_gain_floor_percent'] > 0
    assert r['policies']['released_joint']['policy']['selected_unknown'] > 0
    assert not r['policies']['released_joint']['policy']['budget_status']['supported']
    assert not r['future_inputs_used'] and not r['world_dynamics_training']
    rows, contexts, cfg, s = summary()
    assert not s['fitting_advancement_gate'] and not s['independent_confirmation']
    assert not s['deployment_changed'] and not s['stage5c_executed'] and not s['smc_enabled']
    with pytest.raises(ValueError): api.summarize(rows[:-1], contexts, cfg)


def test_report_does_not_promote_fitting_or_missing_endpoint_evidence():
    s = summary()[-1]
    s['FDE'] = {a: dict(point=None, CI95=None) for a in api.NAMES}
    text = reporting.render(dict(summary=s))
    for required in ('not_estimable', 'not independent', 'not a safety replacement'):
        assert required in text.lower()
    assert 'new trajectory/dynamics training' in text and 'Deployment unchanged' in text
    assert 'reference predictions r/easy_r' in text.lower() and '36 real Torch cost-head fits' in text
