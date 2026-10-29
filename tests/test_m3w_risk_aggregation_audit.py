import numpy as np
import pytest

from src.world_model import m3w_risk_aggregation_audit as api
from tests.test_m3w_easy_harm_deviance_training import data, settings
from tests.test_m3w_task_risk_training import CONFIG


def test_variable_query_sizes_change_feasibility():
    y = np.array([[0.,1.,100.,100.,1.]]+[[0.,.22,1.,1.,.22]]*10)
    result = api.compare_aggregation(y, np.ones(11), np.array([0]+[1]*10), 2)
    for record in result['constraints'].values():
        assert record['query_pass_mass_fail']
        assert record['agent_mass_risk'] == pytest.approx(3.2/110)


def test_equal_size_queries_preserve_sign_and_common_units():
    y = np.array([[1.,.3,2.,1.,.2]]*8)
    ids = np.repeat(np.arange(4), 2)
    a = np.linspace(.1, .8, 8)
    plain = api.compare_aggregation(y, a, ids, 4)
    scaled = api.compare_aggregation(y*17, a, ids, 4)
    for key, record in plain['constraints'].items():
        assert record['query_mean_excess'] == pytest.approx(record['agent_mass_mean_excess'])
        assert record['agent_mass_risk'] == pytest.approx(scaled['constraints'][key]['agent_mass_risk'])


def test_empty_and_missing_reference_never_mean_safe_support():
    y = np.zeros((2, 5))
    result = api.compare_aggregation(y, np.zeros(2), np.arange(2), 2)
    assert result['empty_action']
    assert all(not v['reference_support'] for v in result['constraints'].values())
    with pytest.raises(ValueError): api.compare_aggregation(y+np.nan, np.ones(2), np.arange(2), 2)
    with pytest.raises(ValueError): api.compare_aggregation(y, np.ones(2), np.zeros(2, int), 2)


def test_monitor_replay_and_input_binding(tmp_path):
    args = data(); eligible = np.ones(24, bool)
    state = api.old.fit(*args, eligible=eligible, arm='task_risk', settings=settings(),
        objective_config=CONFIG, seed=17, identity={'test':True}, experiment_sha256='test',
        path=tmp_path/'model.pt.gz', heartbeat=lambda **_:None)
    result = api.frozen_monitor(state, args, eligible)
    assert result['monitor_replayed'] and not result['validation_read']
    assert result['aggregations']['hard']['query_count'] == 128
    with pytest.raises(ValueError): api.frozen_monitor(state, args, ~eligible)
