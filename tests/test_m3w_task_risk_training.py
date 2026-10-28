import copy

import numpy as np
import pytest
import torch

from src.world_model import m3w_task_risk_training as api
from tests.test_m3w_easy_harm_deviance_training import data, settings

CONFIG = dict(task_weight=1., temperature_start=1., temperature_end=.1,
              dual_learning_rate=1., dual_maximum=100., risk_budget=.02)


def fit(tmp_path, arm, name, **kwargs):
    return api.fit(*data(), eligible=np.ones(24, dtype=bool), arm=arm,
                   settings=settings(), objective_config=CONFIG, seed=17, identity={'test':True},
                   experiment_sha256='test', path=tmp_path/(name+'.pt.gz'), heartbeat=lambda **_:None, **kwargs)


def test_quadratic_matches_original_trainer_exactly(tmp_path):
    old = api.parent.fit(*data(), arm='none', settings=settings(), seed=17, identity={'test':True},
                         path=tmp_path/'original.pt.gz', heartbeat=lambda **_:None)
    api.assert_original_control(fit(tmp_path, 'quadratic', 'new'), old)


@pytest.mark.parametrize('arm', api.ARMS)
def test_exact_resume_including_duals_and_temperature(tmp_path, arm):
    direct = fit(tmp_path, arm, 'direct')
    fit(tmp_path, arm, 'resume', stop_at=3)
    resumed = fit(tmp_path, arm, 'resume', resume=True)
    for key in direct:
        if key != 'seconds': api.core.exact(direct[key], resumed[key])


def test_shared_draws_and_nontrivial_decision_update(tmp_path):
    states = [fit(tmp_path, arm, arm) for arm in api.ARMS]
    api.assert_matched(states)
    assert all(s['unknown_rows_sampled'] == 0 for s in states)
    assert not torch.equal(states[0]['model']['primary.output.weight'], states[1]['model']['primary.output.weight'])
    assert states[1]['dual'].sum() == 0
    assert states[2]['dual'].sum() > 0
    assert not states[2]['risk_calibration_claim']


def test_changed_arm_and_budget_cannot_resume(tmp_path):
    fit(tmp_path, 'task_utility', 'resume', stop_at=3)
    with pytest.raises(AssertionError): fit(tmp_path, 'task_risk', 'resume', resume=True)
    config = copy.deepcopy(CONFIG); config['risk_budget'] = .03
    with pytest.raises(ValueError): api.validate_objective(config)


def test_disk_guard_precedes_computation(tmp_path):
    def stop(): raise OSError('storage reserve')
    with pytest.raises(OSError): fit(tmp_path, 'task_risk', 'blocked', checkpoint_guard=stop)
    assert not list(tmp_path.iterdir())
