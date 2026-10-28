import numpy as np
import pytest
import torch

from src.world_model import m3w_constraint_scale as api
from tests.test_m3w_easy_harm_deviance_training import data, settings
from tests.test_m3w_task_risk_training import CONFIG


def test_query_normalization_and_missing_support():
    x, env, y, series, sites, rec, frames, pr = data()
    y[:, 2] = 1.; y[:, 3] = .4; y[-1] = np.nan
    np.testing.assert_allclose(api.train_budget_scales(y, sites, rec, frames, 2.), [.01, .004])
    y[:-1, 3] = 0
    with pytest.raises(ValueError): api.train_budget_scales(y, sites, rec, frames, 2.)


def test_positive_scaling_preserves_feasibility_and_gradient():
    x = torch.tensor([.3, -.4], dtype=torch.float64, requires_grad=True)
    scales = torch.tensor([.02, .0004], dtype=torch.float64)
    v = api.scaled_excess(x, scales)
    assert torch.equal(x <= 0, v <= 0)
    torch.testing.assert_close(torch.autograd.grad(v.sum(), x)[0], 1/scales)
    torch.autograd.gradcheck(lambda z:api.scaled_excess(z, scales), (x,))
    with pytest.raises(ValueError): api.scaled_excess(x, torch.zeros(2))
    with pytest.raises(ValueError): api.scaled_excess(x, scales.requires_grad_())


def test_gradient_diagnosis_replays_original_TRAIN_monitor(tmp_path):
    args=data(); eligible=np.ones(24, bool)
    state=api.old.fit(*args, eligible=eligible, arm='task_risk', settings=settings(),
        objective_config=CONFIG, seed=17, identity={'test':True}, experiment_sha256='test',
        path=tmp_path/'head.pt.gz', heartbeat=lambda **_:None)
    result=api.frozen_monitor(state,args,eligible)
    assert result['monitor_replayed'] and result['gradient_norms']['primary']>0
    assert result['TRAIN_budget_scales'][1]>0
    assert not result['new_training'] and not result['independent_roles_read']
