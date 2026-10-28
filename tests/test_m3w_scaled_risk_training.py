import numpy as np
import pytest
import torch

from src.world_model import m3w_scaled_risk_training as api
from tests.test_m3w_easy_harm_deviance_training import data,settings
from tests.test_m3w_task_risk_training import CONFIG


def fit(path,mode,**kwargs):
    return api.fit(*data(),mode=mode,eligible=np.ones(24,bool),settings=settings(),
        objective_config=CONFIG,seed=17,identity={'test':True},experiment_sha256='test',
        path=path,heartbeat=lambda **_:None,**kwargs)


def test_unit_context_exact_original_and_restoration(tmp_path):
    objective=api.old.decision.objective;update=api.old.decision.update_dual
    prior=api.old.fit(*data(),arm='task_risk',eligible=np.ones(24,bool),settings=settings(),
        objective_config=CONFIG,seed=17,identity={'test':True},experiment_sha256='old',
        path=tmp_path/'old.pt.gz',heartbeat=lambda **_:None)
    control=fit(tmp_path/'unit.pt.gz','unit_control');api.assert_control(control,prior)
    assert api.old.decision.objective is objective and api.old.decision.update_dual is update
    with pytest.raises(ValueError),api.constraint_units(torch.zeros(2)):pass
    assert api.old.decision.objective is objective and api.old.decision.update_dual is update


@pytest.mark.parametrize('mode',api.MODES)
def test_scaled_resume_is_exact_and_mode_is_bound(tmp_path,mode):
    direct=fit(tmp_path/'direct.pt.gz',mode)
    fit(tmp_path/'resume.pt.gz',mode,stop_at=3)
    replay=fit(tmp_path/'resume.pt.gz',mode,resume=True)
    for key in direct:
        if key!='seconds':api.old.core.exact(direct[key],replay[key])
    with pytest.raises(AssertionError):
        fit(tmp_path/'resume.pt.gz',api.MODES[1-api.MODES.index(mode)],resume=True)


def test_scaling_changes_duals_but_not_sampler_or_initialization(tmp_path):
    states=[fit(tmp_path/(m+'.pt.gz'),m) for m in api.MODES]
    api.assert_matched(states)
    assert not torch.equal(states[0]['dual'],states[1]['dual'])
    assert not torch.equal(states[0]['model']['primary.output.weight'],states[1]['model']['primary.output.weight'])
    assert states[1]['identity']['constraint_scales']==states[1]['identity']['TRAIN_budget_scales']
    assert states[1]['unknown_rows_sampled']==0


def test_nested_context_rejected():
    with api.constraint_units(torch.ones(2)):
        with pytest.raises(RuntimeError),api.constraint_units(torch.ones(2)):pass
