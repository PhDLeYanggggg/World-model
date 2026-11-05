import numpy as np
import pytest
import torch
from src.world_model import m3w_scale_ablated_risk as api
from tests.test_m3w_admission_priority import prepared


def test_runner_uses_original_policy_function():
    from scripts import run_m3w_moving_scale_ablation as run
    assert run.choose is api.parent.parent.parent.choose
    assert callable(run.choose)


def test_only_one_input_changes():
    x = torch.randn(3, 7, 428)
    c = {k: None for k in api.base.CAUSAL_KEYS}
    d = dict(causal=c, z=x)
    z = api.prepare(d)['z']
    columns = np.arange(428) != api.COLUMN
    assert torch.equal(z[..., columns], x[..., columns])
    assert not z[..., api.COLUMN].any()
    assert x[..., api.COLUMN].any()


def test_real_training_exact_resume_control_and_scale_independence(tmp_path):
    torch.set_num_threads(1); d, warm, settings = prepared(tmp_path)
    for arm in api.ARMS:
        kw = dict(arm=arm, settings=settings, identity={'fixed': True})
        control = api.parent.fit(d, warm, **kw, path=tmp_path/(arm+'_control.pt.gz'))
        path = tmp_path/(arm+'.pt.gz')
        api.fit(d, warm, **kw, path=path, stop_at=2)
        state = api.fit(d, warm, **kw, path=path, resume=True)
        repeat = api.fit(d, warm, **kw, path=tmp_path/(arm+'_repeat.pt.gz'))
        api.parent.parent.parent.parent.router.exact_state(state, repeat)
        api.assert_control(state, control, d)
        p = api.predict(state, d['causal'])
        c = dict(d['causal'], x=d['causal']['x'].copy())
        c['x'][..., api.COLUMN] += 10000
        np.testing.assert_array_equal(p, api.predict(state, c))
        truth = d['truth_by_arm']['bound_aware'].clone()
        d['truth_by_arm']['bound_aware'][:] = 12345
        np.testing.assert_array_equal(p, api.predict(state, d['causal']))
        d['truth_by_arm']['bound_aware'][:] = truth
        with pytest.raises(ValueError, match='causal inputs'):
            api.predict(state, dict(c, future_endpoint=np.zeros((4, 2))))
