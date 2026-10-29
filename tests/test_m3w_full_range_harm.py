import numpy as np
import pytest

from src.world_model import m3w_full_range_harm as api
from src.world_model import m3w_isolated_harm_fp64 as old
from tests.test_m3w_isolated_harm_fp64 import anchor
from tests.test_m3w_decoder_risk_factorial import packets
from tests.test_m3w_shared_train_risk import settings, CONFIG, packet


def fit(path, a, implementation=api, **kwargs):
    return implementation.fit(packets(), anchor=a, anchor_sha256='frozen', arm='isolated_risk',
        settings=settings(), objective_config=CONFIG, identity={'test': True},
        experiment_sha256='new', path=path, heartbeat=lambda **_: None, **kwargs)


def test_exact_resume_and_matched_control(tmp_path):
    a = anchor(tmp_path)
    direct = fit(tmp_path/'direct.pt.gz', a)
    fit(tmp_path/'resume.pt.gz', a, stop_at=3)
    resumed = fit(tmp_path/'resume.pt.gz', a, resume=True)
    for key in direct:
        if key != 'seconds':
            api.core.exact(direct[key], resumed[key])
    control = fit(tmp_path/'old.pt.gz', a, implementation=old)
    api.assert_control_match(direct, control)


def test_protection_bounds_and_gradient(tmp_path):
    a = anchor(tmp_path)
    model = api.initialize(a, 'isolated_risk')
    old_model = old.initialize(a, 'isolated_risk')
    api.core.exact(model.state_dict(), old_model.state_dict())
    x, env = api.torch.randn(12, 380), api.torch.ones(12)
    api.torch.testing.assert_close(model(x, env), old_model(x, env), rtol=0, atol=0)
    (model(x, env)[:, 1]+model(x, env)[:, 4]).sum().backward()
    assert model.risk_output.bias.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.base.parameters())
    s = fit(tmp_path/'state.pt.gz', a)
    xx, ee = packet()[:2]
    out = api.predict(s, 'a', xx, ee)
    teacher = api.previous.predict(a, 'a', xx, ee)
    np.testing.assert_array_equal(out[:, [0, 2, 3]], teacher[:, [0, 2, 3]])
    assert np.isfinite(out).all() and (out >= 0).all()
    assert (out[:, :2].sum(1) <= ee+1e-6).all() and (out[:, 4] <= out[:, 1]+1e-8).all()
    with pytest.raises(ValueError):
        api.predict(s, 'unknown', xx, ee)


class TinyBase(api.torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.primary = api.torch.nn.Module()
        self.primary.encoder = api.torch.nn.Identity()

    def forward(self, z, envelope):
        h = api.torch.ones_like(envelope)*1e-10
        return api.torch.stack((envelope*.1, h, envelope, envelope, h*1e-8), 1), None


def test_full_range_can_escape_near_zero_anchor():
    model = api.HarmBranch(TinyBase(), 2, True)
    with api.torch.no_grad():
        model.risk_output.bias.fill_(12)
    result = model(api.torch.zeros(1, 2), api.torch.ones(1))
    assert result[0, 1] > .89 and result[0, 4] > .89


def test_zero_envelope_backward_finite():
    model = api.HarmBranch(TinyBase(), 2, True)
    y = model(api.torch.ones(2, 2), api.torch.zeros(2))
    y.sum().backward()
    assert api.torch.isfinite(y).all()
    assert all(p.grad is None or api.torch.isfinite(p.grad).all() for p in model.parameters())
