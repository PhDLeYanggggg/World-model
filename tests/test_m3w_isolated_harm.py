import copy

import numpy as np
import pytest

from src.world_model import m3w_isolated_harm as api
from tests.test_m3w_decoder_risk_factorial import packets
from tests.test_m3w_shared_train_risk import settings, CONFIG, packet


def anchor(tmp_path):
    return api.previous.fit(packets(), arm='shared_utility', settings=settings(), objective_config=CONFIG,
        seed=17, identity={'anchor': True}, experiment_sha256='parent', path=tmp_path/'anchor.pt.gz', heartbeat=lambda **_: None)


def fit(path, a, arm, **kwargs):
    return api.fit(packets(), anchor=a, anchor_sha256='frozen', arm=arm, settings=settings(), objective_config=CONFIG,
        identity={'test': True}, experiment_sha256='new', path=path, heartbeat=lambda **_: None, **kwargs)


@pytest.mark.parametrize('arm', api.ARMS)
def test_exact_continuation_resume(tmp_path, arm):
    a = anchor(tmp_path)
    direct = fit(tmp_path/'direct.pt.gz', a, arm)
    fit(tmp_path/'resume.pt.gz', a, arm, stop_at=3)
    resumed = fit(tmp_path/'resume.pt.gz', a, arm, resume=True)
    for key in direct:
        if key != 'seconds':
            api.core.exact(direct[key], resumed[key])


def test_isolated_benefit_reference_and_causal_bounds(tmp_path):
    a = anchor(tmp_path)
    states = [fit(tmp_path/(arm+'.pt.gz'), a, arm) for arm in api.ARMS]
    api.assert_matched(states)
    x, env = packet()[:2]
    teacher = api.previous.predict(a, 'a', x, env)
    for s in states:
        p = api.predict(s, 'a', x, env)
        assert np.isfinite(p).all() and (p >= 0).all()
        assert (p[:, :2].sum(1) <= env+1e-6).all()
        assert (p[:, 4] <= p[:, 1]+1e-8).all() and (p[:, 3] <= p[:, 2]+1e-8).all()
    isolated = api.predict(states[-1], 'a', x, env)
    np.testing.assert_array_equal(isolated[:, [0, 2, 3]], teacher[:, [0, 2, 3]])
    assert states[-1]['trainable_parameters'] < states[1]['trainable_parameters']
    assert not np.array_equal(isolated[:, 1], teacher[:, 1])
    with pytest.raises(ValueError):
        api.predict(states[-1], 'unknown', x, env)


def test_initial_identity_has_nonzero_harm_gradient(tmp_path):
    a = anchor(tmp_path)
    model = api.initialize(a, 'isolated_risk')
    x = api.torch.randn(12, 380)
    env = api.torch.ones(12)
    out = model(x, env)
    teacher, _ = model.base(x, env)
    api.torch.testing.assert_close(out, teacher, rtol=1e-6, atol=1e-7)
    (out[:, 1]+out[:, 4]).sum().backward()
    assert model.risk_output.bias.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.base.parameters())


def test_changed_anchor_or_protected_weight_rejected(tmp_path):
    a = anchor(tmp_path)
    s = fit(tmp_path/'state.pt.gz', a, 'isolated_risk', stop_at=3)
    s['model']['base.primary.output.bias'][0] += 1
    with pytest.raises(AssertionError):
        api.check_protection(s)
    changed = copy.deepcopy(a)
    changed['model']['primary.output.bias'][0] += 1
    with pytest.raises(AssertionError):
        fit(tmp_path/'state.pt.gz', changed, 'isolated_risk', resume=True)
