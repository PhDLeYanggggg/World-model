import copy

import numpy as np
import pytest

from src.world_model import m3w_query_context_risk as api
from tests.test_m3w_isolated_harm_fp64 import anchor
from tests.test_m3w_decoder_risk_factorial import packets
from tests.test_m3w_shared_train_risk import settings, CONFIG


def test_peer_context_is_current_query_only_and_permutation_equivariant():
    z = np.arange(24, dtype=float).reshape(6, 4)
    recs = np.array(['a', 'a', 'a', 'a', 'b', 'b'])
    frames = np.array([0, 0, 0, 1, 0, 0])
    peer, counts = api.peer_context(z, recs, frames)
    np.testing.assert_allclose(peer[0], z[[1, 2]].mean(0))
    np.testing.assert_array_equal(peer[3], z[3])
    np.testing.assert_array_equal(counts, [3, 3, 3, 1, 2, 2])
    order = np.array([5, 2, 4, 0, 3, 1])
    p, c = api.peer_context(z[order], recs[order], frames[order])
    np.testing.assert_allclose(p, peer[order])
    np.testing.assert_array_equal(c, counts[order])
    changed = z.copy(); changed[3:] += 10000
    np.testing.assert_array_equal(api.peer_context(changed, recs, frames)[0][:3], peer[:3])
    with pytest.raises(ValueError):
        api.peer_context(z, recs, frames[:-1])
    with pytest.raises(ValueError):
        api.peer_context(z * np.nan, recs, frames)


def fit(path, a, arm, **kwargs):
    return api.fit(packets(), anchor=a, anchor_sha256='frozen', arm=arm,
        settings=settings(), objective_config=CONFIG, identity={'test': True},
        experiment_sha256='test', path=path, heartbeat=lambda **_: None, **kwargs)


@pytest.mark.parametrize('arm', api.ARMS)
def test_exact_resume_and_protected_outputs(tmp_path, arm):
    a = anchor(tmp_path)
    direct = fit(tmp_path/'direct.pt.gz', a, arm)
    fit(tmp_path/'resume.pt.gz', a, arm, stop_at=3)
    resumed = fit(tmp_path/'resume.pt.gz', a, arm, resume=True)
    for k in direct:
        if k != 'seconds': api.core.exact(direct[k], resumed[k])
    args = packets()['a']
    p = api.predict(direct, 'a', args[0], args[1], args[5], args[6])
    teacher = api.parent.previous.predict(a, 'a', args[0], args[1])
    np.testing.assert_array_equal(p[:, [0, 2, 3]], teacher[:, [0, 2, 3]])
    assert np.isfinite(p).all() and (p >= 0).all()
    assert (p[:, :2].sum(1) <= args[1] + 1e-6).all()
    assert (p[:, 4] <= p[:, 1]).all()
    with pytest.raises(ValueError):
        api.predict(direct, 'unseen', args[0], args[1], args[5], args[6])


def test_matched_arms_and_unknown_peers_are_not_removed(tmp_path):
    a = anchor(tmp_path)
    states = [fit(tmp_path/(arm+'.pt.gz'), a, arm) for arm in api.ARMS]
    api.assert_matched(*states)
    x = np.array([[0., 0.], [2., 4.], [100., 200.]])
    # No label/known mask enters the API; the third row remains a peer.
    peer, _ = api.peer_context(x, np.repeat('r', 3), np.zeros(3))
    np.testing.assert_array_equal(peer[0], [51., 102.])
    changed = copy.deepcopy(states[0])
    changed['model']['base.primary.output.bias'][0] += 1
    with pytest.raises(AssertionError): api.check_protection(changed)


def test_zero_initialized_context_does_not_change_anchor(tmp_path):
    a = anchor(tmp_path)
    model = api.initialize(a, 'peer')
    z = api.torch.randn(7, 380)
    env = api.torch.ones(7)
    p = model(z, env, -z)
    b, _ = model.base(z, env)
    api.torch.testing.assert_close(p, b, atol=1e-7, rtol=1e-6)
    p[:, [1, 4]].sum().backward()
    assert model.risk_output.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.base.parameters())


def test_copied_risk_encoder_is_trainable_even_when_base_is_frozen(tmp_path):
    model = api.initialize(anchor(tmp_path), 'peer')
    assert all(p.requires_grad for p in model.risk_encoder.parameters())
    with api.torch.no_grad():
        model.risk_output.weight.fill_(.01)
    z = api.torch.randn(7, 380)
    model(z, api.torch.ones(7), -z)[:, [1, 4]].sum().backward()
    assert sum(p.grad.abs().sum() for p in model.risk_encoder.parameters()) > 0
    assert all(not p.requires_grad for p in model.base.parameters())
