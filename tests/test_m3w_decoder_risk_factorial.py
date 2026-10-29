import numpy as np
import pytest

from src.world_model import m3w_decoder_risk_factorial as api
from tests.test_m3w_shared_train_risk import packet, settings, CONFIG


def packets():
    return {name: api.prior.prepare(packet(name), api.prior.recording_partition(name, packet(name)[5]))[0]
            for name in 'abcd'}


def fit(path, arm, **kwargs):
    return api.fit(packets(), arm=arm, settings=settings(), objective_config=CONFIG, seed=17,
        identity={'test': True}, experiment_sha256='test', path=path, heartbeat=lambda **_: None, **kwargs)


def test_exact_original_source_risk_control(tmp_path):
    previous = api.prior.fit_shared(packets(), mode='shared_source', settings=settings(), objective_config=CONFIG,
        seed=17, identity={'test': True}, experiment_sha256='test', path=tmp_path/'old.pt.gz', heartbeat=lambda **_: None)
    api.assert_previous_control(fit(tmp_path/'new.pt.gz', 'shared_risk'), previous)


@pytest.mark.parametrize('arm', api.ARMS)
def test_exact_factorial_resume(tmp_path, arm):
    direct = fit(tmp_path/'direct.pt.gz', arm)
    fit(tmp_path/'resume.pt.gz', arm, stop_at=3)
    resumed = fit(tmp_path/'resume.pt.gz', arm, resume=True)
    for key in direct:
        if key != 'seconds':
            api.old.core.exact(direct[key], resumed[key])
    assert (direct['dual'].sum() > 0) == arm.endswith('_risk')


def test_matched_initialization_exposure_and_causal_inference(tmp_path):
    states = [fit(tmp_path/(arm+'.pt.gz'), arm) for arm in api.ARMS]
    api.assert_matched(states)
    args = packet()
    for state in states:
        y = api.predict(state, 'a', args[0], args[1])
        assert y.shape == (32, 5) and np.isfinite(y).all() and (y >= 0).all()
        assert (y[:, :2].sum(1) <= args[1]+1e-6).all()
        assert (y[:, 3] <= y[:, 2]+1e-8).all() and (y[:, 4] <= y[:, 1]+1e-8).all()
        with pytest.raises(ValueError):
            api.predict(state, 'unseen', args[0], args[1])


def test_source_decoders_start_with_exact_same_predictions_and_explicit_extra_parameters():
    pr = api.prior.shared_preprocess(packets())
    a, b = [api.initialize(pr, settings(), 17, arm, list('abcd')) for arm in ('shared_risk', 'source_risk')]
    x = api.torch.randn(8, 380)
    env = api.torch.ones(8)
    for source in 'abcd':
        api.old.core.exact(a(x, env)[0], b(x, env, source)[0])
    extra = sum(p.numel() for p in b.parameters())-sum(p.numel() for p in a.parameters())
    assert extra == 3*5*(settings()['width']+1)


def test_changed_arm_cannot_resume(tmp_path):
    fit(tmp_path/'state.pt.gz', 'shared_risk', stop_at=3)
    with pytest.raises(AssertionError):
        fit(tmp_path/'state.pt.gz', 'shared_utility', resume=True)
