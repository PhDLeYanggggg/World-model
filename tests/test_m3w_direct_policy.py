import copy
import numpy as np
import pytest
import torch
from src.world_model import m3w_direct_policy as api
from tests.test_m3w_shared_train_risk import packet


def fixture():
    packets = {s: packet(s) for s in ('a', 'b')}
    pred = {s: np.tile([.2, .005, 1., 1., .005], (len(p[0]), 1)) for s, p in packets.items()}
    return packets, pred


def settings():
    return dict(width=8, steps=6, rows_per_record=4, learning_rate=.001,
                gradient_clip=5., dual_learning_rate=1., dual_maximum=100.,
                heartbeat_every=3, checkpoint_every=3)


def test_hard_forward_is_actual_decision_and_has_surrogate_gradient():
    x = torch.tensor([-1., 0., 1.], requires_grad=True)
    out = api.decision(x, True, 1.)
    torch.testing.assert_close(out, torch.tensor([0., 0., 1.]))
    out.sum().backward()
    torch.testing.assert_close(x.grad, torch.sigmoid(x)*(1-torch.sigmoid(x)))
    assert not torch.equal(api.decision(x, False, 1.), out)


def test_unknown_labels_are_worst_case_loss_not_safe_zero():
    y = np.array([[.1, 0., 1., 1., 0.], [np.nan]*5])
    gain, risk, ref, known = api.targets(y, np.array([1., 2.]))
    np.testing.assert_allclose(gain, [.1, -2.])
    np.testing.assert_allclose(risk, [[-.02, -.02], [2., 2.]])
    np.testing.assert_array_equal(ref[1], [0., 0.])
    assert known.tolist() == [True, False]
    y[1, 0] = 0.
    with pytest.raises(ValueError): api.targets(y, np.ones(2))


def test_record_constraint_detects_failure_hidden_by_source_average():
    d = dict(sources=['a'], source_index=torch.tensor([0, 0]),
             within_source=torch.tensor([.99, .01]), source_scales=torch.ones(1, 2),
             record_scales=torch.ones(2, 2))
    excess = torch.tensor([[-.02, -.02], [.08, .08]])
    assert (api.reduce_risk(excess, d, 'soft_source') < 0).all()
    assert (api.reduce_risk(excess, d, 'soft_record')[1] > 0).all()


def test_inputs_and_initial_decisions_do_not_use_future_labels():
    packets, pred = fixture(); args = packets['a']; eligible = api.shared.eligible(args)
    z, anchor = api.inputs(args[0], args[1], pred['a'], args[-1], eligible)
    model = api.Policy(z.shape[1], 8, 17)
    np.testing.assert_array_equal(model(torch.from_numpy(z), torch.from_numpy(anchor)).detach().numpy() > 0, anchor)
    changed = copy.deepcopy(args); changed[2][:] = np.nan
    zz, aa = api.inputs(changed[0], changed[1], pred['a'], changed[-1], api.shared.eligible(changed))
    np.testing.assert_array_equal(z, zz); np.testing.assert_array_equal(anchor, aa)


@pytest.mark.parametrize('arm', api.ARMS)
def test_exact_resume_and_real_parameter_updates(tmp_path, arm):
    packets, pred = fixture(); data = api.prepare(packets, pred)
    kw = dict(arm=arm, settings=settings(), seed=17, identity={'test': True})
    a = api.fit(data, path=tmp_path/'full.pt.gz', **kw)
    api.fit(data, path=tmp_path/'resume.pt.gz', stop_at=3, **kw)
    b = api.fit(data, path=tmp_path/'resume.pt.gz', resume=True, **kw)
    for k in a:
        if k != 'seconds': api.core.exact(a[k], b[k])
    assert any(not torch.equal(a['model'][k], v) for k, v in a['initial_model'].items())
    args = packets['a']; eligible = api.shared.eligible(args)
    score, action = api.predict(a, 'a', args[0], args[1], pred['a'], eligible)
    assert np.isfinite(score).all() and not (action & ~eligible).any()


def test_all_objectives_have_exactly_matched_data_and_initialization(tmp_path):
    data = api.prepare(*fixture())
    states = [api.fit(data, arm=a, settings=settings(), seed=17, identity={'test': True},
                      path=tmp_path/(a+'.pt.gz')) for a in api.ARMS]
    api.assert_matched(states)
