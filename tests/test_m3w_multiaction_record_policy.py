import numpy as np
import torch

from src.world_model import m3w_multiaction_record_policy as api


def data_fixture(arm='physical_neural'):
    n = 12; features = np.random.default_rng(4).normal(size=(n, 7, 12)).astype(np.float32)
    allowed = np.ones((n, 7), bool); anchor = np.zeros_like(allowed); anchor[:, 0] = True
    y = np.tile([.1, 0., 1., 1., 0.], (n, 7, 1)); y[-1] = np.nan
    return api.prepare(features, anchor, allowed, y, np.ones((n, 7)), np.ones(n),
        np.array(['a']*6+['b']*6), np.array(['1']*n), arm=arm, identity={'fixture': 1})


def test_full_gradient_is_chunk_invariant_and_keeps_unknown_worst_case():
    torch.set_num_threads(1); data = data_fixture(); model = api.RecordActionPolicy(8, 17)
    dual = torch.ones((2, 3))
    first = api.full_pass(model, data, dual, .5, 4, backward=True)
    g1 = torch.cat([p.grad.flatten() for p in model.parameters()])
    model.zero_grad(set_to_none=True)
    other = api.full_pass(model, data, dual, .5, 12, backward=True)
    g2 = torch.cat([p.grad.flatten() for p in model.parameters()])
    torch.testing.assert_close(g1, g2, rtol=2e-5, atol=2e-7)
    assert abs(first['loss']-other['loss']) < 1e-6
    # The last unknown action has lower utility -1 and positive harm +1.
    assert data['gain'][-1, 0] == -1 and data['risk'][-1, 0, 0] == 1


def test_initial_warm_policy_matches_only_the_legal_parent_action():
    model = api.RecordActionPolicy(8, 17)
    features = torch.zeros((3, 7, 12)); anchor = torch.zeros((3, 7), dtype=torch.bool)
    anchor[0, 3] = True; anchor[1, 6] = True
    allowed = torch.ones_like(anchor); allowed[1, 6] = False
    p, selected = api.categorical(model(features, anchor, torch.arange(7)), allowed, 1.)
    assert selected.tolist() == [3, -1, -1]
    assert (p.sum(1) <= 1).all()


def test_exact_resume_and_real_support_constraint(tmp_path):
    torch.set_num_threads(1); data = data_fixture()
    settings = dict(width=8, steps=4, learning_rate=.001, gradient_clip=5.,
        dual_learning_rate=1., dual_maximum=100., chunk_size=4, heartbeat_every=2, checkpoint_every=2)
    kwargs = dict(data=data, arm='physical_neural', settings=settings, seed=17)
    direct = api.fit(**kwargs, path=tmp_path/'direct.pt.gz')
    api.fit(**kwargs, path=tmp_path/'resume.pt.gz', stop_at=1)
    resumed = api.fit(**kwargs, path=tmp_path/'resume.pt.gz', resume=True)
    for key in direct:
        if key != 'seconds':
            api.exact(direct[key], resumed[key])
    assert (data['support_min'] > 0).all()
    assert direct['full_gradient_rows'] == 4*12
