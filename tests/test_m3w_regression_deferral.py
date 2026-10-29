import numpy as np
import torch

from src.world_model import m3w_regression_deferral as api


def test_logistic_deferral_is_continuous_cost_weighted_not_best_class_CE():
    logits = torch.tensor([[0., 0.]], requires_grad=True); costs = torch.tensor([[5., 1.]])
    loss = api.deferral_loss(logits, costs)
    torch.testing.assert_close(loss, torch.tensor(6.))
    loss.backward()
    assert logits.grad[0, 0] > 0 and logits.grad[0, 1] < 0
    zero = api.deferral_loss(torch.zeros((1, 3)), torch.zeros((1, 3)))
    assert zero == 0


def test_floor_alias_is_merged_not_counted_as_real_easy_support():
    s = np.zeros((3, 8)); s[:, 2] = 3; s[1:, 4] = 4
    ok = np.ones((3, 7), bool); ok[:, 1] = False; ok[2] = False
    chosen, _ = api.choices(s, ok, arm='physical_neural')
    assert chosen.tolist() == [-1, 3, -1]
    safe = np.ones_like(ok); safe[:, 3] = False
    chosen, _ = api.choices(s, ok, arm='physical_neural', risk_allowed=safe)
    assert chosen.tolist() == [-1, -1, -1]


def test_exact_resume_and_inactive_expert_separation(tmp_path):
    torch.set_num_threads(1); rng = np.random.default_rng(5)
    z = torch.tensor(rng.normal(size=(20, 8, 4)), dtype=torch.float32)
    cost = torch.tensor(rng.uniform(.1, 2, (20, 8)), dtype=torch.float32)
    data = dict(z=z, cost=cost, preprocess={'same': True}, input_hashes={'fixture': 'same'},
                groups=[np.arange(10), np.arange(10, 20)])
    settings = dict(width=8, steps=4, batch_size=8, learning_rate=.001,
                    gradient_clip=5., heartbeat_every=2, checkpoint_every=2)
    kw = dict(data=data, arm='physical_only', settings=settings, seed=17, identity={})
    direct = api.fit(**kw, path=tmp_path/'direct.pt.gz')
    api.fit(**kw, path=tmp_path/'resumed.pt.gz', stop_at=1)
    resumed = api.fit(**kw, path=tmp_path/'resumed.pt.gz', resume=True)
    for key in direct:
        if key != 'seconds':
            api.exact(direct[key], resumed[key])
    changed = {**data, 'z':z.clone(), 'cost':cost.clone()}
    changed['z'][:, 7] += 1000; changed['cost'][:, 7] += 1000
    inactive = api.fit(**{**kw, 'data':changed}, path=tmp_path/'inactive.pt.gz')
    for key in ('model', 'optimizer', 'draw_hash', 'sampler_rng'):
        api.exact(direct[key], inactive[key])
