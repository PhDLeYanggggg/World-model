import numpy as np
import pytest
from src.world_model import m3w_fixed_occurrence as api

torch = api.torch
torch.set_num_threads(4)


def fixture():
    old = api.old.initialize(5, 8, 17)
    state = dict(model=old.state_dict(), settings=dict(width=8), seed=17,
                 norm=dict(mean=np.zeros(5), std=np.ones(5), clip=5., cost_scale=1., training_sites=['a', 'b']))
    x = np.random.default_rng(2).normal(size=(8, 5)).astype(np.float32)
    env = np.ones(8, np.float32)
    y = np.array([[1, 1, .1], [0, 2, .2], [1, 1, 0], [1, 2, .1]]*2, np.float32)
    return state, x, env, y


def test_both_initial_predictions_equal_teacher():
    state, x, env, _ = fixture()
    teacher = api.old.initialize(5, 8, 17); teacher.load_state_dict(state['model'])
    for arm in api.ARMS:
        model = api.initialize(state, arm)
        torch.testing.assert_close(model(torch.from_numpy(x), torch.from_numpy(env)),
                                   teacher(torch.from_numpy(x), torch.from_numpy(env)), rtol=0, atol=0)


def test_only_occurrence_branch_trainability_changes():
    state, _, _, _ = fixture(); a = api.initialize(state, 'trainable'); b = api.initialize(state, 'fixed')
    api.old.sampling.exact(a.state_dict(), b.state_dict())
    assert all(p.requires_grad for p in a.parameters())
    assert not any(p.requires_grad for p in b.occurrence.parameters())
    assert all(p.requires_grad for p in b.cost.parameters())


def run(tmp_path, arm, *, resume=False, stop_at=None):
    state, x, env, y = fixture()
    return api.fit(x, env, y, np.array(['a']*4+['b']*4), np.array(['r']*8), np.array([0, 0, 1, 1]*2),
        state, arm=arm, settings=dict(steps=6, learning_rate=.001, query_batch_size=4,
            gradient_clip=5., checkpoint_every=2, heartbeat_every=2), identity=dict(test='synthetic'),
        path=tmp_path/'checkpoint.pt.gz', heartbeat=lambda **_: None, resume=resume, stop_at=stop_at)


def test_frozen_probability_is_exact_and_costs_learn(tmp_path):
    s = run(tmp_path, 'fixed')
    for k in s['initial_model']:
        if k.startswith('occurrence.'): torch.testing.assert_close(s['model'][k], s['initial_model'][k], rtol=0, atol=0)
    assert any(not torch.equal(s['model'][k], s['initial_model'][k]) for k in s['model'] if k.startswith('cost.'))
    assert s['unknown_rows_sampled'] == 0 and s['occurrence_frozen_exact']


def test_matched_queries_and_initializations(tmp_path):
    a, b = run(tmp_path/'a', 'trainable'), run(tmp_path/'b', 'fixed')
    api.assert_matched(a, b)
    assert any(not torch.equal(a['model'][k], a['initial_model'][k]) for k in a['model'] if k.startswith('occurrence.'))


@pytest.mark.parametrize('arm', api.ARMS)
def test_checkpoint_resume_matches_full_training(tmp_path, arm):
    a = run(tmp_path/'full', arm)
    run(tmp_path/'resume', arm, stop_at=2)
    b = run(tmp_path/'resume', arm, resume=True)
    for k in a:
        if k != 'seconds': api.old.sampling.exact(a[k], b[k])


def test_rejects_unknown_arm():
    with pytest.raises(ValueError): api.initialize(fixture()[0], 'selected_from_test')
