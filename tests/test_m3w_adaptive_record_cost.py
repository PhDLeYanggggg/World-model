import numpy as np
import pytest
import torch
from src.world_model import m3w_adaptive_record_cost as api


def packet():
    c = dict(x=np.zeros((2, 7, 428), np.float32), envelope=np.ones((2, 7)),
        scales=np.array([1., 2.], np.float32), allowed=np.ones((2, 7), bool), groups=[np.arange(2)],
        source_groups=[np.array([0])], sites=np.array(['s', 's']), ids=np.arange(2))
    return dict(causal=c, z=torch.zeros(2, 7, 428), observed_future=np.ones(2, bool),
        defined_by_arm=dict(bound_aware=np.ones(2, bool)), teacher_actions=np.array([6, 6]),
        record_codes=np.array([0, 0]), record_source_codes=np.array([0]),
        preprocess=dict(rms=np.ones(5, np.float32), bias=np.zeros(5, np.float32)),
        truth_by_arm=dict(bound_aware=torch.zeros(2, 7, 5)))


def prediction():
    p = torch.zeros(2, 7, 5)
    p[..., 0] = 1
    p[..., 1] = .01
    p[..., 2] = 2
    p[..., 3] = 1
    p[..., 4] = .005
    return p


def test_frozen_loss_and_gradients_match_original_even_for_duplicate_draws():
    d = packet()
    ix = np.array([0, 1, 0, 1])
    p = prediction()[ix].clone().requires_grad_()
    q = p.detach().clone().requires_grad_()
    a = api.objective(p, d, ix, 'frozen_teacher')[0]['record_excess']
    b = api.frozen.record_excess(q, d, ix)['record_excess']
    assert torch.equal(a, b)
    a.backward()
    b.backward()
    assert torch.equal(p.grad, q.grad)


def test_adaptive_actions_do_not_use_future_targets_or_teacher_actions():
    d = packet()
    _, first = api.objective(prediction(), d, np.arange(2), 'adaptive_selection')
    d['truth_by_arm']['bound_aware'][:] = 1000
    d['teacher_actions'][:] = -1
    _, second = api.objective(prediction(), d, np.arange(2), 'adaptive_selection')
    np.testing.assert_array_equal(first, second)


def test_causal_eligibility_controls_adaptive_selection():
    d = packet()
    d['causal']['allowed'][:] = False
    result, actions = api.objective(prediction(), d, np.arange(2), 'adaptive_selection')
    assert np.all(actions == -1) and result['selected_training_rows'] == 0
    assert result['record_excess'] == 0


def test_actual_small_adaptive_resume_matches_direct(tmp_path):
    settings = dict(width=4, learning_rate=.001, query_batch_size=1, minimum_updates=2,
        largest_source_cycles=1, gradient_clip=5., heartbeat_every=1, checkpoint_every=1)
    d = packet()
    for arm in api.ARMS:
        path = tmp_path / (arm + '.pt.gz')
        api.fit(d, loss_arm=arm, seed=17, settings=settings, identity=dict(test=True), path=path, stop_at=1)
        recovered = api.fit(d, loss_arm=arm, seed=17, settings=settings, identity=dict(test=True), path=path, resume=True)
        direct = api.fit(d, loss_arm=arm, seed=17, settings=settings, identity=dict(test=True), path=tmp_path/(arm+'_direct.pt.gz'))
        api.exact_scientific_state(recovered, direct)
        assert recovered['selection_draw_hash'] == direct['selection_draw_hash']


def test_no_unregistered_arm():
    with pytest.raises(ValueError):
        api.objective(prediction(), packet(), np.arange(2), 'oracle')
