import numpy as np
import pytest
import torch

from src.world_model.m3w_objective_alignment import fit_objective
from src.world_model.m3w_offline_visual_forecast import OfflineVisualForecast
from src.world_model.m3w_track_event_sampling import (
    fit_sampled, sampling_probabilities, train_distribution, training_event_labels,
)


def groups():
    return np.array([0, 0, 0, 0, 1, 1]), np.array(['a', 'a', 'a', 'b', 'c', 'd']), np.array([0, 0, 1, 1, 4, 4])


def test_scene_and_track_masses_have_declared_hierarchy():
    scene, tracks, events = groups()
    row = sampling_probabilities(scene, tracks, events, 'row_uniform')
    np.testing.assert_array_equal(row, np.ones(6)/6)
    p = sampling_probabilities(scene, tracks, events, 'scene_uniform')
    np.testing.assert_allclose(p[:4], .125)
    p = sampling_probabilities(scene, tracks, events, 'scene_track')
    assert np.isclose(p[:3].sum(), .25) and p[3] == .25
    assert p[4] == p[5] == .25


def test_event_then_track_balance_and_missing_categories():
    scene, tracks, events = groups()
    p = sampling_probabilities(scene, tracks, events, 'scene_event_track')
    assert np.isclose(p[(scene == 0) & (events == 0)].sum(), .25)
    assert np.isclose(p[(scene == 0) & (events == 1)].sum(), .25)
    np.testing.assert_allclose(p, [.125, .125, .125, .125, .25, .25])
    assert np.isclose(p[events == 4].sum(), .5)


def test_held_labels_cannot_change_training_distribution():
    scene, tracks, _ = groups()
    geometry, targets = np.zeros((6, 20)), np.zeros((6, 12, 2))
    targets[2, :, 0] = 1
    ids = np.arange(4)
    p, labels = train_distribution(ids, scene, tracks, geometry, targets, 'scene_event_track')
    targets[4:] = np.nan
    geometry[4:] = np.nan
    again, again_labels = train_distribution(ids, scene, tracks, geometry, targets, 'scene_event_track')
    np.testing.assert_array_equal(p, again)
    np.testing.assert_array_equal(labels, again_labels)


def test_event_labels_are_nonoverlapping_supervised_categories():
    geometry, targets = np.zeros((5, 16)), np.zeros((5, 12, 2))
    targets[1, :, 0] = 1
    history = np.column_stack((np.arange(-7, 1), np.zeros(8))).reshape(16)
    geometry[2:] = history
    targets[3, :, 1] = np.arange(1, 13)
    targets[4, :, 0] = np.arange(1, 13)
    np.testing.assert_array_equal(training_event_labels(geometry, targets), np.arange(5))


def fixture(seed=17):
    torch.manual_seed(seed)
    x = torch.randn(20, 16)
    observed = torch.ones(20, 8)
    baseline = torch.zeros(20, 12, 2)
    target = torch.randn(20, 12, 2)*.1
    return lambda ids:(x[ids], observed[ids], baseline[ids], target[ids])


def test_row_control_matches_old_trainer_and_exact_resume(tmp_path):
    torch.set_num_threads(4)
    cfg = dict(updates=120, batch_size=8, learning_rate=.0003, weight_decay=.0001, checkpoint_every=40)
    batch = fixture()
    ids = torch.arange(20)
    folds = np.repeat([0, 1], 10)
    identity = {'synthetic_test':True}
    torch.manual_seed(17)
    old = OfflineVisualForecast(16)
    fit_objective(old, batch, ids, folds, arm='row_log', config=cfg, seed=17,
                  identity=identity, checkpoint=tmp_path/'old.pt', heartbeat=lambda v:None)
    torch.manual_seed(17)
    new = OfflineVisualForecast(16)
    p = np.ones(20)/20
    first = fit_sampled(new, batch, ids, p, mode='row_uniform', config=cfg, seed=17,
        identity=identity, checkpoint=tmp_path/'new.pt', heartbeat=lambda v:None, stop_at=40)
    assert first['step'] == 40 and first['draw_counts'].sum() == 320
    resumed = fit_sampled(new, batch, ids, p, mode='row_uniform', config=cfg, seed=17,
        identity=identity, checkpoint=tmp_path/'new.pt', heartbeat=lambda v:None)
    for key in old.state_dict():
        assert torch.equal(old.state_dict()[key], new.state_dict()[key])
    assert resumed['new_updates_this_invocation'] == 80 and resumed['draw_counts'].sum() == 960
    old_state = torch.load(tmp_path/'old.pt', weights_only=False)
    new_state = torch.load(tmp_path/'new.pt', weights_only=False)
    assert old_state['losses'] == new_state['losses']
    assert torch.equal(old_state['sampler_rng'], new_state['sampler_rng'])
    with pytest.raises(ValueError, match='Changed resume'):
        changed = p.copy(); changed[:2] += [.01, -.01]
        fit_sampled(new, batch, ids, changed, mode='row_uniform', config=cfg, seed=17,
            identity=identity, checkpoint=tmp_path/'new.pt', heartbeat=lambda v:None)


def test_invalid_track_scene_identity_and_indices_rejected():
    with pytest.raises(ValueError, match='one physical scene'):
        sampling_probabilities([0, 1], ['same', 'same'], [0, 1], 'scene_track')
    with pytest.raises(ValueError, match='Unique valid'):
        train_distribution([0, 0], np.zeros(2), ['a', 'b'], np.zeros((2, 16)),
                           np.zeros((2, 12, 2)), 'scene_track')
