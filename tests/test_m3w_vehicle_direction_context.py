import numpy as np
import pytest
import torch

from scripts import run_m3w_vehicle_direction_context as runner
from src.world_model.m3w_vehicle_direction_context import ablate, design, matched


def test_nearest_peers_use_shared_geometry_not_centered_endpoints():
    history = np.zeros((4, 8, 2))
    history[..., 0] = np.arange(-7., 1.)
    offsets = np.array([[0., 0.], [9., 0.], [2., 0.], [100., 0.]])
    x, cv, _, _ = design(history, offsets, np.array([3, 1]))
    np.testing.assert_allclose(x[0, [17, 23]], [2./8, 9./8])
    assert x[0, 22] == 1. and x[3, 17:].sum() == 0.
    translated, cv2, _, _ = design(history, offsets+[500., -33.], np.array([3, 1]))
    np.testing.assert_array_equal(translated, x)
    np.testing.assert_array_equal(cv2, cv)
    np.testing.assert_array_equal(ablate(x)[:, :17], x[:, :17])
    assert not ablate(x)[:, 17:].any()


def test_matched_angle_cannot_change_actor_or_edit_count():
    x = np.array([[2., 1.], [0., -1.]])
    assert matched(x, None) is None
    assert matched(x, (1, 2)) == (1, 1)


def test_training_never_materializes_excluded_targets(tmp_path):
    x = np.zeros((2, 29), dtype=np.float32)
    cv = np.zeros((2, 12, 2), dtype=np.float32)
    source = tmp_path/'donor.npz'
    np.savez(source, future=cv, valid=np.ones((2, 12), dtype=bool))
    rows = dict(file=source, x=x, cv=cv, rotation=np.broadcast_to(np.eye(2), (2, 2, 2)),
                scale=np.ones(2))
    data = dict(train=rows, held=dict(file=tmp_path/'nonexistent.npz'))
    assert len(runner.training_data(data, ['train'], ['held'])) == 1
    with pytest.raises(ValueError, match='roles overlap'):
        runner.training_data(data, ['held'], ['held'])


def test_resume_restores_same_weights_and_sampling(tmp_path, monkeypatch):
    rng = np.random.default_rng(31)
    a = torch.from_numpy(rng.normal(size=(8, 29)).astype(np.float32))
    b = torch.zeros((8, 12, 2))
    y = b+0.2
    donor = [(a, b, y, torch.ones((8, 12), dtype=torch.bool))]
    identity = dict(seed=17, arm='neighbor', cell='example')
    monkeypatch.setattr(runner, 'BATCH', 8)
    monkeypatch.setattr(runner, 'STEPS', 50)
    monkeypatch.setattr(runner, 'LOCAL', tmp_path)
    runner.fit(identity, donor)
    monkeypatch.setattr(runner, 'STEPS', 100)
    resumed, _ = runner.fit(identity, donor)
    full = tmp_path/'full'
    full.mkdir()
    monkeypatch.setattr(runner, 'LOCAL', full)
    expected, _ = runner.fit(identity, donor)
    for x, y in zip(resumed.parameters(), expected.parameters(), strict=True):
        torch.testing.assert_close(x, y, atol=0., rtol=0.)
