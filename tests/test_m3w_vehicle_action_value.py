import numpy as np
import torch

from scripts import run_m3w_vehicle_action_value as runner
from src.evaluation.m3w_interaction_frontier import pair_reduction
from src.world_model import m3w_vehicle_action_value as model


def scenario():
    cv = np.zeros((3, 12, 2))
    a = cv.copy()
    a[:2, :, 0] = 1.
    x = np.ones((3, 29), dtype=np.float32)
    return cv, a, x, np.broadcast_to(np.eye(2), (3, 2, 2)), np.ones(3), np.zeros((3, 2))


def test_targets_are_exact_lower_contrasts_and_ignore_missing_values():
    cv, a, x, r, s, o = scenario()
    design, bank = model.design(cv, a, [0, 1], x, r, s, o)
    assert design.shape == (2, 12, model.WIDTH)
    y = np.ones_like(cv)
    valid = np.ones((3, 12), dtype=bool)
    valid[1, 5:] = False
    y[~valid] = np.nan
    terms = model.targets(a, [0, 1], bank, y, valid)
    candidate = np.broadcast_to(a, (12, *a.shape)).copy()
    candidate[:, 0] = bank[0]
    den = np.linalg.norm(bank[0]-a[0], axis=-1).sum(1)
    np.testing.assert_allclose(terms[0, :, 2]*den*2/36, pair_reduction(candidate, a, y, valid)[:, 1], atol=1e-7)
    den = np.linalg.norm(bank[1]-a[1], axis=-1).sum(1)
    delta = np.linalg.norm(a[1]-y[1], axis=-1)-np.linalg.norm(bank[1]-y[1], axis=-1)
    debit = np.linalg.norm(bank[1]-a[1], axis=-1)[:, ~valid[1]].sum(1)
    np.testing.assert_allclose((terms[1, :, 0]-terms[1, :, 1])*den,
                               delta[:, valid[1]].sum(1)-debit, atol=1e-6)
    assert np.isfinite(terms).all()


def test_matched_choice_does_not_change_actor_and_ego_ignores_context():
    net = model.AngleValue(False)
    a, b = torch.zeros((2, 12, model.WIDTH)), torch.zeros((2, 12, model.WIDTH))
    b[..., model.OWN:] = 4.
    torch.testing.assert_close(net(a), net(b), atol=0., rtol=0.)
    values = np.zeros((2, 12, 3))
    assert model.choose(values)[0] == (-1, -1)
    values[0, 0] = [1., 0., 0.]
    values[1, 3] = [0.8, 0., 0.8]
    assert model.choose(values)[0] == (0, 1)
    assert model.choose(values, joint=True)[0] == (1, 4)
    assert model.choose(values, same_actor=1)[0] == (1, 4)


def test_resume_matches_uninterrupted_action_fit(tmp_path, monkeypatch):
    own = torch.Generator().manual_seed(35)
    x = torch.randn((16, model.WIDTH), generator=own)
    y = torch.zeros((16, 3))
    identity = dict(name='resume_check', seed=17, excluded=runner.base.SITES[:2])
    monkeypatch.setattr(runner.old, 'SAVE', 3)
    monkeypatch.setattr(runner.old, 'LOCAL', tmp_path)

    def fit(steps):
        torch.manual_seed(17)
        return runner.old.fit(identity, [(x, y)], model.loss, model.AngleValue(True), steps, 8)[0]

    fit(3)
    resumed = fit(6)
    full = tmp_path/'full'
    full.mkdir()
    monkeypatch.setattr(runner.old, 'LOCAL', full)
    for a, b in zip(resumed.parameters(), fit(6).parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=0., rtol=0.)
