import numpy as np
import torch

from scripts import run_m3w_av2_margin_assignment as run
from src.world_model import m3w_av2_margin_assignment as model


def test_action_components_recompose_original_full_grid_constraints():
    rng = np.random.default_rng(29)
    cv, ols, y = (rng.normal(size=(3, 12, 2)) for _ in range(3))
    valid = np.array([[True]*12, [False]*12, [True]*4+[False]*8])
    y[~valid] = np.nan
    bank = cv[:, None]+rng.normal(size=(3, 2, 12, 2))*.05
    row, col = np.repeat(np.arange(3), 2), np.tile(np.arange(2), 3)
    opts = np.array([[0, 2], [1, 5], [2, 4], [3, 5]])
    units = np.array([.5, 2., 5.])
    a, b = model.components(cv, ols, bank, y, valid, 2., row, col, units)
    actual = model.margins(*(torch.from_numpy(x) for x in (a, b, opts, row, units))).numpy()
    paths = np.broadcast_to(cv, (len(opts), *cv.shape)).copy()
    paths[np.arange(len(opts))[:, None], row[opts]] = bank[row[opts], col[opts]]
    target = np.where(valid[..., None], y, 0.)
    e = np.where(valid, np.linalg.norm(cv-target, axis=-1), 0.)
    per_actor = e.sum(1)/valid.sum(1).clip(1)
    cap = .02*per_actor[(per_actor > 0.) & (per_actor <= 2.)].sum()
    t = run.prev.pair.audit.terms(paths, cv, ols, cv, target, valid, 2., 1., e.sum(), 1)
    for i, p in enumerate(paths):
        risk = run.prev.metrics(p, dict(cv=cv, ols4=ols), target, valid, (2., 5.))["risk"]
        peak = np.linalg.norm(p-cv, axis=-1).max(1)
        mass = per_actor[peak > 0.].sum()
        assert mass > 0.
        t[i, 4] = (risk["completion_all_risk_upper"]-.02)*mass
    expected = np.column_stack((t[:, 1:3]*e.sum()/(100.*12), -t[:, 4:6], cap-t[:, 6]))/units.mean()
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_only_edited_easy_actor_supplies_support():
    opts = torch.tensor([[0, 1], [0, 2]])
    row = torch.arange(3)
    scene = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., .01, 1.]])
    m = torch.ones((2, 5))
    assert model.qualification(m, scene, opts, row).tolist() == [False, True]


def test_supervised_slate_score_rewards_support_and_all_positive_margins():
    opts, row = torch.tensor([[0, 1], [0, 2], [0, 1]]), torch.arange(3)
    z = torch.tensor([[.3]*5, [.3]*5, [.3, .3, -.5, .3, .3]], requires_grad=True)
    s = model.slate_score(z, torch.tensor([-4., 3., -4.]), opts, row)
    assert s[0] > s[1] and s[0] > s[2]
    loss = model.supervised_loss(s, torch.tensor([True, False, False]))
    loss.backward()
    assert torch.isfinite(z.grad).all()
