import numpy as np
import torch

from scripts.audit_m3w_opendd_exchange_angular_debit import components
from src.world_model.m3w_exchange_angular_head import AngularHead, features, losses, rotate
from src.world_model.m3w_ols_replacement import rollout


def test_angle_preserves_the_norm_and_starts_at_incumbent():
    cv = torch.randn(5, 12, 2, dtype=torch.float64)
    path = cv+torch.randn_like(cv)
    head = AngularHead(8)
    initial = head(torch.randn(5, 8)).double()
    torch.testing.assert_close(rotate(initial, cv, path), path)
    p = rotate(torch.randn(5, 2, dtype=torch.float64), cv, path)
    torch.testing.assert_close(torch.linalg.vector_norm(p-cv, dim=-1),
                               torch.linalg.vector_norm(path-cv, dim=-1))


def test_angular_features_ignore_global_rotation_and_translation():
    h = np.random.default_rng(9).normal(size=(5, 8, 2))
    cv, ols, *_ = rollout(h, 12)
    r = np.array([[.6, -.8], [.8, .6]])
    a = np.ones(len(h))
    x, s = features(h, cv, ols, a)
    x0, s0 = features(h@r+10., cv@r, ols@r, a)
    np.testing.assert_allclose(x, x0, atol=1e-7)
    np.testing.assert_allclose(s, s0)


def test_relative_loss_debits_missing_steps_without_looking_at_their_targets():
    cv = torch.zeros(1, 2, 2)
    ols = cv.clone()
    pred = cv.clone()
    pred[:, -1, 0] = 2.
    valid = torch.tensor([[True, False]])
    a, b = losses(pred, cv, ols, cv, valid, torch.ones(1))
    c, d = losses(pred, cv, ols, pred*100., valid, torch.ones(1))
    assert a == c == 0.
    assert b == d == 1.


def test_observed_and_missing_components_recover_direct_bound():
    y = np.zeros((1, 2, 2))
    p = y.copy()
    q = y.copy()
    p[0, 0, 0] = 2.
    q[0, 1, 0] = 3.
    assert components(p, q, y, np.array([[True, False]])) == dict(observed=2., debit=3., lower=-1.)
