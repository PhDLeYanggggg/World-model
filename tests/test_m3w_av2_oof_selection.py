import numpy as np

from scripts import run_m3w_av2_oof_selection as run
from src.world_model import m3w_av2_interaction_credit as credit


def test_inner_producers_exclude_outer_held_calibration_and_recipient():
    cities = ["a", "b", "c", "d", "e", "f"]
    roles, jobs = run.oof.layout(cities)
    assert len(jobs) == 18
    for held, role in roles.items():
        for recipient, model in role["inner"].items():
            train, predicted = jobs[model]
            assert recipient in predicted
            assert set(train) == set(role["donors"])-{recipient}
            assert set(train).isdisjoint({held, recipient, role["calibration"]})


def test_only_edited_pair_contributes_to_nonadditive_gain():
    rng = np.random.default_rng(314)
    c, y, p = rng.normal(size=(3, 3, 4, 2))
    p = c.copy()+p
    p[2] = c[2]
    i, j = np.triu_indices(3, 1)
    metric = lambda z: np.linalg.norm(z[i]-z[j]-(y[i]-y[j]), axis=-1)
    a, b = c.copy(), c.copy()
    a[0], b[1] = p[0], p[1]
    expected = (metric(a)+metric(b)-metric(p)-metric(c)).sum(0).mean()
    observed = credit.interaction_lower(c, p[None], np.array([[0, 1]]), y, np.ones((3, 4), bool))
    np.testing.assert_allclose(observed, [expected], atol=1e-12)


def test_missing_pair_lower_bound_and_recipient_swap():
    rng = np.random.default_rng(41)
    c, y, d = rng.normal(size=(3, 3, 12, 2))
    p = c.copy()+d
    p[2] = c[2]
    v = np.ones((3, 12), bool)
    v[0, ::2] = False
    a = np.array([[0, 1]])
    lower = credit.interaction_lower(c, p[None], a, y, v)
    np.testing.assert_allclose(lower, credit.interaction_lower(c, p[None], a[:, ::-1], y, v))
    for _ in range(40):
        full_y = np.where(v[..., None], y, rng.normal(size=y.shape))
        g = credit.interaction_lower(c, p[None], a, full_y, np.ones_like(v))
        assert (g >= lower-1e-12).all()
    h = rng.normal(size=(3, 8, 2))
    x, g, b, _ = credit.describe(h, c, c, p[None], a, .5)
    other = credit.describe(h, c, c, p[None], a[:, ::-1], .5)
    assert x.shape == (1, credit.WIDTH)
    for left, right in zip((x, g, b), other[:3], strict=True):
        np.testing.assert_allclose(left, right, atol=1e-6)


def test_risk_and_interaction_share_gain_and_qualification():
    torch = run.torch
    z = torch.zeros((2, 4))
    g, b = torch.ones(2), torch.ones(2)
    original = run.select_scores(z, g, b, 3)
    torch.testing.assert_close(original[:, 0], original[:, 1])
    z[0, 1] = 2.
    revised = run.select_scores(z, g, b, 3)
    torch.testing.assert_close(revised[:, 0], original[:, 0])
    assert revised[0, 1] > original[0, 1]
