import numpy as np
import pytest
import torch

from src.training.m3w_external_cost_bank import assemble_oof, fraction_targets, fit_forest, predict_forest
from src.world_model.m3w_bounded_cost_head import loss_value


def parts():
    sites = np.array(["a", "a", "b", "b", "c", "c"])
    result = []
    for site in sorted(set(sites)):
        ids = np.flatnonzero(sites == site)
        result.append(dict(row_site=site, training_sites=sorted(set(sites)-{site}),
            seed=17, family="transformer", ids=ids, prediction=np.full((len(ids), 12, 2), len(result), np.float32)))
    return sites, result


def test_oof_assembly_excludes_each_rows_site_and_preserves_alignment():
    sites, groups = parts()
    result = assemble_oof(sites, groups[::-1], seed=17, family="transformer")
    np.testing.assert_array_equal(result[:, 0, 0], [0, 0, 1, 1, 2, 2])


@pytest.mark.parametrize("bad", ["exposure", "duplicate", "missing", "row", "seed", "nan"])
def test_oof_rejects_wrong_or_exposed_producer(bad):
    sites, groups = parts()
    if bad == "exposure":
        groups[0]["training_sites"] = list(set(sites))
    elif bad == "duplicate":
        groups.append(groups[0])
    elif bad == "missing":
        groups.pop()
    elif bad == "row":
        groups[0]["ids"] = np.array([0, 2])
    elif bad == "seed":
        groups[0]["seed"] = 29
    else:
        groups[0]["prediction"][0, 0] = np.nan
    with pytest.raises(ValueError):
        assemble_oof(sites, groups, seed=17, family="transformer")


def test_matched_forest_empirical_objective_equals_logged_neural_fraction_objective():
    y = np.array([[1., 0.], [0., 3.], [0., 0.], [np.nan, np.nan]])
    d = np.array([2., 4., 0., 5.])
    known = np.array([True, True, True, False])
    draws = np.array([2, 3, 4, 0])
    q, w = fraction_targets(y, d, known, draws)
    score = np.array([[.5, .2], [1., 1.], [0., 0.], [2., 1.]])
    ids = np.repeat(np.arange(len(y)), draws)
    neural = loss_value(torch.tensor(score[ids]), torch.tensor(y[ids]), torch.tensor(d[ids]), "bounded_fraction")
    active = w > 0
    forest = np.average(((score[active]/d[active, None]-q[active])**2).mean(1), weights=w[active])
    # Distance-zero rows have identically zero neural loss and no fitting gradient.
    assert forest * draws[active].sum()/draws.sum() == pytest.approx(neural.item())
    np.testing.assert_array_equal(w, [2, 3, 0, 0])


@pytest.mark.parametrize("bad", ["unknown_draw", "bound", "negative", "zero"])
def test_fraction_targets_fail_closed(bad):
    y, d = np.array([[1., 0.], [np.nan, np.nan]]), np.array([2., 1.])
    known, draws = np.array([True, False]), np.array([2, 0])
    if bad == "unknown_draw":
        draws[1] = 1
    elif bad == "bound":
        y[0, 0] = 3
    elif bad == "negative":
        draws[0] = -1
    else:
        d[0] = 0
    with pytest.raises(ValueError):
        fraction_targets(y, d, known, draws)


def forest_fixture():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(64, 5)).astype(np.float32)
    d = np.linspace(0, 3, len(x))
    y = np.c_[d*(.25+.15*np.tanh(x[:, 0])), d*(.2+.1*np.tanh(x[:, 1]))]
    known = np.ones(len(x), bool)
    known[-1], y[-1] = False, np.nan
    pr = dict(mean=np.zeros(5), std=np.ones(5), known=known)
    draws = known.astype(np.int64)*2
    settings = dict(trees=4, max_depth=3, min_samples_leaf=2,
        max_features=1., checkpoint_every=2, fit_threads=1)
    return x, y, d, pr, draws, settings


def test_forest_resume_reproduces_endpoint_and_bounded_outputs(tmp_path):
    x, y, d, pr, draws, settings = forest_fixture()
    args = dict(settings=settings, seed=17, identity={"fixture": True}, heartbeat=lambda **k: None)
    one, r = fit_forest(x, y, d, pr, draws, directory=tmp_path/"one", **args)
    assert r["complete"]
    _, pilot = fit_forest(x, y, d, pr, draws, directory=tmp_path/"resume", stop_at=2, **args)
    assert not pilot["complete"]
    two, final = fit_forest(x, y, d, pr, draws, directory=tmp_path/"resume", resume=True, **args)
    p = predict_forest(two, x, d, pr)
    np.testing.assert_array_equal(p, predict_forest(one, x, d, pr))
    assert np.isfinite(p).all() and (p >= 0).all() and (p.sum(1) <= d+1e-12).all()
    assert not p[0].any() and final["sampled_rows"] == draws.sum()
    assert final["nodes"] > settings["trees"]
    changed = draws.copy(); changed[1] += 1
    with pytest.raises(ValueError):
        fit_forest(x, y, d, pr, changed, directory=tmp_path/"resume", resume=True, **args)
