"""Source-only OOF cost assembly and a sampler-matched tree comparator."""
from __future__ import annotations

import os
from pathlib import Path
import time

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor

from src.evaluation.m3w_forecast_cost_bounds import bounded_fractions
from src.world_model.m3w_native_gain_harm import standardized


def assemble_oof(sites, parts, *, seed, family):
    sites = np.asarray(sites)
    roster = set(sites.tolist())
    if sites.ndim != 1 or len(roster) < 2 or len(parts) != len(roster):
        raise ValueError("One site-excluded predictor per complete source-site partition required")
    prediction = np.empty((len(sites), 12, 2), dtype=np.float32)
    seen = set()
    for group in parts:
        site, ids = group["row_site"], np.asarray(group["ids"])
        p = np.asarray(group["prediction"])
        if (site in seen or site not in roster or group["seed"] != seed or group["family"] != family
                or group["training_sites"] != sorted(roster-{site})
                or ids.dtype.kind not in "iu" or not np.array_equal(ids, np.flatnonzero(sites == site))
                or p.shape != (len(ids), 12, 2) or not np.isfinite(p).all()):
            raise ValueError("Exposed, missing, wrong-family or misaligned OOF producer")
        prediction[ids] = p
        seen.add(site)
    if seen != roster:
        raise ValueError("Incomplete OOF source coverage")
    return prediction


def fraction_targets(labels, distance, known, draws):
    y, d, k, count = map(np.asarray, (labels, distance, known, draws))
    if (k.dtype != bool or k.shape != d.shape or count.shape != d.shape
            or count.dtype.kind not in "iu" or (count < 0).any() or count[~k].any()):
        raise ValueError("Aligned complete-label support and neural sampling counts required")
    q = bounded_fractions(y, d, k)
    q[~k] = 0.
    # Match bounded_fraction minibatch squared error. Distance-zero rows have
    # exactly zero loss and gradient; omitting their tree mass changes no optimum.
    weight = count.astype(float) * (d > 0)
    if not (weight > 0).any():
        raise ValueError("No positive fitting mass")
    q /= np.maximum(1., q.sum(1))[:, None]
    return q, weight


def predict_forest(model, x, distance, pr):
    d = np.asarray(distance, float)
    if d.shape != (len(x),) or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError("Finite causal prediction disagreement required")
    z = standardized(x, pr)
    if not np.isfinite(z).all():
        raise ValueError("Nonfinite causal features")
    model.set_params(n_jobs=1)
    q = model.predict(z)
    if (q.shape != (len(x), 2) or not np.isfinite(q).all()
            or (q < 0).any() or (q.sum(1) > 1+1e-12).any()):
        raise ValueError("Predicted fraction outside nonnegative cost simplex")
    return q*d[:, None]


def fit_forest(x, y, distance, pr, draws, *, settings, seed, identity, directory,
               heartbeat, resume=False, stop_at=None):
    q, weight = fraction_targets(y, distance, pr["known"], draws)
    z, use = standardized(x, pr), weight > 0
    if not np.isfinite(z).all():
        raise ValueError("Nonfinite training feature")
    path = Path(directory)/"checkpoint.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    model = ExtraTreesRegressor(n_estimators=1, criterion="squared_error", bootstrap=False,
        max_depth=settings["max_depth"], min_samples_leaf=settings["min_samples_leaf"],
        max_features=settings["max_features"], random_state=seed, warm_start=True,
        n_jobs=settings["fit_threads"])
    seconds, trace = 0., []
    if path.exists():
        if not resume:
            raise ValueError("Existing forest requires explicit resume")
        cp = joblib.load(path)
        if (cp["identity"] != identity or cp["settings"] != settings or cp["seed"] != seed
                or cp["sklearn"] != sklearn.__version__):
            raise ValueError("Changed forest checkpoint identity")
        for key in ("mean", "std", "known"):
            np.testing.assert_array_equal(cp["preprocess"][key], pr[key])
        if not np.array_equal(cp["draws"], draws) or not np.array_equal(cp["sample_weight"], weight):
            raise ValueError("Changed matched fitting budget")
        model, seconds, trace = cp["model"], cp["seconds"], cp["trace"]
    trees = len(getattr(model, "estimators_", []))
    limit = settings["trees"] if stop_at is None else stop_at
    if not 0 < limit <= settings["trees"] or limit < trees:
        raise ValueError("Nondecreasing fixed forest budget required")
    started = time.monotonic()
    try:
        while trees < limit:
            trees = min(limit, trees+settings["checkpoint_every"])
            model.set_params(n_estimators=trees, n_jobs=settings["fit_threads"])
            with joblib.parallel_backend("threading"):
                model.fit(z[use], q[use], sample_weight=weight[use])
            model.set_params(n_jobs=1)
            score = model.predict(z[use])
            row = dict(trees=trees, fitting_fraction_MSE=float(np.average(
                ((score-q[use])**2).mean(1), weights=weight[use])),
                nodes=sum(t.tree_.node_count for t in model.estimators_))
            trace.append(row)
            cp = dict(identity=identity, settings=settings, seed=seed, sklearn=sklearn.__version__,
                objective="neural_draw_count_weighted_fraction_MSE", model=model,
                preprocess=pr, draws=draws, sample_weight=weight,
                seconds=seconds+time.monotonic()-started, trace=trace)
            temporary = path.with_suffix(".tmp")
            joblib.dump(cp, temporary, compress=0)
            os.replace(temporary, path)
            heartbeat(state="training", **row)
    except BaseException:
        heartbeat(state="interrupted_resume_last_atomic_checkpoint", trees=trees)
        raise
    return model, dict(complete=trees == settings["trees"], trees=trees,
        seconds=seconds+time.monotonic()-started, sampled_rows=int(np.asarray(draws).sum()),
        effective_fit_rows=int(use.sum()), complete_rows=int(pr["known"].sum()), trace=trace,
        nodes=sum(t.tree_.node_count for t in model.estimators_))
