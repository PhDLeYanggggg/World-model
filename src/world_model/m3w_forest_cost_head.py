"""Standard cost-regression comparator; not a calibrated probability or novel head."""
import os
from pathlib import Path
import time

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor

from src.world_model.m3w_native_gain_harm import standardized
from src.evaluation.m3w_conditional_cost_audit import strict_bits

ARMS = ('ramp', 'uniform')
POLICIES = tuple(a + '_' + p for a in ARMS for p in
                 ('forest', 'neural', 'neural_matched'))
OBJECTIVE = 'draw_and_distance_weighted_joint_fraction_squared_error'


def fit_targets(labels, distance, known, draws, loss_weights, cost_scale):
    y, d, k = np.asarray(labels, float), np.asarray(distance, float), np.asarray(known)
    draws, w = np.asarray(draws), np.asarray(loss_weights, float)
    n = len(d)
    if (d.shape != (n,) or y.shape != (n, 2) or k.shape != (n,) or k.dtype != bool
            or draws.shape != (n,) or draws.dtype.kind not in 'iu' or (draws < 0).any()
            or w.shape != (n,) or not np.isfinite(w).all() or (w[k] <= 0).any()
            or (w[~k] != 0).any() or (draws[~k] != 0).any()
            or not np.isfinite(d).all() or (d < 0).any()
            or not np.isfinite(y[k]).all() or (y[k] < 0).any()
            or not np.isfinite(cost_scale) or cost_scale <= 0):
        raise ValueError('Aligned complete targets, causal distance and frozen fitting weights required')
    if ((d[k] == 0) & (y[k].sum(1) != 0)).any():
        raise ValueError('Identical forecasts require zero cost')
    q = np.zeros_like(y)
    use = k & (d > 0)
    q[use] = y[use] / d[use, None]
    if (q.sum(1) > 1 + 1e-6).any():
        raise ValueError('Target violates causal forecast-disagreement bound')
    # Match the prior composition's roundoff-only simplex repair.
    q /= np.maximum(1., q.sum(1))[:, None]
    weight = draws.astype(float) * w * d / cost_scale
    positive = weight > 0
    if not positive.any():
        raise ValueError('No positive fitting mass')
    weight /= weight[positive].mean()
    return q, weight


def make_model(settings, seed, trees):
    return ExtraTreesRegressor(n_estimators=trees, criterion='squared_error',
        max_depth=settings['max_depth'], min_samples_leaf=settings['min_samples_leaf'],
        max_features=settings['max_features'], bootstrap=False, random_state=seed,
        n_jobs=settings['fit_threads'], warm_start=True)


def predict(model, x, distance, pr):
    d = np.asarray(distance, float)
    if d.shape != (len(x),) or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError('Finite causal disagreement required')
    z = standardized(x, pr)
    if not np.isfinite(z).all():
        raise ValueError('Nonfinite causal features')
    model.set_params(n_jobs=1)
    q = model.predict(z)
    if (q.shape != (len(x), 2) or not np.isfinite(q).all() or (q < 0).any()
            or (q.sum(1) > 1 + 1e-12).any()):
        raise ValueError('Leaf averages must remain in cost simplex')
    return q * d[:, None]


def selections(forest, neural, past, distance, ids):
    if (forest.shape != (len(ids), 2) or neural.shape != forest.shape
            or len(np.unique(ids)) != len(ids)):
        raise ValueError('Aligned causal scores and unique row ids required')
    f, n = strict_bits(forest, past, distance), strict_bits(neural, past, distance)
    pool = np.flatnonzero((distance > 0) & np.any(past[:, -1] != past[:, -2], axis=1))
    matched = np.zeros(len(ids), bool)
    gain = neural[:, 0] - neural[:, 1]
    matched[pool[np.lexsort((ids[pool], -gain[pool]))[:int(f.sum())]]] = True
    return dict(forest=f, neural=n, neural_matched=matched)


def fit(x, labels, distance, pr, draws, loss_weights, *, settings, seed, identity,
        directory, heartbeat, resume=False, stop_at=None):
    q, weight = fit_targets(labels, distance, pr['known'], draws, loss_weights, pr['cost_scale'])
    z = standardized(x, pr)
    if not np.isfinite(z).all():
        raise ValueError('Finite standardized features required')
    use = weight > 0
    path = Path(directory) / 'checkpoint.joblib'; path.parent.mkdir(parents=True, exist_ok=True)
    model, seconds, trace = make_model(settings, seed, 1), 0., []
    if path.exists():
        if not resume:
            raise ValueError('Existing checkpoint requires resume')
        cp = joblib.load(path)
        if (cp['identity'] != identity or cp['settings'] != settings or cp['seed'] != seed
                or cp['sklearn'] != sklearn.__version__ or cp['objective'] != OBJECTIVE):
            raise ValueError('Changed forest resume identity')
        np.testing.assert_array_equal(cp['draws'], draws)
        np.testing.assert_array_equal(cp['sample_weight'], weight)
        for key in ('mean', 'std', 'known'):
            np.testing.assert_array_equal(cp['preprocess'][key], pr[key])
        model, seconds, trace = cp['model'], cp['seconds'], cp['trace']
    trees = len(getattr(model, 'estimators_', []))
    limit = settings['trees'] if stop_at is None else stop_at
    if not 0 < limit <= settings['trees'] or limit < trees:
        raise ValueError('Fixed nondecreasing forest budget required')
    started = time.monotonic()
    try:
        while trees < limit:
            trees = min(limit, trees + settings['checkpoint_every'])
            model.set_params(n_estimators=trees, n_jobs=settings['fit_threads'])
            with joblib.parallel_backend('threading'):
                model.fit(z[use], q[use], sample_weight=weight[use])
            # A fixed deterministic fitting diagnostic, never used to select a checkpoint.
            model.set_params(n_jobs=1)
            pred = model.predict(z[use])
            mse = float(np.average(((pred-q[use])**2).mean(1), weights=weight[use]))
            row = dict(trees=trees, fitting_fraction_MSE=mse,
                       nodes=sum(t.tree_.node_count for t in model.estimators_))
            trace.append(row)
            cp = dict(identity=identity, settings=settings, seed=seed, objective=OBJECTIVE,
                sklearn=sklearn.__version__, model=model, preprocess=pr, draws=draws,
                sample_weight=weight, seconds=seconds+time.monotonic()-started, trace=trace)
            temp = path.with_suffix('.tmp'); joblib.dump(cp, temp, compress=0); os.replace(temp, path)
            heartbeat(state='training', **row)
    except BaseException:
        heartbeat(state='interrupted_resume_last_atomic_checkpoint', trees=trees)
        raise
    return model, dict(complete=trees == settings['trees'], trees=trees,
        seconds=seconds+time.monotonic()-started, effective_fit_rows=int(use.sum()),
        complete_rows=int(pr['known'].sum()), zero_weight_rows=int((~use).sum()),
        reused_neural_sampler_draws=int(draws.sum()), trace=trace,
        nodes=sum(t.tree_.node_count for t in model.estimators_))
