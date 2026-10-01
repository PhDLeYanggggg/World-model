"""Five-moment nonlinear estimator control, not an independent safety certificate."""
import hashlib
import os
from pathlib import Path
import time

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor

from src.world_model import m3w_inner_separability as core

OBJECTIVE = 'query_weighted_five_moment_and_three_signed_score_quadratic_v1'
FACTORS = np.sqrt(np.r_[np.full(5, .8), np.full(3, 4 / 3)])


def fingerprint(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def causal_inputs(x, envelope, pr):
    x, env = np.asarray(x), np.asarray(envelope)
    if (x.ndim != 2 or x.shape[1] != len(pr['mean']) or env.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isfinite(env).all() or (env < 0).any()):
        raise ValueError('Finite past-only features and causal disagreement required')
    z = (x - pr['mean']) / pr['std']
    support = np.sqrt(np.mean(z * z, axis=1)) <= pr['support_limit']
    # The neural decoder already receives envelope; this exposes the same information to trees.
    return np.column_stack((np.clip(z, -pr['clip'], pr['clip']),
                            np.log1p(env / pr['scale']))).astype(np.float32), support


def transformed_targets(y, pr):
    y = np.asarray(y, float)
    if y.ndim != 2 or y.shape[1] != 5 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Only known five-moment targets may enter fitting')
    normalized = y / pr['scale']
    return np.column_stack((normalized, core.signed(normalized))) / pr['rms'] * FACTORS


def project_moments(p, env):
    p, env = np.asarray(p, float).copy(), np.asarray(env, float)
    if (p.shape != (len(env), 5) or not np.isfinite(p).all() or (p < -1e-12).any()
            or not np.isfinite(env).all() or (env < 0).any()):
        raise ValueError('Finite nonnegative moments and envelope required')
    p = np.maximum(p, 0)
    mass = p[:, :2].sum(1)
    ratio = np.minimum(1., np.divide(env, mass, out=np.ones_like(env), where=mass > 0))
    p[:, :2] *= ratio[:, None]
    p[:, 0] = np.minimum(p[:, 0], p[:, 2])
    p[:, 3] = np.minimum(p[:, 3], p[:, 2])
    p[:, 4] = np.minimum(p[:, 4], p[:, 1])
    return p


def predict(state, x, envelope):
    pr = state['preprocess']; z, support = causal_inputs(x, envelope, pr)
    state['model'].set_params(n_jobs=1)
    extended = state['model'].predict(z) / FACTORS * pr['rms'] * pr['scale']
    np.testing.assert_allclose(extended[:, 5:], core.signed(extended[:, :5]), rtol=2e-6, atol=1e-7)
    return project_moments(extended[:, :5], envelope), support


def signed_score(p, y, pr, sites, recordings, frames):
    known = np.isfinite(y).all(1)
    if not known.any():
        raise ValueError('Known validation targets required')
    w, _ = core.weights(sites, recordings, frames, known)
    err = (core.signed(p[known]) - core.signed(y[known])) / pr['scale'] / pr['rms'][5:]
    return float((w[known, None] * err**2).sum() / 3)


def fit(x, env, y, sites, recordings, frames, pr, *, settings, seed, identity,
        path, heartbeat, resume=False, stop_at=None):
    fresh = core.preprocess(x, env, y, sites, recordings, frames, training_site=pr['training_site'])
    core.exact(pr, fresh)
    z, _ = causal_inputs(x, env, pr)
    known = np.isfinite(y).all(1)
    w, _ = core.weights(sites, recordings, frames, known)
    assert not w[~known].any()
    truth = transformed_targets(y[known], pr)
    z, w = z[known], w[known]
    inputs = dict(features=fingerprint(z), targets=fingerprint(truth), weights=fingerprint(w),
                  known_mask=fingerprint(known))
    model = ExtraTreesRegressor(n_estimators=1, criterion='squared_error',
        max_depth=settings['max_depth'], min_samples_leaf=settings['min_samples_leaf'],
        max_features=settings['max_features'], bootstrap=False, random_state=seed,
        n_jobs=settings['fit_threads'], warm_start=True)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    trace, seconds = [], 0.
    if path.exists():
        if not resume:
            raise ValueError('Existing checkpoint requires explicit resume')
        state = joblib.load(path)
        for k, v in dict(identity=identity, settings=settings, seed=seed, preprocess=pr,
                         sklearn=sklearn.__version__, objective=OBJECTIVE, input_hashes=inputs).items():
            core.exact(state[k], v)
        model, trace, seconds = state['model'], state['trace'], state['seconds']
    trees = len(getattr(model, 'estimators_', []))
    limit = settings['trees'] if stop_at is None else stop_at
    if not trees <= limit <= settings['trees'] or limit <= 0:
        raise ValueError('Nondecreasing fixed forest budget required')
    began = time.monotonic()
    while trees < limit:
        trees = min(limit, trees + settings['checkpoint_every'])
        model.set_params(n_estimators=trees, n_jobs=settings['fit_threads'])
        with joblib.parallel_backend('threading'):
            model.fit(z, truth, sample_weight=w)
        model.set_params(n_jobs=1)
        loss = float(np.average(((model.predict(z) - truth)**2).mean(1), weights=w))
        row = dict(trees=trees, unprojected_training_loss=loss,
                   nodes=sum(t.tree_.node_count for t in model.estimators_))
        trace.append(row)
        state = dict(identity=identity, settings=settings, seed=seed, preprocess=pr,
            sklearn=sklearn.__version__, objective=OBJECTIVE, input_hashes=inputs, model=model,
            trace=trace, seconds=seconds + time.monotonic() - began,
            known_training_rows=int(known.sum()), unknown_training_rows=int((~known).sum()))
        temp = path.with_suffix('.tmp')
        joblib.dump(state, temp, compress=3); os.replace(temp, path)
        heartbeat(**row)
    return joblib.load(path)


def exact_forest(a, b):
    for k in a:
        if k not in ('seconds', 'model'):
            core.exact(a[k], b[k])
    assert len(a['model'].estimators_) == len(b['model'].estimators_)
    for ta, tb in zip(a['model'].estimators_, b['model'].estimators_):
        assert ta.random_state == tb.random_state
        core.exact(ta.tree_.__getstate__(), tb.tree_.__getstate__())
