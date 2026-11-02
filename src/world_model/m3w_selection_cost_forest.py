"""Matched conventional cost control for the complete selection forecast bank."""
import hashlib
import os
from pathlib import Path
import time

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor

from src.world_model import m3w_selection_controller as base
from src.world_model.m3w_source_forest import exact_forest, project_moments

ARMS = ('row_forest', 'context_forest')
OBJECTIVE = 'query_draw_matched_mean_five_cost_MSE_plus_half_absolute_gain_MSE'
FACTORS = np.r_[np.full(5, np.sqrt(6/5)), np.sqrt(3)]


def fingerprint(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    h.update(memoryview(value).cast('B'))
    return h.hexdigest()


def encoded(causal, preprocess, arm):
    if set(causal) != base.CAUSAL_KEYS or arm not in ARMS:
        raise ValueError('Registered past-only schema and arm required; no future labels accepted')
    x = np.asarray(causal['x']); scale = np.asarray(causal['scales'])
    if (x.ndim != 3 or x.shape[1] != 7 or scale.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError('Finite aligned causal actions and positive scales required')
    z = np.clip((x-preprocess['mean'])/preprocess['std'], -preprocess['clip'], preprocess['clip']).astype(np.float32)
    env = (causal['envelope']/scale[:, None]).astype(np.float32)
    if env.shape != x.shape[:2] or not np.isfinite(env).all() or (env < 0).any():
        raise ValueError('Finite causal disagreement required')
    context = None
    if arm == 'context_forest':
        context = np.empty((*z.shape[:2], z.shape[-1]+1), np.float32)
        logenv = np.log1p(env)
        for ix in causal['groups']:
            if len(ix) == 1:
                context[ix] = 0
            else:
                context[ix, :, :-1] = (z[ix].sum(0, dtype=np.float64)-z[ix])/(len(ix)-1)
                context[ix, :, -1] = (logenv[ix].sum(0, dtype=np.float64)-logenv[ix])/(len(ix)-1)
    return z, env, context


def feature_rows(z, env, context, rows, actions):
    values = [z[rows, actions], np.eye(7, dtype=np.float32)[actions], np.log1p(env[rows, actions])[:, None]]
    if context is not None:
        values.append(context[rows, actions])
    return np.concatenate(values, axis=1).astype(np.float32, copy=False)


def augmented_targets(truth, rms):
    truth = np.asarray(truth)
    if truth.ndim != 2 or truth.shape[1] != 5 or not np.isfinite(truth).all() or (truth < 0).any():
        raise ValueError('Only observed nonnegative five-cost training targets permitted')
    gain = (truth[:, 0]-truth[:, 1])/np.linalg.norm(rms[:2])
    return (np.column_stack((truth/rms, gain))*FACTORS).astype(np.float32)


def training_matrix(data, query_counts, arm):
    c = data['causal']; counts = np.asarray(query_counts)
    if counts.shape != (len(c['groups']),) or counts.dtype.kind not in 'iu' or (counts <= 0).any():
        raise ValueError('Original complete positive query-draw multiplicities required')
    if not np.array_equal(np.sort(np.concatenate(c['groups'])), np.arange(len(c['x']))):
        raise ValueError('Every causal row must belong to one query')
    legal = c['allowed'] & data['known'][:, None]
    actions_per_row = legal.sum(1)
    row_weight = np.zeros(len(legal), np.float64)
    for q, ix in enumerate(c['groups']):
        support = actions_per_row[ix] > 0
        if support.any():
            selected = ix[support]
            row_weight[selected] = counts[q]/support.sum()/actions_per_row[selected]
    rows, actions = np.nonzero(legal)
    if not len(rows):
        raise ValueError('Observed legal SOURCE actions required')
    z, env, context = encoded(c, data['preprocess'], arm)
    x = feature_rows(z, env, context, rows, actions)
    truth = data['truth'].numpy()[rows, actions]
    y = augmented_targets(truth, data['preprocess']['rms'])
    weight = row_weight[rows]; rescale = float(weight.mean()); weight /= rescale
    return dict(x=x, y=y, weight=weight, query_weight_rescale=rescale,
        positive_candidate_rows=len(rows), positive_query_rows=int(np.unique(rows).size),
        minimum_query_visits=int(counts.min()),
        matrix_bytes=x.nbytes+y.nbytes+weight.nbytes,
        input_hashes={k:fingerprint(v) for k,v in dict(features=x, targets=y, weights=weight,
            query_counts=counts, rows=rows, actions=actions).items()})


def fit(matrix, *, preprocess, arm, settings, seed, identity, path,
        heartbeat=lambda **_:None, guard=lambda:None, resume=False, stop_at=None):
    if arm not in ARMS:
        raise ValueError('Registered forest arm required')
    binding = dict(preprocess=preprocess, arm=arm, settings=settings, seed=seed,
        identity=identity, input_hashes=matrix['input_hashes'], objective=OBJECTIVE,
        sklearn=sklearn.__version__, feature_dimension=matrix['x'].shape[1],
        positive_candidate_rows=matrix['positive_candidate_rows'],
        positive_query_rows=matrix['positive_query_rows'],
        minimum_query_visits=matrix['minimum_query_visits'],
        query_weight_rescale=matrix['query_weight_rescale'], matrix_bytes=matrix['matrix_bytes'])
    model = ExtraTreesRegressor(n_estimators=1, criterion='squared_error', max_depth=settings['max_depth'],
        min_samples_leaf=settings['min_samples_leaf'], max_features=settings['max_features'],
        bootstrap=False, random_state=seed, n_jobs=settings['fit_threads'], warm_start=True)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    trees, elapsed, trace = 0, 0., []
    if path.exists():
        if not resume:
            raise ValueError('Explicit unchanged forest resume required')
        old = joblib.load(path)
        for k,v in binding.items(): base.exact(old[k], v)
        model, elapsed, trace = old['model'], old['seconds'], old['trace']
        trees = len(getattr(model, 'estimators_', []))
    limit = settings['trees'] if stop_at is None else stop_at
    if not 0 <= trees <= limit <= settings['trees'] or limit <= 0:
        raise ValueError('Fixed nondecreasing forest budget required')
    monitor = np.linspace(0, len(matrix['x'])-1, min(4096, len(matrix['x'])), dtype=int)
    start = time.monotonic()
    while trees < limit:
        guard(); trees = min(limit, (trees//settings['checkpoint_every']+1)*settings['checkpoint_every'])
        heartbeat(state='tree_block_start', arm=arm, trees=trees, total=settings['trees'])
        model.set_params(n_estimators=trees, n_jobs=settings['fit_threads'])
        with joblib.parallel_backend('threading'):
            model.fit(matrix['x'], matrix['y'], sample_weight=matrix['weight'])
        model.set_params(n_jobs=1)
        if trees % settings['checkpoint_every'] == 0 or trees == settings['trees']:
            error = np.square(model.predict(matrix['x'][monitor])-matrix['y'][monitor]).mean(1)
            trace.append(dict(trees=trees, fixed_SOURCE_monitor_objective=float(
                np.average(error, weights=matrix['weight'][monitor])), monitor_candidates=len(monitor)))
        guard()
        state = dict(**binding, model=model, trees=trees, trace=trace,
            seconds=elapsed+time.monotonic()-start,
            nodes=sum(t.tree_.node_count for t in model.estimators_), unknown_supervised_rows=0)
        temp = path.with_suffix('.tmp'); joblib.dump(state, temp, compress=3); os.replace(temp, path)
        heartbeat(state='tree_block_saved', arm=arm, trees=trees, total=settings['trees'])
    return joblib.load(path)


def predict(state, causal, chunk_size=1024):
    pr = state['preprocess']; z, env, context = encoded(causal, pr, state['arm'])
    model = state['model']; model.set_params(n_jobs=1)
    result = np.empty((len(z), 7, 5), np.float64)
    for lo in range(0, len(z), chunk_size):
        hi = min(len(z), lo+chunk_size)
        rows = np.repeat(np.arange(lo, hi), 7); actions = np.tile(np.arange(7), hi-lo)
        raw = model.predict(feature_rows(z, env, context, rows, actions))/FACTORS
        cost = raw[:, :5]*pr['rms']
        np.testing.assert_allclose(raw[:, 5], (cost[:, 0]-cost[:, 1])/np.linalg.norm(pr['rms'][:2]),
                                   rtol=1e-5, atol=2e-6)
        cost = cost.reshape(-1, 7, 5)
        # Keep reference costs candidate-invariant, as in the matched neural decoder.
        cost[:, :, 2:4] = cost[:, :, 2:4].mean(1, keepdims=True)
        cost = project_moments(cost.reshape(-1, 5), env[lo:hi].ravel()).reshape(-1, 7, 5)
        result[lo:hi] = cost*causal['scales'][lo:hi, None, None]
    return result
