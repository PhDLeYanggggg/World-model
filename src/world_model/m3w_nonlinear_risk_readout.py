"""Nonlinear risk-error readout on frozen neural features, not new dynamics."""
import os
from pathlib import Path
import platform
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 before numerical imports')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from src.world_model import m3w_frozen_readout_refit as linear

ARMS = ('score_only', 'latent_score')
base = linear.base


def features(extracted, arm):
    if arm not in ARMS:
        raise ValueError('Fixed score-only or latent-plus-score arm required')
    x = extracted['prediction']
    if arm == 'latent_score':
        x = np.column_stack((extracted['design'][:, :-1], x))
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Finite inference-derived features required')
    return x.astype(np.float64)


def matrix(data, warm, frozen, *, heartbeat=lambda **_: None, guard=lambda: None):
    c = data['causal']
    base.exact(warm['fit_ids'], c['ids'])
    for key, value in dict(input_hash=linear.digest(data['z'].numpy()),
            target_hash=linear.digest(data['truth_by_arm']['bound_aware'].numpy()),
            proposal_mask_hash=linear.digest(c['allowed'])).items():
        base.exact(warm[key], value); base.exact(frozen[key], value)
    for key in ('preprocess', 'settings', 'fit_ids', 'source_sites', 'seed'):
        base.exact(frozen[key], warm[key])
    if warm['step'] != 2000 or frozen['new_encoder_updates'] != 0:
        raise ValueError('Completed unchanged neural parent required')
    counts = warm['sampler']['counts']
    base.exact(frozen['query_counts_hash'], linear.digest(counts))
    w = linear.query_weights(c, counts)
    extracted = linear.extract(frozen, c, heartbeat=heartbeat, guard=guard)
    rows = np.flatnonzero(w > 0)
    inputs = {arm: features(extracted, arm)[rows] for arm in ARMS}
    offset = extracted['prediction'][rows].copy()
    # Causal hidden states, offsets and proposal row indices precede target access.
    truth = (linear.parent.signed(data['truth_by_arm']['bound_aware']) /
        linear.torch.from_numpy(linear.parent.normalizer(data['preprocess']))).numpy()
    truth = truth[rows, extracted['choice'][rows]]
    solver_target = np.zeros((len(w), 2), np.float64); solver_target[rows] = truth
    base.exact(frozen['diagnostics']['target_solver_sha256'], linear.digest(solver_target))
    result = dict(**inputs, truth=truth, offset=offset, residual=truth-offset,
        weights=w[rows], rows=rows, known=data['observed_future'][rows])
    result['hashes'] = {k: linear.digest(v) for k, v in result.items()}
    result['stats'] = dict(rows=len(rows), known_rows=int(result['known'].sum()),
        unknown_bound_rows=int((~result['known']).sum()), empirical_weight_mass=float(w.sum()),
        score_dimensions=inputs['score_only'].shape[1], latent_score_dimensions=inputs['latent_score'].shape[1])
    return result


def new_regressor(settings, seed):
    return HistGradientBoostingRegressor(loss='squared_error', learning_rate=settings['learning_rate'],
        max_iter=1, max_leaf_nodes=settings['max_leaf_nodes'], min_samples_leaf=settings['min_samples_leaf'],
        l2_regularization=0., max_bins=255, categorical_features=None,
        early_stopping=False, warm_start=True, random_state=seed)


def correction(models, x):
    with threadpool_limits(limits=4):
        result = np.column_stack([m.predict(x) for m in models])
    if result.shape != (len(x), 2) or not np.isfinite(result).all():
        raise FloatingPointError('Finite two-output risk correction required')
    return result


def fit(m, frozen, *, arm, settings, identity, path, resume=False, stop_at=None,
        heartbeat=lambda **_: None, guard=lambda: None):
    if arm not in ARMS or not 0 < settings['learning_rate'] <= 1:
        raise ValueError('Registered model family and learning rate required')
    for key, digest in m['hashes'].items():
        base.exact(digest, linear.digest(m[key]))
    if (m['residual'].shape != (len(m['weights']), 2) or (m['weights'] <= 0).any()
            or any(not np.isfinite(m[k]).all() for k in (*ARMS, 'residual', 'truth', 'offset', 'weights'))):
        raise ValueError('Aligned finite prepared fitting matrix required')
    binding = dict(arm=arm, settings=settings, identity=identity, sklearn=sklearn.__version__,
        matrix_hashes=m['hashes'], stats=m['stats'], frozen=frozen,
        source_sites=frozen['source_sites'], fit_ids=frozen['fit_ids'], risk_budget=.02,
        new_encoder_updates=0, new_adam_updates=0, early_stopping=False,
        threshold_search=False, calibrated_safety_bound=False)
    models = [new_regressor(settings, frozen['seed']) for _ in range(2)]
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    done, elapsed, trace = 0, 0., []
    if path.exists():
        if not resume:
            raise ValueError('Explicit same-binding resume required')
        old = joblib.load(path)
        for key, value in binding.items():
            base.exact(old[key], value)
        models, done, elapsed, trace = (old[k] for k in ('models', 'rounds', 'seconds', 'trace'))
    limit = settings['rounds'] if stop_at is None else stop_at
    if not 0 <= done <= limit <= settings['rounds'] or limit <= 0:
        raise ValueError('No rollback or budget change')
    x, y, w = m[arm], m['residual'], m['weights']
    # A common scaling avoids tiny Hessian weights; relative query weights stay fixed.
    sample_weight = w / w.mean()
    mse = lambda p: float(np.sum(w[:, None]*(p-m['truth'])**2)/2)
    before = mse(m['offset']); started = time.monotonic()
    try:
        while done < limit:
            guard()
            next_round = min(limit, (done//settings['checkpoint_every']+1)*settings['checkpoint_every'])
            heartbeat(state='nonlinear_risk_block_start', arm=arm, rounds=next_round, total=settings['rounds'])
            with threadpool_limits(limits=4):
                for j, model in enumerate(models):
                    model.set_params(max_iter=next_round)
                    model.fit(x, y[:, j], sample_weight=sample_weight)
            if any(model.n_iter_ != next_round for model in models):
                raise ValueError('Every target must finish the fixed round budget')
            done = next_round
            after = mse(m['offset']+correction(models, x))
            trace.append(dict(rounds=done, before_fitting_MSE=before, fitting_MSE=after,
                fitting_loss_reduction_percent=100*(1-after/before) if before else None))
            state = dict(**binding, models=models, rounds=done, trace=trace,
                seconds=elapsed+time.monotonic()-started, raw_weight_mass=float(w.sum()),
                node_count=sum(len(p[0].nodes) for model in models for p in model._predictors))
            guard(); temp = path.with_suffix('.tmp'); joblib.dump(state, temp, compress=3); os.replace(temp, path)
            heartbeat(state='nonlinear_risk_block_saved', arm=arm, **trace[-1])
    except (KeyboardInterrupt, TimeoutError):
        heartbeat(state='interrupted_resume_last_complete_atomic_block', arm=arm, rounds=done)
        raise
    return joblib.load(path)


def exact_fit(a, b):
    for key in set(a)-{'seconds', 'models'}:
        base.exact(a[key], b[key])
    for x, y in zip(a['models'], b['models']):
        base.exact(x.get_params(), y.get_params())
        if x.n_iter_ != y.n_iter_:
            raise AssertionError('Different boosting budgets')
        np.testing.assert_array_equal(x._baseline_prediction, y._baseline_prediction)
        for ax, ay in zip(x._bin_mapper.bin_thresholds_, y._bin_mapper.bin_thresholds_):
            np.testing.assert_array_equal(ax, ay)
        for ax, ay in zip(x._predictors, y._predictors):
            for key in ('nodes', 'binned_left_cat_bitsets', 'raw_left_cat_bitsets'):
                np.testing.assert_array_equal(getattr(ax[0], key), getattr(ay[0], key))


def predict(state, causal, proposal):
    if set(causal) != base.CAUSAL_KEYS:
        raise ValueError('Only original causal inputs; future labels forbidden')
    proposal = np.asarray(proposal)
    if (proposal.shape != (len(causal['x']),) or proposal.dtype.kind not in 'iu'
            or (proposal < -1).any() or (proposal >= 7).any()):
        raise ValueError('Aligned frozen proposal required')
    rows = np.flatnonzero(proposal >= 0)
    if not causal['allowed'][rows, proposal[rows]].all():
        raise ValueError('Proposed action outside original causal support')
    mask = np.zeros_like(causal['allowed']); mask[rows, proposal[rows]] = True
    c = dict(causal, allowed=mask)
    frozen = state['frozen']
    extracted = linear.extract(frozen, c)
    result = linear.predict(frozen, causal)
    if len(rows):
        x = features(extracted, state['arm'])[rows]
        prediction = extracted['prediction'][rows] + correction(state['models'], x)
        result[rows, proposal[rows]] = prediction * linear.parent.normalizer(frozen['preprocess']) * causal['scales'][rows, None]
    return result
