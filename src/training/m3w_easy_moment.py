"""Atomic, resumable multi-output moments matched to frozen source draw counts."""
import os
from pathlib import Path
import time

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor

from src.world_model.m3w_native_gain_harm import standardized


def predict(model, x, pr):
    model.set_params(n_jobs=1)
    return model.predict(standardized(x,pr))


def fit(x, target, distance, pr, draws, *, settings, seed, identity,
        directory, heartbeat, resume=False, stop_at=None):
    y, d, draws = map(np.asarray,(target,distance,draws))
    if (y.shape!=(len(x),4) or d.shape!=(len(x),) or draws.shape!=d.shape
            or draws.dtype.kind not in 'iu' or (draws<0).any()
            or not np.isfinite(y).all() or (y<0).any() or (y>1+1e-12).any()
            or not np.isfinite(d).all() or (d<0).any() or draws[~pr['known']].any()):
        raise ValueError('Valid source-only moment labels and matched draws required')
    weight = draws.astype(float)*(d>0)
    use = weight>0
    if not use.any():
        raise ValueError('No moment fitting support')
    z = standardized(x,pr)
    if not np.isfinite(z).all():
        raise ValueError('Nonfinite causal input')
    path = Path(directory)/'checkpoint.joblib'
    path.parent.mkdir(parents=True,exist_ok=True)
    model = ExtraTreesRegressor(n_estimators=1, criterion='squared_error',
        max_depth=settings['max_depth'], min_samples_leaf=settings['min_samples_leaf'],
        max_features=settings['max_features'], bootstrap=False, warm_start=True,
        random_state=seed, n_jobs=settings['fit_threads'])
    seconds, trace = 0., []
    if path.exists():
        if not resume:
            raise ValueError('Checkpoint exists; explicit resume required')
        cp = joblib.load(path)
        if (cp['identity']!=identity or cp['settings']!=settings or cp['seed']!=seed
                or cp['sklearn']!=sklearn.__version__):
            raise ValueError('Changed resume identity')
        np.testing.assert_array_equal(cp['draws'],draws)
        np.testing.assert_array_equal(cp['sample_weight'],weight)
        for key in ('mean','std','known'):
            np.testing.assert_array_equal(cp['preprocess'][key],pr[key])
        model, seconds, trace = cp['model'], cp['seconds'], cp['trace']
    trees = len(getattr(model,'estimators_',[]))
    limit = settings['trees'] if stop_at is None else stop_at
    if not 0<limit<=settings['trees'] or limit<trees:
        raise ValueError('Nondecreasing fixed tree budget required')
    started = time.monotonic()
    while trees<limit:
        trees = min(limit,trees+settings['checkpoint_every'])
        model.set_params(n_estimators=trees,n_jobs=settings['fit_threads'])
        with joblib.parallel_backend('threading'):
            model.fit(z[use],y[use],sample_weight=weight[use])
        model.set_params(n_jobs=1)
        mse = np.average((model.predict(z[use])-y[use])**2,axis=0,weights=weight[use])
        trace.append(dict(trees=trees, fitting_mse_by_target=mse.tolist(),
                          fitting_mean_mse=float(mse.mean())))
        cp = dict(identity=identity,settings=settings,seed=seed,sklearn=sklearn.__version__,
            model=model,preprocess=pr,draws=draws,sample_weight=weight,trace=trace,
            seconds=seconds+time.monotonic()-started)
        tmp = path.with_suffix('.tmp')
        joblib.dump(cp,tmp,compress=0); os.replace(tmp,path)
        heartbeat(state='training',**trace[-1])
    return dict(complete=trees==settings['trees'],trees=trees,
        seconds=seconds+time.monotonic()-started,trace=trace,sampled_rows=int(draws.sum()),
        effective_fit_rows=int(use.sum()),unknown_sampled=int(draws[~pr['known']].sum()))
