"""Past-only release-label support probe; not a physical-visibility model."""
import hashlib
import os
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ARMS = ('geometry', 'geometry_detector')


def features(*, geometry, detector_history, arm):
    g, d = np.asarray(geometry), np.asarray(detector_history)
    if (arm not in ARMS or g.shape != (len(g), 476) or d.shape != (len(g), 64)
            or not np.isfinite(g).all() or not np.isfinite(d).all()):
        raise ValueError('Only the frozen past geometry476 and detector64 schema is accepted')
    return np.asarray(g if arm == 'geometry' else np.concatenate((g, d), 1), np.float32)


def preprocessing(x, sites):
    names = sorted(set(sites))
    weight = np.zeros(len(x), float)
    mean = np.zeros(x.shape[1], float); second = mean.copy()
    for site in names:
        at = np.asarray(sites) == site
        z = x[at].astype(float)
        mean += z.mean(0)/len(names); second += np.square(z).mean(0)/len(names)
        weight[at] = len(x)/(len(names)*at.sum())
    std = np.sqrt(np.maximum(second-mean*mean, 1e-12))
    return dict(mean=mean, std=std), weight


def normalize(x, state, clip):
    return np.clip((x-state['mean'])/state['std'], -clip, clip).astype(np.float32)


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial.joblib')
    joblib.dump(value, tmp, compress=3); os.replace(tmp, path)


def tree_fingerprint(model):
    h = hashlib.sha256()
    for tree in model.estimators_:
        for key in ('children_left', 'children_right', 'feature', 'threshold', 'value',
                    'n_node_samples', 'weighted_n_node_samples', 'impurity'):
            v = np.asarray(getattr(tree.tree_, key))
            h.update(str((key, v.shape, v.dtype.str)).encode()); h.update(v.tobytes())
    return h.hexdigest()


def fit(x, y, sites, *, settings, seed, identity, path, clip, chunk=32,
        resume=False, stop_at=None, heartbeat=lambda **_:None, guard=lambda:None):
    x, y, sites = np.asarray(x), np.asarray(y), np.asarray(sites)
    if (y.shape != (len(x),) or sites.shape != y.shape or y.dtype != bool
            or len(set(y)) != 2 or not np.isfinite(x).all()):
        raise ValueError('Training-only paired features and both binary release-support classes required')
    pre, weight = preprocessing(x, sites); z = normalize(x, pre, clip)
    binding = dict(settings=settings, seed=seed, identity=identity, clip=clip,
        preprocess={k:v.tolist() for k,v in pre.items()}, training_rows=len(x),
        training_sites=sorted(set(sites)), unknown_labels=int(y.sum()))
    target = settings['n_estimators'] if stop_at is None else stop_at
    if not 0 < target <= settings['n_estimators']: raise ValueError('Nonzero fixed forest budget required')
    elapsed = 0.; path = Path(path)
    if path.exists():
        if not resume: raise ValueError('Explicit same-binding resume required')
        old = joblib.load(path)
        if old['binding'] != binding: raise ValueError('Resume binding changed')
        model, elapsed = old['model'], old['seconds']
        if len(model.estimators_) > target: raise ValueError('No rollback of a completed prefix')
    else:
        model = ExtraTreesClassifier(**{**settings, 'n_estimators':min(chunk, target)}, random_state=seed)
    start = time.monotonic()
    done = len(model.estimators_) if hasattr(model, 'estimators_') else 0
    while done < target:
        guard(); count = min(done+chunk, target); model.set_params(n_estimators=count, n_jobs=settings['n_jobs'])
        model.fit(z, y, sample_weight=weight); done = count
        state = dict(binding=binding, model=model, trees=done, tree_fingerprint=tree_fingerprint(model),
            seconds=elapsed+time.monotonic()-start)
        save(path, state)
        heartbeat(state='forest_checkpoint', trees=done, budget=settings['n_estimators'],
                  fitting_rows=len(x), unknown_labels=int(y.sum()))
    return joblib.load(path)


def predict(state, x):
    model = state['model']; model.set_params(n_jobs=1)
    z = normalize(x, state['binding']['preprocess'], state['binding']['clip'])
    classes = np.asarray(model.classes_)
    if not np.array_equal(classes, [False, True]): raise ValueError('Fixed binary target encoding required')
    p = model.predict_proba(z)[:, 1]
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Finite binary probabilities required')
    return p


def score(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if y.dtype != bool or p.shape != y.shape or not len(y): raise ValueError('Aligned nonempty labels and scores required')
    defined = len(set(y)) == 2
    bins = np.minimum((p*10).astype(int), 9)
    calibration = []
    for k in range(10):
        at = bins == k
        calibration.append(dict(bin=k, rows=int(at.sum()),
            probability=float(p[at].mean()) if at.any() else None,
            positive_rate=float(y[at].mean()) if at.any() else None))
    ece = sum(r['rows']/len(y)*abs(r['probability']-r['positive_rate']) for r in calibration if r['rows'])
    ap = float(average_precision_score(y, p)) if defined else None
    base = float(y.mean())
    return dict(rows=len(y), unknown_rows=int(y.sum()), positive_rate=base,
        AUROC=float(roc_auc_score(y, p)) if defined else None, AUPRC=ap,
        AUPRC_minus_prevalence=ap-base if ap is not None else None,
        Brier=float(brier_score_loss(y, p)), ECE=ece, calibration_curve=calibration,
        unknown_score_mean=float(p[y].mean()) if y.any() else None,
        known_score_mean=float(p[~y].mean()) if (~y).any() else None,
        status='defined' if defined else 'one_class_AUROC_AUPRC_not_estimable')
