"""Fixed source-held-out linear/nonlinear probes with explicit no-selection scope."""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from src.evaluation import m3w_candidate_neighbor_probe as previous

ARMS = ('summary120', 'aligned180', 'misaligned180')
FAMILIES = ('ridge', 'extra_trees')
TREE = dict(n_estimators=64, max_depth=8, min_samples_leaf=32,
    max_features=1., bootstrap=False, random_state=17, n_jobs=4)


def fit_predict(x, target, train, held, weights, family):
    if family == 'ridge':
        pred, state = previous.fit_predict(x, target, train, held, weights, penalty=.01)
        return pred, dict(**state, family=family)
    if family != 'extra_trees' or not train.any() or not held.any() or (train & held).any():
        raise ValueError('Registered family and source-disjoint fitting required')
    z, pr = previous.normalized(x, train, weights)
    w = weights[train]/weights[train].sum()
    scale = np.sqrt(np.einsum('n,nc->c', w, target[train]**2)).clip(1e-6)
    model = ExtraTreesRegressor(**TREE)
    model.fit(z[train, 1:], target[train]/scale, sample_weight=w)
    # Sum tree outputs in a fixed order to avoid thread-order reduction drift.
    state = dict(**pr, target_scale=scale, model=model, family=family)
    return predict(state, x[held]), state


def predict(state, x):
    z = np.clip((x-state['mean'])/state['std'], -8, 8)
    if state['family'] == 'ridge':
        return np.column_stack((np.ones(len(z)), z))@state['coefficient']*state['target_scale']
    out = np.zeros((len(z), 2), float)
    for estimator in state['model'].estimators_:
        out += estimator.predict(z)
    return out/len(state['model'].estimators_)*state['target_scale']


def fingerprint(state):
    result = {k:state[k] for k in ('mean', 'std', 'target_scale')}
    if state['family'] == 'ridge':
        result['coefficient'] = state['coefficient']
    else:
        for i, estimator in enumerate(state['model'].estimators_):
            for k, value in estimator.tree_.__getstate__().items():
                value = np.asarray(value)
                if value.dtype.names:
                    for field in value.dtype.names:
                        result[f'tree{i}_{k}_{field}'] = value[field].copy()
                else:
                    result[f'tree{i}_{k}'] = value
    return result


def summarize(rows):
    if len(rows) < 2 or len({r['held_source'] for r in rows}) != len(rows):
        raise ValueError('At least two unique source folds required')
    result = {}
    for family in FAMILIES:
        mean = {arm:{key:float(np.mean([r['families'][family][arm][key] for r in rows]))
            for key in ('normalized_MSE', 'total_joint_cost', 'false_safe_cost', 'false_reject_cost')}
            for arm in ARMS}
        a, b, null = (mean[k] for k in ARMS)
        gain = 1-b['normalized_MSE']/a['normalized_MSE'] if a['normalized_MSE'] > 0 else None
        sign = b['total_joint_cost'] <= a['total_joint_cost']
        harm = b['false_safe_cost'] <= a['false_safe_cost']
        aligned = b['normalized_MSE'] < null['normalized_MSE'] and b['total_joint_cost'] <= null['total_joint_cost']
        result[family] = dict(summary=mean, MSE_improvement=gain, joint_cost_nonworse=bool(sign),
            false_safe_cost_nonworse=bool(harm), beats_misaligned=bool(aligned),
            passes_internal_screen=bool(gain is not None and gain >= .01 and sign and harm and aligned))
    return result
