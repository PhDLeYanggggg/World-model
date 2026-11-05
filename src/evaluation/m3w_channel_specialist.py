"""Separate all/easy forest partitions, preserving original signed risk units."""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from src.evaluation import m3w_peer_path_probe as parent

ARMS = ('shared32', 'separate64')


def fit(x, y, train, weights, arm):
    x, y, train, weights = map(np.asarray, (x, y, train, weights))
    if (arm not in ARMS or x.ndim != 2 or y.shape != (len(x), 2)
            or train.shape != (len(x),) or train.dtype != bool or not train.any() or train.all()
            or weights.shape != train.shape or (weights < 0).any() or weights[train].sum() <= 0
            or not all(np.isfinite(v).all() for v in (x, y, weights))):
        raise ValueError('Finite causal features and separate source-disjoint fitting targets required')
    z, pr = parent.previous.normalized(x, train, weights)
    w = weights[train]/weights[train].sum()
    scale = np.sqrt(np.einsum('n,nc->c', w, y[train]**2)).clip(1e-6)
    target = y[train]/scale
    settings = dict(parent.TREE, n_estimators=32)
    models = []
    for value in ([target] if arm == 'shared32' else [target[:, 0], target[:, 1]]):
        model = ExtraTreesRegressor(**settings)
        model.fit(z[train, 1:], value, sample_weight=w)
        models.append(model)
    return dict(**pr, target_scale=scale, models=models, arm=arm,
        trees=sum(len(m.estimators_) for m in models),
        nodes=sum(t.tree_.node_count for m in models for t in m.estimators_))


def predict(state, x):
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1:] != state['mean'].shape or not np.isfinite(x).all():
        raise ValueError('Inference accepts finite feature vectors only')
    z = np.clip((x-state['mean'])/state['std'], -8, 8)
    outputs = []
    for model in state['models']:
        shape = (len(x), 2) if state['arm'] == 'shared32' else (len(x),)
        out = np.zeros(shape)
        for tree in model.estimators_: out += tree.predict(z)
        outputs.append(out/len(model.estimators_))
    prediction = outputs[0] if state['arm'] == 'shared32' else np.column_stack(outputs)
    return prediction*state['target_scale']


def fingerprint(state):
    out = {k:state[k] for k in ('mean', 'std', 'target_scale')}
    for i, model in enumerate(state['models']):
        f = parent.fingerprint(dict(state, model=model, family='extra_trees'))
        out.update({f'model{i}_{k}':v for k,v in f.items()})
    return out


def summarize(folds):
    keys = ('normalized_MSE', 'total_joint_cost', 'false_safe_cost', 'false_reject_cost')
    means = {arm:{k:float(np.mean([f['arms'][arm]['metrics'][k] for f in folds])) for k in keys}
        for arm in ('shared64', *ARMS)}
    old, new = means['shared64'], means['separate64']
    gain = 1-new['total_joint_cost']/old['total_joint_cost'] if old['total_joint_cost'] > 0 else None
    return dict(source_macro=means, primary_cost_reduction=gain,
        passes_information_screen=bool(gain is not None and gain >= .01
            and new['false_safe_cost'] <= old['false_safe_cost']), safety_certificate=False)
