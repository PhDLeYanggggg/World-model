"""Change fitted partitions while retaining original two-channel risk means."""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from src.evaluation import m3w_peer_path_probe as parent
from src.evaluation.m3w_decision_semantics import joint_target

ARMS = ('mse_partition', 'sign_partition', 'cost_partition')


def leaf_moments(leaves, y, weights, size):
    y, weights, leaves = np.asarray(y), np.asarray(weights), np.asarray(leaves)
    if (y.shape != (len(leaves), 2) or weights.shape != leaves.shape
            or not all(np.isfinite(v).all() for v in (y, weights)) or (weights < 0).any()
            or weights.sum() <= 0 or (leaves < 0).any() or (leaves >= size).any()):
        raise ValueError('Finite original weighted two-channel fitting labels required')
    mass = np.bincount(leaves, weights=weights, minlength=size)
    out = dict(mass=mass)
    for name, values in (('mean', y), ('positive', np.maximum(y, 0)), ('negative', np.maximum(-y, 0)),
            ('positive_probability', (y > 0).astype(float)), ('negative_probability', (y < 0).astype(float))):
        sums = np.stack([np.bincount(leaves, weights=weights*values[:, j], minlength=size) for j in range(2)], -1)
        out[name] = np.divide(sums, mass[:, None], out=np.zeros_like(sums), where=mass[:, None] > 0)
    for sign in ('positive', 'negative'):
        p = out[sign+'_probability']
        out[sign+'_severity'] = np.divide(out[sign], p, out=np.zeros_like(p), where=p > 0)
    return out


def normalize(state, x):
    return np.clip((x-state['mean'])/state['std'], -8, 8)


def fit(x, y, train, weights, arm):
    if (arm not in ARMS[1:] or x.ndim != 2 or y.shape != (len(x), 2)
            or train.shape != (len(x),) or train.dtype != bool or not train.any()
            or weights.shape != train.shape or not all(np.isfinite(v).all() for v in (x, y, weights))):
        raise ValueError('Fixed fitted partition arm and separate finite supervision required')
    z, pr = parent.previous.normalized(x, train, weights)
    w = weights[train]/weights[train].sum(); target = y[train]
    scale = np.sqrt(np.einsum('n,nc->c', w, target**2)).clip(1e-6)
    joint = joint_target(target); cost = np.abs(joint)
    split_weight = w if arm == 'sign_partition' else w*cost
    if split_weight.sum() <= 0:
        raise ValueError('No fitting decision-cost mass')
    model = ExtraTreesClassifier(**parent.TREE, criterion='gini', class_weight=None)
    model.fit(z[train, 1:], (joint > 0).astype(int), sample_weight=split_weight/split_weight.sum())
    # The changed fitting measure selects splits only, never the final risk mean.
    tables = [leaf_moments(tree.apply(z[train, 1:]), target, w, tree.tree_.node_count)
        for tree in model.estimators_]
    return dict(**pr, target_scale=scale, model=model, tables=tables, arm=arm,
        fit_class_counts=np.bincount((joint > 0).astype(int), minlength=2),
        fit_cost_mass=float(w@cost), split_weight_sum=float(split_weight.sum()),
        original_mean_readout=True, probability_used_for_acceptance=False)


def predict(state, x):
    if state['arm'] == 'mse_partition':
        return parent.predict(state, x)
    z = normalize(state, x); out = np.zeros((len(z), 2))
    for tree, table in zip(state['model'].estimators_, state['tables']):
        leaf = tree.apply(z)
        if (table['mass'][leaf] <= 0).any():
            raise ValueError('A predicted leaf has no fitting mass')
        out += table['mean'][leaf]
    return out/len(state['model'].estimators_)


def fingerprint(state):
    out = parent.fingerprint(dict(state, family='extra_trees'))
    for i, table in enumerate(state.get('tables', [])):
        for k, v in table.items():
            out[f'table{i}_{k}'] = v
    return out


def decomposition(state, x, y, train, weights):
    z = normalize(state, x); w = weights[train]/weights[train].sum()
    direct = np.zeros((len(x), 2)); factored = np.zeros_like(direct)
    for tree in state['model'].estimators_:
        table = leaf_moments(tree.apply(z[train]), y[train], w, tree.tree_.node_count)
        leaf = tree.apply(z)
        direct += table['mean'][leaf]
        factored += (table['positive_probability']*table['positive_severity']-
            table['negative_probability']*table['negative_severity'])[leaf]
    direct /= len(state['model'].estimators_); factored /= len(state['model'].estimators_)
    original = predict(state, x)
    np.testing.assert_allclose(direct, original, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(factored, direct, rtol=1e-10, atol=1e-12)
    return dict(same_partition_factorization_max_abs_difference=float(np.max(np.abs(factored-direct))),
        original_leaf_mean_reconstruction_max_abs_difference=float(np.max(np.abs(direct-original))),
        direct_vs_factorized_joint_sign_changes=int(((direct.max(-1) <= 0) != (factored.max(-1) <= 0)).sum()),
        original_vs_reconstructed_joint_sign_changes=int(((direct.max(-1) <= 0) != (original.max(-1) <= 0)).sum()),
        arithmetic_check_only=True, original_predictions_unchanged=True)


def summary(folds):
    keys = ('normalized_MSE', 'total_joint_cost', 'false_safe_cost', 'false_reject_cost')
    means = {a:{k:float(np.mean([f['arms'][a]['metrics'][k] for f in folds])) for k in keys} for a in ARMS}
    control, sign, cost = (means[a] for a in ARMS)
    gain = 1-cost['total_joint_cost']/control['total_joint_cost'] if control['total_joint_cost'] > 0 else None
    return dict(source_macro=means, primary_joint_cost_reduction=gain,
        passes_internal_screen=bool(gain is not None and gain >= .01
            and cost['false_safe_cost'] <= control['false_safe_cost']
            and cost['total_joint_cost'] <= sign['total_joint_cost']),
        safety_certificate=False)
