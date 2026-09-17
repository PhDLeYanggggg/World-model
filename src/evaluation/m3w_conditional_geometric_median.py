"""Training-label conditional distributions and deterministic ADE point forecasts."""
from __future__ import annotations

import numpy as np


def forest_training_weights(forest, train_features, query_features):
    """Leaf-uniform weights for an unbootstrapped, unweighted fitted forest.

    Only causal query features are accepted; query labels have no role here.
    The caller must separately verify original training membership and fitting.
    """
    train, query = np.asarray(train_features), np.asarray(query_features)
    if (train.ndim!=2 or query.ndim!=2 or train.shape[1]!=query.shape[1]
            or not len(train) or not np.isfinite(train).all() or not np.isfinite(query).all()):
        raise ValueError('Finite aligned train/query features required')
    if forest.bootstrap or forest.get_params().get('max_samples') is not None:
        raise ValueError('Bootstrap multiplicities are not supported by this exact replay')
    weights = np.zeros((len(query), len(train)))
    for tree in forest.estimators_:
        if (tree.tree_.n_node_samples[0]!=len(train)
                or not np.allclose(tree.tree_.weighted_n_node_samples, tree.tree_.n_node_samples)):
            raise ValueError('Training count or weighting mismatch')
        train_leaf, query_leaf = tree.apply(train), tree.apply(query)
        for leaf in np.unique(query_leaf):
            members, queries = np.flatnonzero(train_leaf==leaf), np.flatnonzero(query_leaf==leaf)
            if len(members)!=tree.tree_.n_node_samples[leaf]:
                raise ValueError('Original training membership is missing')
            weights[np.ix_(queries, members)] += 1./len(members)
    weights /= len(forest.estimators_)
    np.testing.assert_allclose(weights.sum(1), 1., rtol=0, atol=1e-12)
    return weights


def geometric_median(points, weights, *, tolerance=1e-8, max_iterations=5000):
    """Modified Weiszfeld updates, preserving the atom at an exact zero forecast.

    Minimize a weighted sum of Euclidean distances, not coordinatewise L1 or MSE.
    A subgradient/constrained-hull bound certifies the numerical objective gap.
    Iteration-limit results remain explicit approximations, never fake convergence.
    """
    y, w = np.asarray(points, float), np.asarray(weights, float)
    if (y.ndim!=2 or y.shape[1]!=2 or w.shape!=(len(y),) or not len(y)
            or not np.isfinite(y).all() or not np.isfinite(w).all() or np.any(w<0)
            or w.sum()<=0 or not np.isfinite(tolerance) or tolerance<=0 or max_iterations<1):
        raise ValueError('Finite two-dimensional support and nonnegative weights required')
    active = w>0
    y, w = y[active], w[active]/w[active].sum()
    support, inverse = np.unique(y, axis=0, return_inverse=True)
    w = np.bincount(inverse, weights=w, minlength=len(support))
    scale = float(np.linalg.norm(support, axis=1).max())
    if scale==0:
        return np.zeros(2), {'converged': True, 'iterations': 0, 'objective': 0., 'gap_bound': 0., 'zero_optimal': True}
    y = support/scale
    x = np.zeros(2)
    for iteration in range(max_iterations+1):
        offset = y-x
        distance = np.linalg.norm(offset, axis=1)
        # Snap only at machine-scale coincidence; certify after snapping.
        closest = int(np.argmin(distance))
        if 0<distance[closest]<=1e-14:
            x = y[closest].copy()
            offset, distance = y-x, np.linalg.norm(y-x, axis=1)
        coincident = distance==0
        mass = float(w[coincident].sum())
        inv = np.divide(w, distance, out=np.zeros_like(w), where=~coincident)
        residual = (offset*inv[:, None]).sum(0)
        norm = float(np.linalg.norm(residual))
        gap = float(max(norm-mass, 0.)*distance.max())
        objective = float(w @ distance)
        converged = gap<=tolerance
        if converged or iteration==max_iterations:
            return x*scale, {'converged': converged, 'iterations': iteration,
                'objective': objective*scale, 'gap_bound': gap*scale,
                'zero_optimal': bool(converged and np.all(x==0))}
        step = (y*inv[:, None]).sum(0)/inv.sum()
        beta = min(1., mass/norm) if norm>0 else 1.
        x = beta*x+(1-beta)*step
    raise AssertionError('Unreachable solver state')


def conditional_ade_prediction(train_targets, query_weights, **solver_options):
    targets, weights = np.asarray(train_targets, float), np.asarray(query_weights, float)
    if (targets.ndim!=3 or targets.shape[1:]!=(12,2) or weights.ndim!=2
            or weights.shape[1]!=len(targets)):
        raise ValueError('Twelve-step training labels and aligned query weights required')
    predictions, diagnostics = [], []
    for row_weights in weights:
        path, per_step = [], []
        for step in range(12):
            value, info = geometric_median(targets[:, step], row_weights, **solver_options)
            path.append(value); per_step.append(info)
        predictions.append(path); diagnostics.append(per_step)
    return np.asarray(predictions), diagnostics
