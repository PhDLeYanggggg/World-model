"""Structured ADE gain estimates for fixed candidate paths, not trajectory generation."""
import numpy as np


def local_axes(history):
    delta = np.diff(history, axis=1)
    recent = delta[:, -3:].sum(1)
    for j in range(delta.shape[1] - 1, -1, -1):
        recent = np.where((np.linalg.norm(recent, axis=-1) > 1e-8)[:, None],
                          recent, delta[:, j])
    norm = np.linalg.norm(recent, axis=-1, keepdims=True)
    along = np.where(norm > 1e-8, recent / norm.clip(1e-8), [1., 0.])
    across = np.stack((-along[:, 1], along[:, 0]), axis=-1)
    return np.stack((along, across), axis=-1)


def features(history, scale, crowd, bank, floor):
    axes = local_axes(history)
    history_local = (history[:, :-1] - history[:, -1:]) @ axes
    history_local /= scale[:, None, None]
    z = np.column_stack((np.arcsinh(history_local.reshape(len(scale), -1)),
                         np.log1p(scale), np.log1p(crowd)))
    direction_features = np.column_stack((np.ones(len(z)), z, np.tanh(z)))
    change = np.einsum('natd,ndc->natc', bank - floor[:, None], axes)
    return direction_features, change, axes


def targets(floor, target, valid, axes, change, scale):
    """A valid norm subgradient at zero error is zero."""
    e = np.where(valid[..., None], target - floor, 0.)
    norm = np.linalg.norm(e, axis=-1)
    direction = (e @ axes) / np.where(norm > 0., norm, 1.)[..., None]
    linear = np.einsum('ntd,natd->nat', direction, change)
    gain = norm[:, None] - np.linalg.norm(change - (e @ axes)[:, None], axis=-1)
    remainder = np.where(valid[:, None], (linear - gain).clip(0.), 0.)
    curvature = remainder.sum(axis=-1) / (valid.sum(1).clip(1) * scale)[:, None]
    return direction, curvature


def moments(z, action_x, direction, curvature, valid, crowd):
    row = valid.any(1) / crowd
    if not row.any():
        raise ValueError('Require observed fitting labels')
    row /= row.sum()
    step = valid / (valid.sum(1).clip(1) * crowd)[:, None]
    step /= step.sum(0, keepdims=True).clip(1e-12)
    direction_moments = (np.stack([z.T @ (w[:, None] * z) for w in step.T]),
                         np.stack([z.T @ (w[:, None] * t)
                                   for w, t in zip(step.T, direction.swapaxes(0, 1))]))
    by_action = action_x.swapaxes(0, 1)
    curvature_moments = (np.stack([x.T @ (row[:, None] * x) for x in by_action]),
                         np.stack([x.T @ (row * t) for x, t in
                                   zip(by_action, curvature.T)])[..., None])
    return direction_moments, curvature_moments


def fit(sources):
    weights = []
    for component in zip(*sources):
        gram, rhs = (np.mean(tensors, axis=0) for tensors in zip(*component))
        weights.append(np.linalg.solve(gram + .002 * np.eye(gram.shape[-1]), rhs))
    return weights


def predict(z, change, action_x, scale, weights):
    direction = np.einsum('nd,tdk->ntk', z, weights[0])
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True).clip(1.)
    tangent = np.einsum('ntd,natd->nat', direction, change).mean(-1)
    curvature = np.einsum('nad,adk->nak', action_x, weights[1])[..., 0].clip(0.)
    return tangent, tangent - scale[:, None] * curvature


def coherent_value(gain, action_x, original, scale, change):
    """Use the same predicted harm and missingness envelope as the control."""
    radius = np.linalg.norm(change, axis=-1).max(2)
    p = np.einsum('nad,adc->nac', action_x, original)
    missing = p[..., 5].clip(0., 1.) * radius
    harm = np.minimum(p[..., 1].clip(0.) * scale[:, None], radius - missing)
    return (np.clip(gain, -harm, radius - missing - harm) - missing) / scale[:, None]
