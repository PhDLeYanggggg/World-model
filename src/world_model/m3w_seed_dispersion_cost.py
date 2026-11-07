"""Matched-count cost ranking from fixed forecasts, with optional seed dispersion."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def inputs(history, forecasts, seed, crowd):
    """No outcome or validity arrays are accepted by the inference interface."""
    velocity = np.diff(history, axis=1)
    last = velocity[:, -1]
    speed = np.linalg.norm(velocity, axis=-1)
    scale = np.maximum(1., 6.5 * speed[:, -1])
    direction = last / np.maximum(speed[:, -1, None], 1e-8)
    tangent = np.stack((-direction[:, 1], direction[:, 0]), axis=-1)
    floor = np.arange(1, 13)[None, :, None] * last[:, None]
    delta = (forecasts[seed] - floor) / scale[:, None, None]
    past_acc = np.diff(velocity, axis=1)
    path = history[:, -1] - history[:, 0]
    common = np.column_stack((np.log1p(speed),
        np.log1p(np.linalg.norm(past_acc, axis=-1)),
        np.linalg.norm(path, axis=-1) / np.maximum(speed.sum(1), 1e-8),
        np.log1p(crowd), np.linalg.norm(delta, axis=-1)[:, [2, 5, 11]],
        (delta * direction[:, None]).sum(-1)[:, [2, 5, 11]],
        (delta * tangent[:, None]).sum(-1)[:, [2, 5, 11]]))
    centered = (forecasts - forecasts.mean(0)) / scale[None, :, None, None]
    spread = np.sqrt(np.square(centered).sum(-1).mean(0))
    extra = np.column_stack((spread[:, [2, 5, 11]], spread.mean(1)))
    return common, extra, floor, scale


def targets(floor, forecast, future, valid, scale):
    count = valid.sum(1)
    diff = (np.linalg.norm(np.where(valid[..., None], floor-future, 0.), axis=-1)
            - np.linalg.norm(np.where(valid[..., None], forecast-future, 0.), axis=-1))
    gain = np.divide(diff.sum(1), count*scale, out=np.full(len(count), np.nan), where=count > 0)
    return np.column_stack((gain, np.maximum(-gain, 0.)))


def fit(features, costs, weights):
    known = np.isfinite(costs).all(1)
    weight = weights[known] / weights[known].mean()
    return tuple(HistGradientBoostingRegressor(max_iter=64, max_leaf_nodes=15, max_bins=63,
        l2_regularization=1., early_stopping=False, random_state=20261107).fit(
        features[known], target[known], sample_weight=weight) for target in costs.T)


def predict(models, features, scale):
    gain, harm = (m.predict(features) * scale for m in models)
    return np.column_stack((gain, np.maximum(harm, 0.)))


def matched_choices(query, ids, predictions):
    score = predictions[:, 0] - predictions[:, 1]
    order = np.lexsort((ids, -score, query))
    groups = np.split(order, np.flatnonzero(query[order][1:] != query[order][:-1])+1)
    selected = np.zeros(len(query), dtype=bool)
    for group in groups:
        selected[group[:min(2, len(group)-1)]] = True
    return selected
