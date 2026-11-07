"""Past-only native 8/12 inputs shared by causal and frozen Torch predictors."""
from dataclasses import dataclass

import numpy as np

from src.data_unification.m3w_dronecrowd_frozen_inputs import adapt_past
from src.world_model.m3w_european_source_forecast import BASELINES, baseline_numpy, pack_scene
from src.world_model.m3w_native_forecast import predict


@dataclass(frozen=True)
class PrefixBatch:
    agent_id: np.ndarray
    origin: np.ndarray
    geometry: np.ndarray
    baselines: dict


def prepare_prefix(raw):
    inputs = adapt_past(raw)
    geometry, target = pack_scene(inputs)
    history = inputs['history_xy'][target]
    origin = history[:, -1]
    baselines = {key: baseline_numpy(history, i)-origin[:, None]
                 for i, key in enumerate(BASELINES)}
    np.testing.assert_array_equal(baselines[BASELINES[1]],
                                  inputs['baseline_cv'][target]-origin[:, None])
    return PrefixBatch(inputs['agent_id'][target], origin, geometry, baselines)


def predict_relative(model, geometry, *, batch_size):
    if geometry.ndim != 2 or geometry.shape[1] != 476 or batch_size < 1:
        raise ValueError('Expected 476-column geometry and positive batch size')
    if not len(geometry):
        return np.empty((0, 12, 2), dtype=np.float32)
    y = predict(model, {'geometry': geometry}, np.arange(len(geometry)), batch_size)
    if y.shape != (len(geometry), 12, 2) or not np.isfinite(y).all():
        raise ValueError('Predictor must return finite relative 12-step paths')
    return y
