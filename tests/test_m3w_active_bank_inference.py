import numpy as np
import pytest
from src.world_model import m3w_active_bank_cost as api
from tests.test_m3w_active_bank_cost import packets_and_data


def test_fresh_torch_lazy_bank_predictions_match_materialized_inputs_without_future_fields(tmp_path):
    original, old, new = packets_and_data()
    data = api.frozen_training_data(original, old, new)
    settings = dict(width=64, steps=6, batch_size=8, learning_rate=.001, underharm_weight=4.,
                    ranking_weight=.5, gradient_clip=5., heartbeat_every=2, checkpoint_every=2)
    state = api.fit(data, variant='new_neural', settings=settings, seed=17, identity={}, path=tmp_path/'unit.gz')
    for packet in new.values():
        extra = np.zeros((len(packet['ids']), 64), np.float32)
        args = (state, packet['x'], packet['scene'], extra, packet['envelope'], packet['scales'])
        lazy = api.detector.predict(*args)
        materialized = api.detector.predict(state, np.asarray(packet['x']), np.asarray(packet['scene']),
                                           extra, packet['envelope'], packet['scales'])
        np.testing.assert_array_equal(lazy, materialized)
        with pytest.raises(TypeError):
            api.detector.predict(*args, future_endpoint=np.ones((len(extra), 2)))
