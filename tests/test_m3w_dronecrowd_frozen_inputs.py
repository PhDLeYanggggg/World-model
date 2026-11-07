from dataclasses import replace

import numpy as np
import pytest

from src.data_unification.m3w_dronecrowd_frozen_inputs import adapt_past
from src.evaluation.m3w_dronecrowd_windows import PROBES, Recording, past_inputs
from src.world_model.m3w_european_source_forecast import pack_scene


def test_future_perturbation_preserves_packed_geometry_and_causal_roster():
    x = np.arange(30, dtype=float); positions = np.tile(np.stack((x, x*0), axis=1), (3, 1, 1))
    positions[1, :, 1] = 2.; valid = np.ones((3, 30), bool); valid[1, 5] = False; valid[2, :13] = False
    r = Recording('synthetic', 'audit', np.arange(3), positions, valid, valid.copy())
    query = 12
    a = adapt_past(past_inputs(r, query, PROBES[0]))
    p = positions.copy(); p[:, query+1:] += 10_000
    v = valid.copy(); v[:, query+1:] = False
    b = adapt_past(past_inputs(replace(r, positions=p, valid=v), query, PROBES[0]))
    np.testing.assert_array_equal(a['agent_id'], [0, 1])
    np.testing.assert_array_equal(a['target_eligible'], [True, False])
    np.testing.assert_array_equal(a['baseline_cv'][0, :, 0], np.arange(13, 25))
    for left, right in zip(pack_scene(a), pack_scene(b)):
        np.testing.assert_array_equal(left, right)


def test_future_keys_are_rejected_before_packing():
    with pytest.raises(ValueError, match='observed annotation prefix'):
        adapt_past({'future_mask': np.ones(12, bool)})
