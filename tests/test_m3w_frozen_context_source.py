import numpy as np
import pytest

from src.world_model.m3w_frozen_context_source import original_context


def packets():
    metadata = {k: np.arange(3) for k in ('sites', 'recordings', 'frames', 'agents', 'history', 'origin')}
    g = np.zeros((3, 476), np.float32)
    p = dict(metadata, geometry=g)
    c = dict(metadata, geometry=g.copy(), detector_history=np.ones((3, 8)))
    c['geometry'][:, 38:294] = 1
    return c, p


def test_uses_original_neighbor_packet_without_replacing_controller_rows():
    c, p = packets()
    out = original_context(c, p)
    assert out['geometry'] is p['geometry']
    assert out['detector_history'] is c['detector_history']
    np.testing.assert_array_equal(c['geometry'][:, 38:294], 1)
    assert set(out) == set(c)


def test_row_ego_and_future_input_mismatches_are_rejected():
    c, p = packets()
    p['frames'] = np.arange(3) + 1
    with pytest.raises(AssertionError):
        original_context(c, p)
    c, p = packets()
    p['geometry'][0, 0] = 2
    with pytest.raises(AssertionError):
        original_context(c, p)
    c, p = packets()
    with pytest.raises(ValueError):
        original_context(c, dict(p, future_endpoint=np.zeros((3, 2))))
