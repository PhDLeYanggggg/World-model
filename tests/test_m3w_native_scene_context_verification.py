import numpy as np
import pytest
from src.data_unification.m3w_native_scene_context import SourceSceneContext
from src.evaluation.m3w_native_scene_alignment import scene_index, resolve_target_neighbors


def test_positional_unique_match_can_still_be_wrong():
    # A non-target identity is coincident with a target. Position-only matching
    # looks unique within the *target subset*, but silently substitutes identity.
    rows = np.array([[0, -1, -1, 1, 1, 84, 0, 0, 1],
                     [1, 4, -1, 6, 1, 84, 0, 0, 1],
                     [2, 4, -1, 6, 1, 84, 0, 0, 1]], float)
    source = SourceSceneContext(rows, np.array(['Pedestrian']*3))
    keys = np.array([[84, 0], [84, 2]])
    g = np.zeros((2, 476), np.float32); g[:, 300] = 2
    origins = np.array([[0., 0.], [5., 0.]])
    for i, others in enumerate(([1, 2], [1, 0])):
        for slot, other in enumerate(others):
            g[i, 38:166].reshape(8, 8, 2)[slot, -1] = source.xy[other]-origins[i]
            g[i, 230:294].reshape(8, 8)[slot, -1] = 1
    rotation = np.tile(np.eye(2), (2, 1, 1))
    index = scene_index(np.array(['a', 'a']), keys[:, 0], keys[:, 1])
    old = resolve_target_neighbors(g, origins, rotation, np.ones(2), index)
    assert old['target_row'][0, 0] == 1
    new = source.build(keys, origins, rotation, np.ones(2), g, np.array([0, 1]))
    assert new['neighbor_agent_ids'][0, 0] == 1
    assert new['neighbor_target_rows'][0, 0] == -1
    assert not new['context_cv_valid'].any()
    assert new['context_history_mask'].sum() == 3


def test_false_future_neighbor_timestamp_rejected():
    rows = np.array([[0, -1, -1, 1, 1, 84, 0, 0, 1], [1, 4, -1, 6, 1, 84, 0, 0, 1]], float)
    source = SourceSceneContext(rows, np.array(['Pedestrian']*2))
    g = np.zeros((1, 476), np.float32); g[0, 300] = 1
    g[0, 52:54] = [5, 0]; g[0, 237] = 1; g[0, 173] = 1/12
    with pytest.raises(AssertionError):
        source.build(np.array([[84, 0]]), np.array([[0., 0.]]), np.eye(2)[None], np.ones(1), g, np.array([0]))
