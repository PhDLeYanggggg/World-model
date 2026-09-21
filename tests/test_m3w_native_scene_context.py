import numpy as np
import pytest

from src.data_unification.m3w_native_scene_context import SourceSceneContext


def source():
    rows = []
    for agent in (0, 1, 2):
        for t in range(12):
            xy = np.array([t if agent == 0 else 5, agent])
            rows.append([agent, *(xy-1), *(xy+1), t*12, 0, 0, 1])
    return np.asarray(rows, float), np.array(['Pedestrian']*24+['Biker']*12)


def test_same_position_different_identity_and_future_independence():
    rows, labels = source()
    a = SourceSceneContext(rows, labels)
    ids = a.at_frame(84)
    xy, times, mask = a.history(ids)
    assert mask.all() and times.max() == 0
    changed = rows.copy(); changed[changed[:, 5] > 84, 1:5] += 100000
    b = SourceSceneContext(changed, labels)
    c = SourceSceneContext(rows[rows[:, 5] <= 84], labels[rows[:, 5] <= 84])
    for other in (b, c):
        j = other.at_frame(84)
        np.testing.assert_array_equal(a.agent[ids], other.agent[j])
        for before, after in zip((xy, times, mask), other.history(j)):
            np.testing.assert_array_equal(before, after)


def test_short_history_masks_and_irregular_times():
    rows, labels = source(); keep = (rows[:, 0] == 0) | ((rows[:, 0] == 1) & np.isin(rows[:, 5], [0, 84]))
    a = SourceSceneContext(rows[keep], labels[keep]); ids = a.at_frame(84)
    _, t, m = a.history(ids)
    np.testing.assert_array_equal(m.sum(1), [8, 2])
    np.testing.assert_array_equal(t[1, -2:], [-84, 0])


def test_duplicate_rejected():
    rows, labels = source()
    with pytest.raises(ValueError, match='Duplicate'):
        SourceSceneContext(np.concatenate((rows, rows[:1])), np.r_[labels, labels[:1]])


def test_build_explicit_neighbors_and_short_context():
    rows, labels = source()
    # Both other agents share a current position; ID order breaks distance ties.
    rows[(rows[:, 0] == 2) & (rows[:, 5] == 84), 2:5:2] -= 1
    a = SourceSceneContext(rows, labels)
    current = a.at_frame(84); pos = a.xy[current[0:1]]
    g = np.zeros((1, 476), np.float32); g[0, 300] = 2
    h, t, m = a.history(current[1:])
    g[0, 38:166].reshape(8, 8, 2)[:2] = h-pos[:, None]
    g[0, 166:230].reshape(8, 8)[:2] = t/144
    g[0, 230:294].reshape(8, 8)[:2] = m
    out = a.build(np.array([[84, 0]]), pos, np.eye(2)[None], np.ones(1), g, np.array([13]))
    np.testing.assert_array_equal(out['neighbor_agent_ids'][0, :2], [1, 2])
    assert (out['neighbor_target_rows'] == -1).all()
    np.testing.assert_array_equal(out['context_target_rows'], [13, -1, -1])
    assert out['context_cv_valid'].all()
