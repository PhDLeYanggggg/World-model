import numpy as np
import pytest

from src.evaluation.m3w_external_step_budget import VARIANTS, build_pair, score_pair


def packet(collapsed=False):
    cv = np.zeros((7, 12, 2))
    off = np.array([0, 3, 7])
    a, b = np.zeros((1, 2, 12, 2)), np.zeros((1, 2, 12, 2))
    a[..., 0], b[..., 1] = np.linspace(.3, 1., 12), np.linspace(1., .1, 12)
    if collapsed:
        a[:, :, 6:] = 0.
        b[:, :, :6] = 0.
        a[:, :, 0, 0] = b[:, :, 6, 1] = 1.
    z = dict(independent_rows=np.array([[3, 4]]), joint_rows=np.array([[4, 6]]),
             independent_paths=a, joint_paths=b, radius=np.array([1.]))
    return cv, off, np.array([1]), z


def test_matching_preserves_directions_recipients_and_the_score_roster():
    cv, off, q, z = packet()
    pair = build_pair(cv, off, q, z, 'independent')
    np.testing.assert_array_equal(pair['rows'], [[[3, 4], [4, 6]]])
    p = pair['path']['matched_steps']
    np.testing.assert_allclose(p[0, 0, :, :, 0], np.tile(pair['profiles'], (2, 1)))
    np.testing.assert_allclose(p[0, 1, :, :, 1], np.tile(pair['profiles'], (2, 1)))
    assert (p[0, 0, :, :, 1] == 0.).all() and (p[0, 1, :, :, 0] == 0.).all()
    y, valid = cv+.17, np.ones((7, 12), bool)
    valid[4, -3:] = False
    base = score_pair(cv, off, pair, y, valid)
    assert set(base) == set(VARIANTS)
    for v in base.values():
        assert v['lower_pp'] <= v['observed_pp'] <= v['upper_pp']
    y[:3] += 50.
    assert score_pair(cv, off, pair, y, valid) == base


def test_missing_shared_motion_is_unestimable_not_a_zero_gain():
    cv, off, q, z = packet(collapsed=True)
    pair = build_pair(cv, off, q, z, 'independent')
    assert pair['collapsed'] == 1 and not len(pair['queries'])
    assert score_pair(cv, off, pair, cv, np.ones(cv.shape[:2], bool)) is None


def test_cross_query_edits_are_rejected():
    cv, off, q, z = packet()
    z['joint_rows'][0, 0] = 2
    with pytest.raises(ValueError, match='Unbound'):
        build_pair(cv, off, q, z, 'independent')
