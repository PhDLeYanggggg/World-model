import numpy as np
import pytest

from src.world_model import m3w_complete_risk_donors as m


def packet():
    paths = np.zeros((3, 7, 12, 2))
    paths[:, :, :, 0] = np.arange(7)[None, :, None]
    return dict(paths=paths, floor=paths[:, 0], cv=paths[:, 1],
                causal=dict(envelope=np.linalg.norm(paths-paths[:, :1], axis=-1).max(-1)))


def test_unknown_contributions_are_positive_bounds_not_observed_zero():
    p = packet(); target = np.ones((3, 12, 2)); valid = np.ones((3, 12), bool)
    valid[1] = False
    y, known = m.signed_targets(p, target, valid, 10., np.ones(3))
    assert known.tolist() == [True, False, True]
    np.testing.assert_array_equal(y[1, :, 0], np.arange(7))
    np.testing.assert_array_equal(y[1, :, 1], np.arange(7))
    target[1] = 1e8
    np.testing.assert_array_equal(y, m.signed_targets(p, target, valid, 10., np.ones(3))[0])


def test_unknown_target_api_has_no_future_argument():
    result = m.unknown_targets(np.array([1., 2.]), np.array([2., 4.]))
    np.testing.assert_array_equal(result, np.ones((2, 2))*.5)
    with pytest.raises(ValueError):
        m.unknown_targets(np.array([1.]), np.array([0.]))


def bank(ids, sites):
    return dict(x=np.ones((len(ids), 120)), target=np.ones((len(ids), 2)),
                sites=np.array(sites), query=np.arange(len(ids)), ids=np.array(ids))


def test_append_preserves_known_features_labels_and_marks_bound_role():
    known = bank([1, 2], ['a', 'b']); bound = bank([3], ['a'])
    complete = m.append_unknown(known, bound)
    for k in ('x', 'target', 'sites', 'query', 'ids'):
        np.testing.assert_array_equal(complete[k][:2], known[k])
    assert complete['observed_future'].tolist() == [True, True, False]
    assert np.isclose(complete['weights'][complete['sites']=='a'].sum(), .5)
    with pytest.raises(ValueError):
        m.append_unknown(known, known)


def test_held_bound_label_does_not_change_fitting_preprocessing():
    own = bank([1, 2, 3, 4], ['a', 'a', 'b', 'b'])
    own['weights'] = np.ones(4)/4
    added = bank([71, 72], ['c', 'c']); added['weights'] = np.ones(2)/2
    a = m.donor.parent.prepare(m.donor.combined(own, added, 'a', 'augmented'), 'a')
    own['target'][:2] = 1e9
    b = m.donor.parent.prepare(m.donor.combined(own, added, 'a', 'augmented'), 'a')
    m.donor.parent.storage.base.exact(a, b)
