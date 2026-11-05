import copy
import numpy as np
import pytest
from src.world_model import m3w_candidate_peer_paths as features
from src.evaluation import m3w_peer_path_probe as method
from tests.test_m3w_candidate_neighbor_probe import packet


def test_directional_profiles_and_future_label_free_schema():
    x = packet(); out = features.basis(x)
    assert out.shape == (2, 7, 60) and len(features.NAMES) == 60
    np.testing.assert_allclose(out[0, 0, :24].reshape(12, 2), np.column_stack((6-np.arange(1, 13), np.zeros(12))))
    np.testing.assert_allclose(out[0, 0, 24:48].reshape(12, 2), np.column_stack((np.full(12, 6), -np.arange(1, 13))))
    np.testing.assert_allclose(out[0, 0, 48:], np.sqrt(36+np.arange(1, 13)**2)-abs(6-np.arange(1, 13)), atol=1e-6)
    np.testing.assert_array_equal(features.chunked(x, 1), out)
    x[..., 354:] = 9999
    np.testing.assert_array_equal(features.basis(x), out)
    with pytest.raises(ValueError):
        features.basis(x[..., :-1])


def test_masks_timestamp_alignment_stationary_and_ties():
    x = packet(); x[..., 224:288] = 0; x[..., 32:224] = 1e10
    np.testing.assert_array_equal(features.basis(x), np.zeros((2, 7, 60)))
    x = packet(); x[..., 32:224].reshape(2, 7, 8, 8, 3)[..., 0, -1, 2] = 99
    assert not features.basis(x).any()
    x = packet(); x[..., :24].reshape(2, 7, 8, 3)[..., :2] = 0
    assert np.isfinite(features.basis(x)).all()
    x[..., 306:354] = 0
    n = x[..., 32:224].reshape(2, 7, 8, 8, 3); n[..., 1, :, :] = n[..., 0, :, :]
    n[..., 1, :, 0] = -6
    x[..., 224:288].reshape(2, 7, 8, 8)[..., 1, :] = 1
    np.testing.assert_allclose(features.basis(x), 0)
    x[..., 224] = .5
    with pytest.raises(ValueError):
        features.basis(x)


def test_rotation_translation_neighbor_permutation_and_last_nonzero_heading():
    x = packet(); expected = features.basis(x); rotation = np.array([[0., 1.], [-1., 0.]])
    for start, stop, shape in ((0, 24, (2, 7, 8, 3)), (32, 224, (2, 7, 8, 8, 3))):
        p = x[..., start:stop].reshape(shape); p[..., :2] = p[..., :2]@rotation+35
    for start in (306, 330):
        p = x[..., start:start+24].reshape(2, 7, 12, 2); p[:] = p@rotation
    n = x[..., 32:224].reshape(2, 7, 8, 8, 3); n[:] = n[..., ::-1, :, :].copy()
    m = x[..., 224:288].reshape(2, 7, 8, 8); m[:] = m[..., ::-1, :].copy()
    np.testing.assert_allclose(features.basis(x), expected, atol=1e-6)
    x = packet(); h = x[..., :24].reshape(2, 7, 8, 3); h[..., -2, :2] = h[..., -1, :2]
    np.testing.assert_allclose(features.basis(x), expected, atol=1e-6)


def test_misalignment_query_boundary_row_order_and_complete_coverage():
    x = np.arange(5*7*60).reshape(5, 7, 60).astype(float); ids = np.array(['c', 'a', 'b', 'e', 'd'])
    groups = [np.array([0, 1, 2]), np.array([3]), np.array([4])]
    shifted, donors = features.misaligned(x, groups, ids)
    assert donors.tolist() == [2, 0, 1, 3, 4]
    np.testing.assert_array_equal(shifted, x[donors])
    order = np.array([2, 0, 1, 4, 3]); inverse = np.argsort(order)
    other, _ = features.misaligned(x[order], [inverse[g] for g in groups], ids[order])
    np.testing.assert_array_equal(other[inverse], shifted)
    for bad in (groups[:-1], groups+[np.array([0])]):
        with pytest.raises(ValueError):
            features.misaligned(x, bad, ids)


def test_both_models_fit_without_held_labels_or_statistics():
    rng = np.random.default_rng(7); x = rng.normal(size=(160, 6)); y = rng.normal(size=(160, 2))
    train = np.arange(160) < 120; held = ~train; weights = np.ones(160)
    for family in method.FAMILIES:
        pred, a = method.fit_predict(x, y, train, held, weights, family)
        xx, yy = x.copy(), y.copy(); xx[held] *= 1000; yy[held] = 999999
        _, b = method.fit_predict(xx, yy, train, held, weights, family)
        for key, value in method.fingerprint(a).items():
            np.testing.assert_array_equal(value, method.fingerprint(b)[key])
        np.testing.assert_array_equal(pred, method.predict(b, x[held]))


def test_primary_screen_requires_correct_association_and_harm_control():
    base = dict(normalized_MSE=1., total_joint_cost=1., false_safe_cost=1., false_reject_cost=1.)
    rows = [dict(held_source=k, families={f:{a:dict(base, normalized_MSE=.8 if a == 'aligned180' else 1.)
        for a in method.ARMS} for f in method.FAMILIES}) for k in ('a', 'b')]
    assert method.summarize(rows)['extra_trees']['passes_internal_screen']
    changed = copy.deepcopy(rows)
    for r in changed:
        r['families']['extra_trees']['misaligned180']['normalized_MSE'] = .7
    assert not method.summarize(changed)['extra_trees']['passes_internal_screen']
    changed = copy.deepcopy(rows)
    for r in changed:
        r['families']['extra_trees']['aligned180']['false_safe_cost'] = 1.01
    assert not method.summarize(changed)['extra_trees']['passes_internal_screen']


def test_nonlinear_probe_can_learn_known_synthetic_relationship():
    rng = np.random.default_rng(17); x = rng.normal(size=(2000, 4))
    y = np.column_stack((x[:, 0]**2, (x[:, 1] > 0).astype(float)))
    train = np.arange(len(x)) < 1500; held = ~train
    a, _ = method.fit_predict(x, y, train, held, np.ones(len(x)), 'ridge')
    b, _ = method.fit_predict(x, y, train, held, np.ones(len(x)), 'extra_trees')
    assert np.mean((b-y[held])**2) < .8*np.mean((a-y[held])**2)
