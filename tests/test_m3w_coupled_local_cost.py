import inspect

import numpy as np
import pytest

from src.world_model import m3w_coupled_local_cost as model


def labels(n):
    value = np.zeros((n, 7, 5)); value[..., 0] = .2; value[..., 1] = .3
    value[..., 2] = 2.; value[..., 3] = .5; value[..., 4] = .1
    return value


def test_finite_completion_keeps_original_labels_unknown_and_coupled():
    y = labels(4); y[1] = np.nan; original = y.copy()
    target, known = model.encode(y, np.ones((4, 7)), np.ones(4))
    np.testing.assert_equal(y, original)
    assert known.tolist() == [True, False, True, True]
    np.testing.assert_array_equal(target[1, :, 1], 1.)
    np.testing.assert_array_equal(target[1, :, 2:4], 0.)
    with pytest.raises(ValueError):
        broken = y.copy(); broken[0, 0, 4] = .9
        model.encode(broken, np.ones((4, 7)), np.ones(4))


def test_common_reference_convex_estimator_masks_unknowns_not_query_features():
    y = labels(4); y[1] = np.nan
    target, known = model.encode(y, np.ones((4, 7)), np.ones(4))
    args = target[None], known[None], np.ones((1, 4, 7), bool), np.ones((1, 4)), np.full((1, 7), 2.), np.full(1, 3.)
    a, yes = model.estimate(*args, bounds=False); b, _ = model.estimate(*args, bounds=True)
    assert yes.all()
    np.testing.assert_allclose(a[..., 2], 6.)
    np.testing.assert_allclose(b[..., 2], 4.5)
    assert (b[..., 4] <= b[..., 1]).all() and (b[..., 3] <= b[..., 2]).all()
    assert (b[..., 0]+b[..., 1] <= 2+1e-10).all()
    assert not any('future' in x or 'label' in x for x in inspect.signature(model.retrieve).parameters)


def test_exact_retrieval_excludes_whole_recording_and_unknown_support_is_not_a_filter():
    z = np.arange(80, dtype=np.float32)[:, None]
    rec = np.array(['a']*40+['b']*40); causal = np.ones(80, bool)
    ids, nn, refs = model.retrieve(z, rec, causal, query_ids=np.array([0, 79]))
    assert ids.tolist() == [0, 79]
    assert not np.any(rec[nn] == rec[ids, None])
    assert all(r['same_recording_donors'] == 0 for r in refs)
    assert set(nn[0]).issubset(set(range(40, 80)))


def test_shuffle_preserves_joint_moments_missingness_and_recordings():
    y = labels(80); y[:40, :, 1] = np.linspace(.1, .4, 40)[:, None]
    y[:40, :, 4] = y[:40, :, 1]/2; y[3] = np.nan
    target, known = model.encode(y, np.ones((80, 7)), np.ones(80))
    rec = np.array(['a']*40+['b']*40)
    a = model.shuffled_targets(target, known, rec)
    b = model.shuffled_targets(target, known, rec)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[~known], target[~known])
    assert not np.array_equal(a[:40], target[:40])
    assert (a[..., 4] <= a[..., 1]).all()


def test_global_decodes_with_actual_query_envelope_before_benefit_cap():
    y = labels(80); y[..., 2] = .21; y[..., 3] = .1
    target, known = model.encode(y, np.ones((80, 7)), np.ones(80))
    rec = np.array(['a']*40+['b']*40); real = np.ones((80, 7), bool)
    queries = np.array([0, 79]); neighbors = np.array([np.arange(40, 72), np.arange(32)])
    out, _ = model.predictions(target, known, real, rec, rec, np.ones(80, bool), queries,
                              neighbors, np.full((80, 7), .1), np.full(80, 10.))
    np.testing.assert_allclose(out['global_bounds'][..., 0], .02)
    np.testing.assert_allclose(out['global_bounds'][..., 2], 2.1)


def test_changing_query_recording_labels_cannot_change_its_predictions():
    y = labels(80); rec = np.array(['a']*40+['b']*40)
    env = np.ones((80, 7)); scale = np.ones(80); real = np.ones((80, 7), bool)
    query = np.array([0, 1]); neighbors = np.tile(np.arange(40, 72), (2, 1))
    first, known = model.encode(y, env, scale)
    before, _ = model.predictions(first, known, real, rec, rec, np.ones(80, bool), query, neighbors, env, scale)
    y[:40] = np.nan
    second, known = model.encode(y, env, scale)
    after, _ = model.predictions(second, known, real, rec, rec, np.ones(80, bool), query, neighbors, env, scale)
    for mode in model.MODES:
        np.testing.assert_array_equal(before[mode], after[mode])
