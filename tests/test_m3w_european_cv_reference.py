import inspect
import json

import numpy as np
import pytest

from src.world_model.m3w_european_cv_reference import cv_reference_design, choose_errors, query_decisions
from src.world_model.m3w_european_source_intervention import paired_cost_labels


def test_reference_repair_does_not_modify_neural_design_or_splits():
    design = dict(baseline_index=3, train_ids=np.array([2, 3]), held_ids=np.array([0, 1]), easy_cut=.5)
    repaired = cv_reference_design(design)
    assert design['baseline_index'] == 3 and repaired['baseline_index'] == 1
    assert repaired['easy_cut'] == design['easy_cut']
    np.testing.assert_array_equal(repaired['train_ids'], design['train_ids'])
    with pytest.raises(ValueError):
        cv_reference_design(dict(design, held_ids=np.array([2])))


def test_relative_reference_mismatch_counterexample_and_zero_atom():
    # Improvement over a poor floor can still damage CV-relative easy samples.
    old = paired_cost_labels(np.array([2., 1.]), np.array([1.5, .2]))
    cv = paired_cost_labels(np.array([1., 0.]), np.array([1.5, .2]))
    np.testing.assert_allclose(old, [[.5, 0.], [.8, 0.]])
    np.testing.assert_allclose(cv, [[0., .5], [0., .2]])


def test_binary_forecast_selection_keeps_unknown_support():
    candidate = np.array([1., 3., np.nan])
    ref = np.array([2., 2., np.nan])
    np.testing.assert_equal(choose_errors(np.array([True, False, True]), candidate, ref), [1., 2., np.nan])
    with pytest.raises(ValueError):
        choose_errors(np.array([1., 0., 1.]), candidate, ref)
    with pytest.raises(ValueError):
        choose_errors(np.array([True]*3), candidate, np.ones(3))


def test_query_decisions_have_no_future_api_and_keep_all_past_targets():
    assert not {'target', 'valid', 'future', 'future_endpoint'} & set(inspect.signature(query_decisions).parameters)
    h = np.tile(np.arange(-7, 1)[None, :, None], (3, 1, 2)).astype(float)
    b = np.ones((3, 12, 2))
    reg = dict(predicted_positive_harm_budget=.02, pair_weight=.1,
        edge_radius_bbox_widths=3., proximity_threshold_bbox_widths=.5, solver_seconds=2.)
    kw = dict(history=h, current_xy=np.array([[0., 0.], [10., 0.], [0., 20.]]),
        widths=np.ones(3)*2, recordings=np.array([0, 0, 1]), frames=np.array([10, 10, 20]),
        sites=np.array(['a', 'a', 'b']), baseline=b, candidate=b+.1,
        costs=np.tile([2., .001], (3, 1)), held_ids=np.arange(3), query_mask=np.ones(3, bool),
        cost_scale=1., heartbeat=lambda **kw: None)
    choices, reports = query_decisions(reg, **kw)
    assert choices['ids'].tolist() == [0, 1, 2]
    assert len(reports) == 2
    for name in ('independent', 'unary_exact', 'joint_exact'):
        assert choices[name].sum() == 3
    json.dumps(reports, allow_nan=False)
    with pytest.raises(ValueError):
        query_decisions(reg, **dict(kw, cost_scale=0))


def nested_fixture(tmp_path, monkeypatch):
    from scripts import run_m3w_european_cv_reference as runner
    previous = runner.previous
    predpath = tmp_path/'predictions'
    predpath.mkdir()
    for fitted, site in [(1, 'b'), (2, 'c')]:
        (predpath/f'single{fitted}_seed17.json').write_text(json.dumps(dict(
            identity=dict(fit_sites=[site]), sha256='synthetic_prediction_receipt')))
    monkeypatch.setattr(previous.parent, 'PRIVATE', tmp_path)
    p = np.ones((6, 12, 2), np.float32)*2
    monkeypatch.setattr(previous, 'prediction', lambda key, ids: p[ids].copy())
    h = np.zeros((6, 8, 2)); h[:, :, 0] = np.arange(-7, 1)
    g = np.zeros((6, 476), np.float32); g[:, :16] = h.reshape(6, -1)
    g[:, 16:24] = np.arange(-7, 1)/12
    origin = np.zeros((6, 2))
    target = np.tile(np.arange(1, 13)[None, :, None], (6, 1, 2)).astype(float)
    valid = np.ones((6, 12), bool)
    valid[3] = False
    base = np.ones((6, 6)); base[3] = np.nan
    data = dict(sites=np.array(['a', 'a', 'b', 'b', 'c', 'c']), history=h, origin=origin,
        geometry=g, target_eval=target, valid=valid, baseline_ade=base)
    design = dict(train_ids=np.array([2, 3, 4, 5]), held_ids=np.array([0, 1]), baseline_index=3,
        easy_cut=.5, hard_cut=2.)
    identity = dict(folds=dict(a=0, b=1, c=2))
    ti = dict(fold=0, seed=17)
    return runner, data, identity, design, ti, predpath


def test_held_future_mutation_cannot_change_features_cost_labels_or_preprocessing(tmp_path, monkeypatch):
    runner, data, identity, design, ti, _ = nested_fixture(tmp_path, monkeypatch)
    original = runner.assemble(data, identity, 'complement0_seed17', design, ti)
    changed = dict(data, target_eval=data['target_eval'].copy(), baseline_ade=data['baseline_ade'].copy())
    changed['target_eval'][:2] += 100000
    changed['baseline_ade'][:2] += 10000
    repaired = runner.assemble(changed, identity, 'complement0_seed17', design, ti)
    for name in ('x', 'y', 'p', 'b'):
        np.testing.assert_array_equal(original[name], repaired[name])
    for name in ('mean', 'std', 'constant', 'weights', 'known'):
        np.testing.assert_array_equal(original['pr'][name], repaired['pr'][name])
    assert original['lineage'] == repaired['lineage']
    assert original['lineage']['baseline_index'] == 1
    assert original['lineage']['frozen_neural_producer_internal_baseline_index'] == 3
    assert original['pr']['training_sites'] == ['b', 'c']
    assert original['lineage']['unknown_training_rows'] == 1


def test_nested_cost_producer_cannot_include_outer_source(tmp_path, monkeypatch):
    runner, data, identity, design, ti, p = nested_fixture(tmp_path, monkeypatch)
    (p/'single2_seed17.json').write_text(json.dumps(dict(identity=dict(fit_sites=['a', 'c']), sha256='bad')))
    with pytest.raises(ValueError, match='producer fitted'):
        runner.assemble(data, identity, 'complement0_seed17', design, ti)


def test_unknown_or_absent_control_files_are_not_silently_verified(tmp_path, monkeypatch):
    from scripts import run_m3w_european_cv_reference as runner
    monkeypatch.setattr(runner, 'PRIVATE', tmp_path)
    with pytest.raises(ValueError, match='absent'):
        runner.get_decisions({}, {}, {}, np.array([0]), np.zeros((1, 2)), np.array([True]), 'missing', {}, True)


def test_atomic_arrays_and_changed_completed_head_rejected(tmp_path, monkeypatch):
    from scripts import run_m3w_european_cv_reference as runner
    monkeypatch.setattr(runner, 'PRIVATE', tmp_path)
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner.previous, 'ROOT', tmp_path)
    path = tmp_path/'heads'/'test_ridge'/'scores.npz'
    runner.atomic_arrays(path, ids=np.array([1]), costs=np.ones((1, 2)))
    assert not path.with_suffix('.tmp.npz').exists()
    r = dict(identity=dict(identity={'fixed': True}), fit={}, artifacts=dict(scores=runner.artifact(path)))
    (path.parent/'complete.json').write_text(json.dumps(r))
    assert runner.checked_head('test_ridge', {'fixed': True}) == r
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        runner.checked_head('test_ridge', {'fixed': True})
