import copy

import numpy as np
import pytest
import torch

from src.world_model.m3w_native_nested import nested_fold_design, require_source_producer, cost_supervision
from src.world_model.m3w_native_forecast import fit_trial
from src.world_model.m3w_supervised_intervention import build_forecaster
from scripts.run_m3w_native_nested import write_arrays, independent_label_check


def population():
    n = 16
    g = np.zeros((n, 476), np.float32)
    g[:, :16] = np.tile(np.column_stack((np.arange(-7, 1)/12, np.zeros(8))).ravel(), (n, 1))
    g[:, 16:24] = np.arange(-7, 1)/12
    g[:, 332:356] = np.tile(np.column_stack((np.arange(1, 13)/12, np.zeros(12))).ravel(), (n, 1))
    target = g[:, 332:356].reshape(n, 12, 2).copy()*1.2
    valid = np.ones((n, 12), bool); valid[1] = False; target[1] = np.nan
    return dict(geometry=g, target=target, valid=valid, scale=np.arange(1, n+1, dtype=float),
                sites=np.repeat(['a', 'b', 'c', 'd'], 4))


def test_both_exclusions_and_their_labels_cannot_affect_loss_preprocessing():
    data = population(); original = nested_fold_design(data, ['a', 'b'])
    poisoned = copy.deepcopy(data)
    poisoned['target'][:8] = 1e15; poisoned['scale'][:8] = -500
    poisoned['valid'][:8] = False
    again = nested_fold_design(poisoned, ['b', 'a'])
    np.testing.assert_array_equal(original['train_ids'], np.arange(8, 16))
    np.testing.assert_array_equal(original['held_ids'], np.arange(8))
    for k in ['factors', 'train_ids', 'held_ids', 'hard_cut']:
        np.testing.assert_array_equal(original[k], again[k])
    assert original['normalizers'] == again['normalizers']
    assert not original['factors'][:8].any()


@pytest.mark.parametrize('pair', [[], ['a'], ['a', 'a'], ['a', 'b', 'c'], ['a', 'missing']])
def test_invalid_exclusions_rejected(pair):
    with pytest.raises(ValueError):
        nested_fold_design(population(), pair)


def record():
    return dict(id='pair_ab_17', seed=17, excluded_sites=['a', 'b'], training_sites=['c', 'd'],
        preprocessing_fit_sites=['c', 'd'], parents=[], initialization='random_seed',
        checkpoint_selection_sites=[], calibration_sites=[], objective='native_coordinate',
        research_design_exposed_sites=['a', 'b', 'c', 'd'])


def test_clean_source_producer_is_not_independent_confirmation():
    result = require_source_producer(record(), outer_site='a', row_site='b', roster='abcd', seed=17)
    assert result['fitting_exclusion_pass']
    assert not result['independent_confirmation']


@pytest.mark.parametrize('field,value', [
    ('training_sites', ['a', 'c', 'd']), ('preprocessing_fit_sites', ['a', 'c', 'd']),
    ('parents', ['seen_a_parent']), ('checkpoint_selection_sites', ['a']),
    ('calibration_sites', ['a']), ('seed', 29), ('excluded_sites', ['b']),
    ('research_design_exposed_sites', [])])
def test_indirect_and_undeclared_exposure_rejected(field, value):
    r = record(); r[field] = value
    with pytest.raises(ValueError, match='exclusion'):
        require_source_producer(r, outer_site='a', row_site='b', roster='abcd', seed=17)


def test_supervision_preserves_unknowns_and_gain_harm_identity():
    d = population(); baseline = d['geometry'][:, 332:356].reshape(-1, 12, 2)
    s = cost_supervision(baseline*.9, baseline, d['target'], d['valid'], d['scale'])
    for k in ['neural_ade', 'baseline_ade', 'gain', 'benefit', 'harm', 'neural_fde', 'baseline_fde']:
        assert np.isnan(s[k][1])
    np.testing.assert_allclose(s['benefit']-s['harm'], s['gain'], equal_nan=True)
    assert s['supported_steps'][1] == 0 and not s['complete_future'][1]
    assert s['harm'][0] > 0 and s['benefit'][0] == 0


def test_actual_pair_excluded_training_and_resume_match(tmp_path):
    torch.set_num_threads(2)
    d = population(); fold = nested_fold_design(d, ['a', 'b'])
    settings = dict(steps=4, batch_size=4, learning_rate=.001, minimum_lr_ratio=.01,
        weight_decay=.0001, gradient_clip=5, checkpoint_every=2, heartbeat_every=1)
    architecture = dict(width=8, heads=2, layers=1, neighbor_policy='complete_aligned_history',
        input_conditioning='observed_joint_max_norm', output_parameterization='motion_bounded')
    def run(path, resume=False, stop_at=None):
        torch.manual_seed(17); m = build_forecaster(architecture)
        r = fit_trial(m, d, fold, seed=17, settings=settings, identity=record(), directory=path,
            resume=resume, stop_at=stop_at, heartbeat=lambda **_:None)
        return m, r
    full, _ = run(tmp_path/'full')
    run(tmp_path/'resume', stop_at=2)
    resumed, report = run(tmp_path/'resume', resume=True)
    for k, v in full.state_dict().items():
        torch.testing.assert_close(v, resumed.state_dict()[k], atol=0, rtol=0)
    cp = torch.load(tmp_path/'resume'/'checkpoint.pt', weights_only=False)
    assert not cp['draws'][:8].any() and report['held_rows_sampled'] == 0


def test_cached_arrays_are_exact_and_cannot_gain_hidden_labels(tmp_path):
    path = tmp_path/'predictions.npz'
    arrays = dict(ids=np.arange(2), prediction=np.ones((2, 12, 2), np.float32))
    write_arrays(path, arrays); write_arrays(path, arrays)
    with pytest.raises(ValueError, match='schema'):
        write_arrays(path, dict(arrays, future_endpoint=np.zeros((2, 2))))
    with pytest.raises(ValueError, match='dtype'):
        write_arrays(path, dict(arrays, prediction=arrays['prediction'].astype(float)))
    with pytest.raises(AssertionError):
        write_arrays(path, dict(arrays, ids=np.array([1, 0])))


def test_independent_label_check_detects_wrong_harm_and_scale():
    d = population(); baseline = d['geometry'][:, 332:356].reshape(-1, 12, 2)
    prediction = baseline*.9
    labels = cost_supervision(prediction, baseline, d['target'], d['valid'], d['scale'])
    independent_label_check(prediction, baseline, d['target'], d['valid'], d['scale'], labels)
    with pytest.raises(AssertionError):
        independent_label_check(prediction, baseline, d['target'], d['valid'], d['scale']*2, labels)
    labels['harm'][0] += .2
    with pytest.raises(AssertionError):
        independent_label_check(prediction, baseline, d['target'], d['valid'], d['scale'], labels)
