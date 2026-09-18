import copy

import pytest

from scripts.analyze_m3w_auxiliary_mechanism import analyze_report


def synthetic_report():
    trials = []
    costs = {'main4k': 10., 'sdd_permuted': 9., 'sdd_aux': 8., 'no_aux': 12.}
    for arm, cost in costs.items():
        for modality in ('geometry', 'mask_only', 'past_rgb'):
            for seed in (17, 29, 43):
                for fold in range(3):
                    trials.append(dict(arm=arm, modality=modality, seed=seed, fold=fold,
                        vs_CV=dict(primary_ADE=cost, reference_ADE=5., improvement_percent=100*(1-cost/5),
                                   easy_degradation_percent=cost, easy_absolute_harm=.1, tail_ADE_p95=15.),
                        slices={}, train_equal_scene_gain_percent=2.))
    return dict(complete=True, trials=trials)


def test_exposure_source_decomposition_and_no_false_safety():
    result = analyze_report(synthetic_report(), [17, 29, 43], ['geometry', 'mask_only', 'past_rgb'])
    d = result['decomposition']['geometry']
    assert d['actual_source_minus_main4k_ADE'] == -2
    assert d['main4k_minus_main6k_ADE'] == -2
    assert d['actual_source_minus_main6k_ADE'] == -4
    assert result['contrasts']['geometry:sdd_aux_vs_main4k']['gain_percent'] == pytest.approx(20)
    assert result['contrasts']['geometry:sdd_aux_vs_sdd_permuted']['gain_percent'] == pytest.approx(100/9)
    assert result['groups']['sdd_aux_geometry']['safe_positive_fits'] == 0
    assert not result['independent_confirmation']


def test_duplicate_or_missing_fit_rejected():
    report = synthetic_report()
    report['trials'][-1] = copy.deepcopy(report['trials'][0])
    with pytest.raises(ValueError, match='unique'):
        analyze_report(report, [17, 29, 43], ['geometry', 'mask_only', 'past_rgb'])


def test_changed_cv_reference_rejected():
    report = synthetic_report()
    report['trials'][0]['vs_CV']['reference_ADE'] = 7.
    with pytest.raises(AssertionError):
        analyze_report(report, [17, 29, 43], ['geometry', 'mask_only', 'past_rgb'])
