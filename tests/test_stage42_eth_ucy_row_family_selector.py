import numpy as np

from src import stage42_eth_ucy_row_family_selector as jk


def test_select_errors_uses_best_family_on_switched_rows():
    floor_ade = np.asarray([10.0, 10.0, 10.0])
    floor_fde = floor_ade.copy()
    family_ade = np.asarray([[1.0, 9.0], [8.0, 2.0], [3.0, 4.0]])
    family_fde = family_ade.copy()
    selected_ade, selected_fde = jk._select_errors(
        floor_ade,
        floor_fde,
        family_ade,
        family_fde,
        np.asarray([0, 1, 0]),
        np.asarray([True, True, False]),
    )
    assert selected_ade.tolist() == [1.0, 2.0, 10.0]
    assert selected_fde.tolist() == [1.0, 2.0, 10.0]


def test_thresholds_include_zero_and_quantiles():
    score = np.asarray([-2.0, -1.0, 0.5, 4.0])
    thresholds = jk._thresholds(score, np.asarray([True, True, True, False]))
    assert 0.0 in thresholds
    assert min(thresholds) == -2.0
    assert max(thresholds) == 0.5


def test_deployable_requires_easy_safe_and_core_positive():
    assert jk._deployable(
        {
            "all_improvement": 0.05,
            "t50_improvement": 0.04,
            "hard_failure_improvement": 0.0,
            "easy_degradation": 0.0,
        }
    )
    assert not jk._deployable(
        {
            "all_improvement": 0.05,
            "t50_improvement": 0.04,
            "hard_failure_improvement": 0.0,
            "easy_degradation": 0.03,
        }
    )
