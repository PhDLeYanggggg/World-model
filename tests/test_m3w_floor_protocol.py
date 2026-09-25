import copy
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.run_m3w_european_floor_opportunity import choices, validate
from scripts.verify_m3w_floor_accounting import verify_ledger
from src.evaluation.m3w_floor_opportunity import floor_ledger

CFG = json.loads(Path('configs/m3w_european_floor_opportunity_v1.json').read_text())


def test_registered_both_modes_no_deployment():
    validate(CFG)
    for field, value in (('modes', ['fitting']), ('new_training', True), ('risk_budget', .03),
                         ('deployment_changed', True), ('reserved_roles_opened', True)):
        bad = copy.deepcopy(CFG); bad[field] = value
        with pytest.raises(ValueError):
            validate(bad)


def test_oracle_fde_is_same_ade_chosen_forecast_not_second_oracle():
    a = lambda x: np.array(x, float)
    d, fd, ade, fde = choices(a([1, 9, 8]), a([20, 0, 1]), a([2, 2, 4]), a([1, 7, 8]),
        a([3, 3, 1]), a([3, 2, 9]), np.array([False, True, False]), np.ones(3, bool))
    np.testing.assert_array_equal(ade['floor_neural_oracle'], [1, 2, 4])
    np.testing.assert_array_equal(fde['floor_neural_oracle'], [20, 7, 8])
    np.testing.assert_array_equal(fde['union_oracle'], [20, 7, 9])
    np.testing.assert_array_equal(fde['rebased_neural'], [1, 0, 8])


def test_alternate_ledger_reduction_detects_tampering():
    cv = np.array([10., 10., 4., 0., np.nan])
    d = np.array([8., 12., 3., 0., np.nan]); n = np.array([6., 11., 5., 1., np.nan])
    why = np.array([5, 4, 5, 5, 4]); sites = np.array(['a', 'a', 'b', 'b', 'b'])
    mask = np.ones(5, bool)
    r = floor_ledger(cv, d, n, why, sites, expected_scenes=['a', 'b'])
    verify_ledger(cv, d, n, why, sites, mask, r, CFG)
    r['by_scene']['a']['sums']['fallback_relief'] += 1
    with pytest.raises(AssertionError):
        verify_ledger(cv, d, n, why, sites, mask, r, CFG)
