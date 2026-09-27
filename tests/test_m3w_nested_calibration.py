import numpy as np
import pytest

from src.evaluation.m3w_nested_calibration import inner_halves, role_sets, check_producer_roles
from src.world_model.m3w_european_source_forecast import fit_design


def roster():
    return {f'site{i}': i//4 for i in range(12)}


def test_inner_halves_depend_on_causal_support_only():
    folds = roster()
    counts = {s: 10+i for i, s in enumerate(folds)}
    halves = inner_halves(folds, counts, 'fixed')
    assert halves == inner_halves(dict(reversed(list(folds.items()))), counts, 'fixed')
    for f, parts in halves.items():
        assert list(map(len, parts)) == [2, 2]
        assert not set(parts[0]) & set(parts[1])
        assert set(parts[0]+parts[1]) == {s for s in folds if folds[s] == int(f)}


def test_all_six_rotations_have_disjoint_roles_and_inner_producers():
    folds = roster()
    halves = inner_halves(folds, {s: 10 for s in folds}, 'fixed')
    for fit in range(3):
        for cal in range(3):
            if cal == fit:
                continue
            roles = role_sets(folds, fit, cal)
            assert set().union(*map(set, roles.values())) == set(folds)
            for side in range(2):
                assert check_producer_roles(halves[str(fit)][side], halves[str(fit)][1-side],
                    roles['calibration'], roles['readout'])


def test_any_producer_exposure_to_calibration_is_rejected():
    with pytest.raises(ValueError, match='disjoint'):
        check_producer_roles(['a', 'c'], ['b'], ['c'], ['d'])
    with pytest.raises(ValueError):
        role_sets(roster(), 0, 0)


def test_forecast_design_ignores_calibration_and_readout_labels():
    sites = np.repeat(list(roster()), 3)
    data = dict(sites=sites)
    errors = np.tile(np.arange(1., 7.), (len(sites), 1))
    fit = ['site0', 'site1']
    first = fit_design(data, fit, errors)
    errors[~np.isin(sites, fit)] = np.nan
    second = fit_design(data, fit, errors)
    for name in ('baseline_index', 'normalizers', 'easy_cut', 'hard_cut'):
        assert first[name] == second[name]
    np.testing.assert_array_equal(first['factors'], second['factors'])
