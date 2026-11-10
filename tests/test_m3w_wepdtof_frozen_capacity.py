from itertools import product

import numpy as np
import pytest

from scripts.audit_m3w_wepdtof_frozen_capacity import archived_actions, grouping, menu_distance, upper_bounds
from src.evaluation.m3w_opendd_location import paired


def example(seed):
    rng = np.random.default_rng(seed)
    cv, ols, target, arbitrary = rng.normal(size=(4, 5, 12, 2))
    direction = rng.normal(size=(5, 2, 12, 2))
    radius = np.array([1.2, 1.2, 1.2, .4, .4])
    offset = direction*radius[:, None, None, None]/np.linalg.norm(direction, axis=-1).max(2)[..., None, None]
    paths = cv[:, None]+offset
    q = np.array([0, 0, 0, 1, 1])
    groups = grouping(q, np.linalg.norm(offset, axis=-1))
    w = 1./np.bincount(q)[q]
    valid = rng.random((len(cv), 12)) > .3
    return cv, ols, target, arbitrary, paths, groups, w, valid


def proposals(cv, paths, groups):
    for choice in product(range(-1, paths.shape[1]), repeat=len(cv)):
        choice = np.array(choice)
        take = choice >= 0
        if all(take[ix].sum() <= n for ix, _, n, _ in groups):
            p = cv.copy()
            p[take] = paths[take, choice[take]]
            yield p


@pytest.mark.parametrize('seed', [17, 29, 43])
def test_both_utility_bounds_cover_all_legal_menu_choices(seed):
    cv, ols, target, arbitrary, paths, groups, w, valid = example(seed)
    mass = float(w @ np.where(valid, np.linalg.norm(cv-target, axis=-1), 0.).sum(1))

    def scores(p):
        a, b = paired(ols, p, target, valid)
        return {'known_vs_ols4': 100*(w @ a.sum(1))/mass, 'lower_vs_ols4': 100*(w @ b.sum(1))/mass}

    options = list(proposals(cv, paths, groups))
    anchors = {k: (p, scores(p)) for k, p in (('cv', cv), ('ols4', ols), ('other', arbitrary))}
    result = upper_bounds(cv, paths, groups, anchors, mass)
    for p, _ in anchors.values():
        assert menu_distance(cv, paths, p, groups) == pytest.approx(
            max(w @ np.linalg.norm(v-p, axis=-1).sum(1) for v in options))
    for field, bounds in result.items():
        assert max(scores(v)[field] for v in options) <= min(bounds.values()) + 1e-10
        assert bounds['menu_multi_anchor'] <= bounds['menu_cv_anchor'] <= bounds['arbitrary_radius']


def test_frozen_path_and_count_validation():
    cv, _, _, _, paths, groups, _, _ = example(17)
    p = cv.copy()
    p[[0, 2, 4]] = paths[[0, 2, 4], [0, 1, 1]]
    archived_actions(cv, paths, p, groups, 3)
    p[4] = cv[4]
    with pytest.raises(ValueError, match='fixed query budget'):
        archived_actions(cv, paths, p, groups, 2)
    p[4] = paths[4, 1] + .1
    with pytest.raises(ValueError, match='frozen path'):
        archived_actions(cv, paths, p, groups, 3)


def test_arbitrary_radius_can_be_tight():
    cv = np.zeros((2, 12, 2))
    paths = np.ones((2, 1, 12, 2))
    groups = grouping(np.zeros(2, dtype=int), np.linalg.norm(paths-cv[:, None], axis=-1))
    p = cv.copy()
    p[0] = paths[0, 0]
    assert menu_distance(cv, paths, cv, groups) == pytest.approx(6*np.sqrt(2.))
    assert menu_distance(cv, paths, cv, groups) == pytest.approx(np.linalg.norm(p, axis=-1).sum()/2)
