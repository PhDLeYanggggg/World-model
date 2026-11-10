import numpy as np
import pytest

from scripts.audit_m3w_songdo_contrast_support import contrast, seed_view, summarize
from src.evaluation.m3w_interaction_frontier import pair_reduction
from src.evaluation.m3w_opendd_location import paired
from src.evaluation.m3w_songdo_matched_readout import ARMS


@pytest.mark.parametrize('coverage', [.0, .4, 1.])
def test_geometry_bounds_both_observed_and_missing_debited_contrasts(coverage):
    rng = np.random.default_rng(13)
    base, joint, target = rng.normal(size=(3, 5, 12, 2))
    joint[4] = base[4]
    valid = rng.random((5, 12)) < coverage
    v = contrast(joint, base, {0: 1, 1: 0}, {2: 2, 3: 1}, 8)
    gain, lower = paired(base, joint, target, valid)
    assert abs(gain.mean()) <= v['query_mean_waypoint_upper_native']
    assert abs(lower.mean()) <= v['query_mean_waypoint_upper_native']
    pair = pair_reduction(joint[None], base, target, valid)[0]
    assert abs(pair[1]) <= v['query_pair_grid_mean_upper_native']
    assert v['flight_grid_mean_waypoint_upper_native'] == pytest.approx(v['query_mean_waypoint_upper_native']/8.)


def test_uniform_translation_changes_absolute_but_not_relative_motion():
    a = np.zeros((4, 12, 2))
    v = contrast(a+1., a, {0: 1, 1: 2}, {0: 0, 1: 1}, 1)
    assert v['shared_recipients'] == 2 and v['family_changes_on_shared'] == 2
    assert v['changed_rows'] == 4
    assert not v['relative_nonzero'] and v['query_pair_grid_mean_upper_native'] == 0.
    assert v['query_mean_waypoint_upper_native'] == pytest.approx(np.sqrt(2.))


def test_seed_query_and_radius_must_match_frozen_receipt():
    z = dict(seed17_bounds=np.array([0, 3]), seed17_query_index=np.array(0), seed17_legal_pairs=np.array(3),
             seed17_radius=np.array(1.))
    for a in ARMS:
        z[f'seed17_{a}_rows'] = np.array([0, 1])
        z[f'seed17_{a}_families'] = np.array([0, 0])
        z[f'seed17_{a}_paths'] = np.tile(np.array([1., 0.]), (2, 12, 1))
    info = dict(query=0, agents=3, pairs=3, radius=1., joint_differs={a: False for a in ARMS[:-1]})
    cv = np.zeros((3, 12, 2))
    r = seed_view(cv, np.array([0, 3]), z, 17, info)
    assert all(v['changed_rows'] == 0 for v in r.values())
    assert summarize([(1, r['independent'])])['components_with_trajectory_difference'] == 0
    z['seed17_joint_paths'] *= 2.
    with pytest.raises(ValueError, match='radius differs'):
        seed_view(cv, np.array([0, 3]), z, 17, info)
