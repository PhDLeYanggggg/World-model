import pytest

from scripts.verify_m3w_ht21_support import recount
from src.evaluation.m3w_ht21_admission import admission, require_role


def recording(motion=False, ground_truth=True):
    return dict(sequence='fixture', metadata=dict(camera_motion=motion),
                ground_truth_present=ground_truth)


@pytest.mark.parametrize('role', [
    'supervised_training', 'representation_pretraining', 'risk_calibration',
    'official_eval', 'confirmation',
])
def test_even_static_camera_with_gt_is_not_scientifically_admitted(role):
    with pytest.raises(ValueError, match='source-audit only'):
        require_role(recording(), role)


@pytest.mark.parametrize('motion', [True, None])
def test_camera_motion_blocker(motion):
    result = admission(recording(motion), 'source_audit')
    assert result['allowed']
    assert 'camera_motion_present_or_unknown_no_compensation' in result['scientific_use_blockers']


def test_detections_cannot_substitute_missing_gt():
    result = admission(recording(ground_truth=False), 'official_eval')
    assert not result['allowed']
    assert 'released_ground_truth_missing_detections_are_not_targets' in result['scientific_use_blockers']


def test_unknown_role_fails_closed():
    with pytest.raises(ValueError, match='Unknown'):
        require_role(recording(), 'train')


@pytest.mark.parametrize('stride', [1, 5, 10])
def test_independent_recount_contiguous_fixture(stride):
    counts, endpoints = recount({1: set(range(1, 241))}, 8, stride)
    assert counts == dict(past_eligible=240-7*stride,
                         complete_future12=240-19*stride,
                         partial_future12=11*stride, no_future12=stride)
    assert endpoints == {str(h): max(0, 240-7*stride-h) for h in (10,25,50,100)}


def test_independent_recount_does_not_bridge_gaps():
    counts, _ = recount({1: set(range(1,21))-{10}}, 8, 1)
    assert counts['past_eligible'] == 5
