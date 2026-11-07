import numpy as np
import pytest

from scripts.run_m3w_dronecrowd_blind_preflight import aggregate, role_sets, stream


def proposed():
    empty = dict(recordings=[], groups=[])
    return dict(excluded=empty,
        split_calibration_confirmation=dict(
            calibration=dict(recordings=['a', 'b'], groups=['x']),
            confirmation=dict(recordings=['c'], groups=['y'])),
        frozen_zero_shot=dict(calibration=empty, confirmation=dict(
            recordings=['a', 'b', 'c'], groups=['x', 'y'])))


def test_keeps_whole_groups_in_both_alternatives():
    group = dict(a='x', b='x', c='y')
    sets = role_sets(proposed(), group)
    assert sets['split_calibration_confirmation']['calibration'] == {'a', 'b'}
    plan = proposed()
    plan['split_calibration_confirmation']['calibration']['recordings'].remove('b')
    with pytest.raises(ValueError, match='whole exclusion groups'):
        role_sets(plan, group)


def test_stream_uses_no_label_api_and_retains_masked_neighbors():
    class Reader:
        query_frames = np.array([7])

        def get_inputs(self, frame):
            mask = np.ones((2, 8), dtype=bool)
            mask[1, 0] = False
            return dict(query_frame=frame, observation_frames=np.arange(8), positions=np.ones((2, 8, 2)),
                history_mask=mask, agent_ids=np.array([10, 11]), history_complete=mask.all(1))

        def get_labels(self, _):
            raise AssertionError('Outcome interface must not be called')

    result, fingerprint = stream(Reader())
    assert result == dict(queries=1, with_targets=1, visible=2, eligible=1, masked_neighbors=1)
    assert stream(Reader())[1] == fingerprint
    assert aggregate({'a'}, {'a': result}, {'a': 'x'})['largest_group_row_share'] == 1.
    assert aggregate(set(), {}, {})['largest_group_row_share'] is None
