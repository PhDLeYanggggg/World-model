import pytest

from src.evaluation.m3w_traf_intake import (
    audit_directory, audit_recording, frame_support, parse_frame,
    require_geometry_conversion,
)


def row(frame, agent='ped0', box=(20, 30, 4, 6)):
    return ','.join(map(str, (frame, 1, *box, agent)))


def test_raw_fields_are_not_silently_interpreted_as_coordinates():
    frame, agents = parse_frame(row(4))
    assert frame == 4
    assert agents == [('ped0', 'ped', (20., 30., 4., 6.))]
    with pytest.raises(ValueError, match='unresolved'):
        require_geometry_conversion()


@pytest.mark.parametrize('text', [
    '0,2,0,0,2,3,ped0', '0,0,', '0.5,0', '-1,0', '0,-1',
    '0,1,0,0,nan,3,ped0', '0,1,0,0,2,3,',
    '0,2,0,0,2,3,ped0,1,1,2,3,ped0',
])
def test_invalid_frames_rejected(text):
    with pytest.raises(ValueError):
        parse_frame(text)


def test_empty_frame_and_unknown_class_are_preserved():
    assert parse_frame('0,0') == (0, [])
    assert parse_frame(row(0, 'unknown0'))[1][0][1] == 'unknown'
    assert parse_frame(row(0, '0'))[1][0][0:2] == ('0', 'untyped_id')
    assert parse_frame(row(0, 'ped'))[1][0][0:2] == ('ped', 'untyped_id')


def test_history_future_and_gaps_are_separate():
    support = frame_support(list(range(40)))
    assert support['stride1']['history_only']['8'] == 33
    assert support['stride1']['history_and_12step_future']['8'] == 21
    assert support['history8_full_raw_horizon']['10'] == 23
    assert support['history8_full_raw_horizon']['50'] == 0
    gapped = frame_support(list(range(10)) + list(range(11, 21)))
    assert gapped['stride1']['history_and_12step_future']['8'] == 0
    assert gapped['discontinuous_edges_at_stride1'] == 1


def test_stride12_is_availability_only_and_retains_phases():
    support = frame_support(list(range(240)))
    assert support['stride12']['history_and_12step_future']['8'] == 12
    assert support['stride12']['history_only']['8'] == 156
    assert support['stride1']['history_and_12step_future']['8'] == 221


def test_duplicate_agent_frame_support_is_refused():
    with pytest.raises(ValueError, match='unique'):
        frame_support([0, 1, 1, 2])


@pytest.mark.parametrize('frames', [[0, 1.5], [-1, 0], [[0, 1]]])
def test_frame_support_does_not_silently_cast_identities(frames):
    with pytest.raises(ValueError, match='integer sequence'):
        frame_support(frames)


def test_bad_frame_quarantines_recording_not_silent_partial_acceptance(tmp_path):
    p = tmp_path / 'TRAF1_gt.txt'
    p.write_text('\n'.join([row(i) for i in range(30)] + [row(29)]))
    result, _ = audit_recording(p)
    assert result['quality_status'] == 'quarantined'
    assert result['frame_errors'][0]['kind'] == 'nonincreasing_frame'
    assert result['counts_before_recording_quarantine']['track_count'] == 1
    assert result['eligible_for_forecast_use'] is False


def test_xyxy_conflict_and_duplicates_do_not_prove_xywh(tmp_path):
    for n in (1, 2):
        (tmp_path / f'TRAF{n}_gt.txt').write_text('\n'.join(row(i) for i in range(20)))
    result = audit_directory(tmp_path)
    assert result['annotation_file_count'] == 2
    assert result['bbox_checks']['incompatible_with_xyxy'] == 40
    assert result['bbox_checks']['positive_third_fourth_values'] == 40
    assert result['geometry_interpretation'] == 'unresolved_no_center_or_footpoint_conversion'
    assert len(result['exact_file_duplicate_groups']) == 1
    assert len(result['exact_relative_frame_box_track_duplicate_groups']) == 1
    assert result['independent_physical_site_count'] is None
    assert result['admitted_recordings'] == 0


def test_prefix_groups_not_independent_sites_and_unknown_types_block(tmp_path):
    (tmp_path / 'TRAF53_1_gt.txt').write_text(row(0))
    (tmp_path / 'TRAF53_2_gt.txt').write_text(row(0, 'alien0'))
    result = audit_directory(tmp_path)
    assert result['filename_family_count'] == 1
    assert result['quarantined_recordings'] == ['TRAF53_2']
    assert result['physical_scene_mapping_status'] == 'unknown_filename_families_are_not_sites'


def test_duplicate_box_trajectories_under_different_ids_are_flagged(tmp_path):
    p = tmp_path / 'TRAF3_gt.txt'
    p.write_text('\n'.join(f'{i},2,0,0,3,4,ped0,0,0,3,4,ped1' for i in range(20)))
    result, _ = audit_recording(p)
    assert result['quality_status'] == 'quarantined'
    assert result['duplicate_tracks_within_recording'] == [['ped0', 'ped1']]


def test_no_annotation_files_is_not_success(tmp_path):
    with pytest.raises(ValueError, match='No TRAF'):
        audit_directory(tmp_path)


def test_empty_file_quarantined(tmp_path):
    path = tmp_path / 'TRAF4_gt.txt'
    path.write_text('')
    result, _ = audit_recording(path)
    assert result['quality_status'] == 'quarantined'
    assert result['frame_errors'][0]['kind'] == 'empty_or_unparseable_recording'
