import json
import tarfile

import numpy as np
import pytest

from src.evaluation.m3w_imptc_intake import (ROW_DTYPE, integer, parse_master,
    parse_track, past_scene, past_window, safe_member, strict_json, support)


def fixture(n=24):
    timestamp = 1679472906800035
    master = {str(timestamp+i*40000): dict(id=i) for i in range(n)}
    track = dict(overview=dict(lenght=n, class_id=0), track_data={str(i+1):
        dict(ts=str(timestamp+i*40000), coordinates=[i*2., i*3., 1.6],
            velocity=1e9, status=1, class_prob=.5) for i in range(n)})
    return parse_master(json.dumps(master)), track


def test_release_ordinal_keys_and_exact_microsecond_join():
    master, track = fixture()
    rows, audit = parse_track(json.dumps(track), master, 42)
    assert rows.dtype == ROW_DTYPE
    np.testing.assert_array_equal(rows['timestamp'], master[:, 0])
    assert audit['overview_length'] == 24
    assert audit['status_counts'] == {'1': 24}


def test_metadata_and_published_velocity_are_never_inputs():
    master, track = fixture()
    rows, _ = parse_track(json.dumps(track), master, 42)
    track['overview'] = dict(lenght=10**9, class_id=8, last_ts=999999999999999)
    for event in track['track_data'].values():
        event.update(velocity=-1e8, status=2, class_prob=1., source_type=2)
    altered, _ = parse_track(json.dumps(track), master, 42)
    np.testing.assert_array_equal(rows, altered)
    h = past_window(rows, 42, 12)
    np.testing.assert_array_equal(h['velocity_causal_fd'][1:], np.tile([2., 3.], (7, 1)))
    assert not h['velocity_valid_mask'][0]


def test_future_mutation_and_future_arrival_invariance():
    master, track = fixture()
    rows, _ = parse_track(json.dumps(track), master, 42)
    altered = rows.copy()
    altered['x'][altered['frame_id'] > 12] = 1e10
    late = rows[rows['frame_id'] > 12].copy(); late['agent_id'] = 99
    alternatives = [altered, rows[rows['frame_id'] <= 12], np.concatenate([rows, late])]
    expected = past_scene(rows, 12)
    for alternative in alternatives:
        got = past_scene(alternative, 12)
        assert got.keys() == expected.keys()
        for key in expected[42]:
            np.testing.assert_equal(got[42][key], expected[42][key])


def test_future_not_required_for_population_and_gap_not_interpolated():
    master, track = fixture(20)
    rows, _ = parse_track(json.dumps(track), master, 1)
    result = support(rows, lengths=(8,), strides=(1,))['K8_stride1']
    assert result['past_eligible'] == 13
    assert result['complete_future12'] == 1
    assert result['partial_future12'] == 11 and result['no_future12'] == 1
    rows = rows[rows['frame_id'] != 9]
    h = past_window(rows, 1, 11)
    assert h['valid_mask'].sum() == 7
    assert h['velocity_valid_mask'].sum() == 5
    assert support(rows, lengths=(8,), strides=(1,))['K8_stride1']['past_eligible'] == 5


def test_stride_uses_master_clock_not_track_ordinal_or_shorter_gap():
    master, track = fixture(201)
    rows, _ = parse_track(json.dumps(track), master, 1)
    out = support(rows, lengths=(8,), strides=(10,))['K8_stride10']
    assert out['past_eligible'] == 131 and out['complete_future12'] == 11
    h = past_window(rows, 1, 80, stride=10)
    np.testing.assert_array_equal(h['velocity_causal_fd'][1:], np.tile([2., 3.], (7, 1)))
    assert not past_window(rows, 1, 69, stride=10)['valid_mask'].all()


@pytest.mark.parametrize('value', [True, '1.2', 'NaN', 'Infinity', 2**64])
def test_integer_fail_closed(value):
    with pytest.raises(ValueError): integer(value)


@pytest.mark.parametrize('text', ['{"a":1,"a":2}', '{"a":NaN}'])
def test_json_rejects_ambiguity(text):
    with pytest.raises(ValueError): strict_json(text)


@pytest.mark.parametrize('mutation', ['off_clock', 'duplicate_time', 'nan', 'bad_xyz'])
def test_track_rejects_bad_observations(mutation):
    master, track = fixture()
    if mutation == 'off_clock': track['track_data']['1']['ts'] = '123'
    if mutation == 'duplicate_time': track['track_data']['2']['ts'] = track['track_data']['1']['ts']
    if mutation == 'nan': track['track_data']['1']['coordinates'][0] = float('nan')
    if mutation == 'bad_xyz': track['track_data']['1']['coordinates'] = [1, 2]
    with pytest.raises(ValueError): parse_track(json.dumps(track), master, 1)


def test_master_rejects_nonmonotonic_or_duplicate_index():
    with pytest.raises(ValueError): parse_master('{"100":{"id":1},"140":{"id":1}}')
    with pytest.raises(ValueError): parse_master('{"100":{"id":2},"140":{"id":1}}')


@pytest.mark.parametrize('name', ['/absolute', '../escape', 'seq/../escape', 'seq\\bad'])
def test_archive_rejects_unsafe_paths(name):
    with pytest.raises(ValueError): safe_member(tarfile.TarInfo(name))


@pytest.mark.parametrize('kind', [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_archive_rejects_links_and_special_files(kind):
    member = tarfile.TarInfo('seq/track.json'); member.type = kind
    with pytest.raises(ValueError): safe_member(member)


def test_empty_queries_and_unknown_agent_do_not_look_ahead():
    master, track = fixture()
    rows, _ = parse_track(json.dumps(track), master, 1)
    assert past_scene(rows, -1) == {}
    assert not past_window(rows, 99, 12)['valid_mask'].any()
