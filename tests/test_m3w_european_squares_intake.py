import io
import stat
import zipfile

import numpy as np
import pytest

from src.evaluation.m3w_european_squares_intake import (
    COLUMNS, DTYPE, frame_support, past_scene, read_raw_csv, safe_zip_member, summarize)


def csv(rows):
    return io.StringIO(','.join(COLUMNS)+'\n'+'\n'.join(','.join(map(str, r)) for r in rows)+'\n')


def row(frame, agent=1):
    return [frame, 2, frame+2, 6, 0, .8, agent, 'person', frame]


def test_source_fields_are_retained_without_filtering_short_track():
    values, names = read_raw_csv(csv([row(i) for i in range(4)]), chunksize=2)
    assert len(values) == 4 and names == {'person': 4}
    assert values['x_min'].tolist() == list(range(4))
    result, _ = summarize(values, names)
    assert result['tracks_shorter_than_30'] == 1
    assert result['support']['K8_stride1']['past_eligible'] == 0


@pytest.mark.parametrize('field,value', [(0, float('nan')), (6, 1.5), (8, -1), (5, 1.1), (2, -1)])
def test_bad_raw_fields_refused(field, value):
    r = row(1)
    r[field] = value
    with pytest.raises(ValueError):
        read_raw_csv(csv([r]))


def test_duplicate_scoped_identity_refused_even_identical():
    with pytest.raises(ValueError, match='Duplicate'):
        read_raw_csv(csv([row(1), row(1)]))
    values, _ = read_raw_csv(csv([row(1, 1), row(1, 2)]))
    assert len(values) == 2


def test_frame_order_checked_across_chunks():
    with pytest.raises(ValueError, match='frame order'):
        read_raw_csv(csv([row(2), row(1)]), chunksize=1)


@pytest.mark.parametrize('stride', [1, 2, 12])
def test_support_matches_independent_set_enumeration(stride):
    rng = np.random.default_rng(38)
    for _ in range(10):
        f = np.flatnonzero(rng.random(1000) > .07)
        fs = set(map(int, f))
        for k in (8, 16, 32, 64):
            expected = [q for q in f if all(q-j*stride in fs for j in range(k))]
            future = [sum(q+j*stride in fs for j in range(1, 13)) for q in expected]
            got, q = frame_support(f, k, stride)
            np.testing.assert_array_equal(q, expected)
            assert got == dict(past_eligible=len(q), complete_future12=future.count(12),
                               partial_future12=sum(0 < v < 12 for v in future), no_future12=future.count(0))


def test_partial_future_beyond_gap_is_not_called_absent():
    f = np.r_[np.arange(8), [10, 40]]
    counts, q = frame_support(f, 8, 1)
    assert q.tolist() == [7] and counts['partial_future12'] == 1
    assert counts['no_future12'] == 0


def test_reader_prefix_invariant_and_current_agent_population():
    values, _ = read_raw_csv(csv([row(i) for i in range(100)] + [row(100, 2)]))
    expected = past_scene(values, 84)
    assert list(expected) == [1]
    changed = values.copy()
    changed['x_min'][changed['frame'] > 84] = 1e9
    for variant in (changed, values[values['frame'] <= 84]):
        actual = past_scene(variant, 84)
        assert actual.keys() == expected.keys()
        for agent in expected:
            for key in expected[agent]:
                np.testing.assert_array_equal(expected[agent][key], actual[agent][key])
    np.testing.assert_array_equal(expected[1]['velocity_causal_fd'][1:, 0], 1)
    assert expected[1]['valid'].all()


@pytest.mark.parametrize('name', ['/root.csv', '../root.csv', 'a/../../b.csv', 'a\\b.csv'])
def test_unsafe_archive_paths(name):
    with pytest.raises(ValueError):
        safe_zip_member(zipfile.ZipInfo(name))


def test_symlink_and_encryption_refused():
    info = zipfile.ZipInfo('a.csv')
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(ValueError):
        safe_zip_member(info)
    info.external_attr = 0
    info.flag_bits = 1
    with pytest.raises(ValueError):
        safe_zip_member(info)


def test_ambiguous_publisher_coordinate_is_retained_not_guessed(tmp_path, monkeypatch):
    from scripts import audit_m3w_european_squares as runner
    monkeypatch.setattr(runner, 'RAW', tmp_path)
    text = 'No.,City,Country,Date,timeslot,lat,long\n9,Varberg,Sweden,20250101,noon,"57,106,027","12,252,087"\n'
    for kind in ('comparative', 'season'):
        (tmp_path / f'stats_{kind}.csv').write_text(text)
    result = runner.metadata_sites()
    assert len(result) == 2
    assert result[0]['latitude'] is None
    assert result[0]['unparsed_latitude_values'] == ['57,106,027']


def test_conflicting_square_name_not_silently_assigned(tmp_path, monkeypatch):
    from scripts import audit_m3w_european_squares as runner
    monkeypatch.setattr(runner, 'RAW', tmp_path)
    text = 'No.,City,Country,Date,timeslot,lat,long\n9,Varberg,Sweden,20250101,noon,57.1,12.2\n9,Biberach,Germany,20250102,noon,48.1,9.8\n'
    for kind in ('comparative', 'season'):
        (tmp_path / f'stats_{kind}.csv').write_text(text)
    result = runner.metadata_sites()
    assert result[0]['site_identity_conflict'] and result[0]['city'] is None
    assert result[0]['latitude'] is None and result[0]['longitude'] is None
    assert len(result[0]['city_country_variants']) == 2
