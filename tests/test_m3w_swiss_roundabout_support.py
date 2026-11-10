from io import StringIO

import numpy as np
import pytest

from src.data_unification.m3w_swiss_roundabout_support import clock_token, coverage, grid, roster, scan


def test_rounded_clock_and_missing_track_do_not_invent_a_tick():
    clock = np.rint(np.arange(900)*100/30).astype(int)
    tick, width = grid(clock, 30)
    np.testing.assert_array_equal(tick, np.arange(900))
    assert width < 10000.
    assert grid(clock, 30000/1001)[1] > 10000.
    assert clock_token('07:37:58.430000') == 2747843


def test_roster_unchanged_when_chosen_query_loses_all_future_labels():
    common = np.arange(8)
    tracks = [np.arange(20), np.arange(16), common]
    before = coverage(tracks, 20, 1)
    after = coverage([common]*3, 20, 1)
    assert before['queries'] == after['queries'] == 1
    assert before['target_rows'] == after['target_rows'] == 3
    assert before['complete'] == 1 and before['partial'] == before['absent'] == 1
    assert before['observed_pair_step_slots'] == 8 and before['pair_step_slots'] == 36
    assert after['absent'] == 3


def test_stride_with_nonzero_phase_and_nonoverlap():
    t = np.arange(1, 90, 3)
    queries, members = roster([t, t], 90, 3)
    np.testing.assert_array_equal(queries, [22])
    assert all(m.tolist() == [0] for m in members)


def test_scan_rejects_duplicate_clock_tokens_without_decoding_geometry():
    header = 'track_id,type,lon,lat,time\n'
    r = '1,Car,X,Y,01:00:00.03\n'
    _, stats = scan(StringIO(header+r))
    assert stats['tracks'] == 1
    with pytest.raises(ValueError, match='Duplicate'):
        scan(StringIO(header+r+r))


def test_clock_reset_is_not_bridged_by_the_roster():
    rows = ['track_id,type,lon,lat,time']
    for offset in (0, 80):
        for tick in np.rint(np.arange(18)*100/30).astype(int)+offset:
            for agent in (1, 2):
                rows.append(f'{agent},Car,X,Y,00:00:{tick//100:02}.{tick%100:02}')
    segments, _ = scan(StringIO('\n'.join(rows)))
    assert [s['ticks'] for s in segments] == [18, 18]
    assert not any(coverage(list(s['tracks'].values()), s['ticks'], 1)['queries'] for s in segments)
