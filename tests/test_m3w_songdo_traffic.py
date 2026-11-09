import numpy as np
import pandas as pd

from scripts import audit_m3w_songdo_traffic as a


def test_clock_preserves_missing_frames_and_accepts_small_jitter():
    ticks = np.array([0, 1, 2, 4, 5, 7, 8])
    t = 1000 + np.rint(ticks * a.DT_MS).astype(int) + np.array([0, -1, 2, 1, -2, 1, 0])
    frames, report = a.clock(np.r_[t, t])
    assert report['clock_supported']
    np.testing.assert_array_equal(frames, np.r_[ticks, ticks])
    assert not a.clock(np.arange(150) * 38)[1]['clock_supported']


def test_affine_sensitivity_recovers_small_drift_but_not_ambiguous_increments():
    ticks = np.arange(10000)
    t = np.rint(ticks * (a.DT_MS + .0033)).astype(int)
    f, report = a.clock(t)
    assert not report['clock_supported']
    np.testing.assert_array_equal(f, ticks)
    assert a.affine_clock(t, f)['accepted']
    bad = np.array([0, 10, 33, 67])
    f2, _ = a.clock(bad)
    assert not a.affine_clock(bad, f2)['accepted']


def test_pair_coverage_keeps_history_roster_and_missing_future():
    # Two complete tracks and one target that disappears just after its prefix.
    ids = np.repeat([1, 2, 3], [20, 20, 8])
    frames = np.r_[np.arange(20), np.arange(20), np.arange(8)]
    result = a.coverage(ids, frames, 19, stride=1)
    assert result['eligible_agent_windows'] == 3
    assert result['complete_agent_windows'] == 2
    assert result['nonoverlapping_queries'] == 1
    assert result['nonoverlapping_queries_with_complete_pair'] == 1
    assert result['nonoverlapping_full_roster'] == 0
    assert result['nonoverlapping_complete_pairs'] == 1
    assert result['nonoverlapping_eligible_pairs'] == 3


def test_fast_coverage_matches_exhaustive_rosters_on_a_gappy_grid():
    rng = np.random.default_rng(90)
    for _ in range(20):
        seen = rng.random((5, 90)) > .11
        ids, frames = np.nonzero(seen)
        stat = a.coverage(ids, frames, 89, stride=3, history=4, future=6)
        starts = np.arange(90 - 3 * 9)
        index = starts[:, None] + 3 * np.arange(10)
        past = seen[:, index[:, :4]].all(axis=2).sum(axis=0)
        full = seen[:, index].all(axis=2).sum(axis=0)
        assert stat['eligible_agent_windows'] == past.sum()
        assert stat['complete_agent_windows'] == full.sum()
        assert stat['eligible_pairs'] == (past * (past - 1) // 2).sum()
        assert stat['complete_pairs'] == (full * (full - 1) // 2).sum()


def test_fragments_and_drones_are_not_joined_into_a_single_roster():
    ts = pd.to_timedelta(np.r_[np.rint(np.arange(20) * a.DT_MS), 5000.], unit='ms')
    text = [str(t).removeprefix('0 days ') for t in ts]
    table = pd.DataFrame(dict(Local_Time=text * 2, Vehicle_ID=[1] * 42, Drone_ID=[1] * 21 + [2] * 21,
                              Vehicle_Class=0, Visibility=1))
    r = a.session(table, 'x', 'A')
    assert len(r['drones']) == 4
    assert r['vehicle_ids_on_multiple_drones'] == 1
    assert sum(x['clock_supported'] for x in r['drones']) == 2
    assert all(x.get('any_visibility', {}).get('multiagent_queries', 0) == 0 for x in r['drones'])


def test_same_drone_overlap_across_sites_flags_potential_shared_video():
    record = dict(site='A', drone=1, fragment=0, session='A_AM1', first_ms=10, last_ms=30)
    r = a.overlap([record, dict(record, site='B', session='B_AM1', first_ms=29),
                   dict(record, site='C', session='C_AM1', drone=2)])
    assert r == [dict(drone=1, sessions=['A_AM1', 'B_AM1'], fragments=[0, 0], overlap_ms=1)]
