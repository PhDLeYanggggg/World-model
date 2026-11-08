from io import StringIO

import pytest

from scripts.audit_m3w_external_causal_sources import read_tracks, timing_support


def test_future_presence_does_not_pick_the_first_disjoint_query():
    common = set(range(8)) | set(range(20, 40))
    partial = timing_support([common | {8}, common], (0, 39), 1)
    full = timing_support([set(range(40)), common], (0, 39), 1)
    assert partial["disjoint_first_ticks"] == full["disjoint_first_ticks"] == [0, 20]
    assert partial["disjoint_past_selected"] == {
        "queries": 2, "targets": 4, "complete12": 2, "partial12": 1,
        "no_future": 1, "complete_fraction": 0.5, "queries_with_two_complete": 1,
    }
    assert full["disjoint_past_selected"]["complete12"] == 3


def test_stride_preserves_missing_steps_and_rejects_short_histories():
    full = set(range(0, 80, 4))
    short = set(range(0, 28, 4))
    gapped = full - {36}
    results = timing_support([full, short, gapped], (0, 76), 4)
    assert results["disjoint_first_ticks"] == [0]
    assert results["disjoint_past_selected"]["targets"] == 2
    assert results["disjoint_past_selected"]["complete12"] == 1
    assert results["disjoint_past_selected"]["partial12"] == 1


@pytest.mark.parametrize("rows,reason", [
    ("A,0.1\nA,0.1\n", "Duplicate"),
    ("A,0.11\n", "off-grid"),
    ("A,inf\n", "Nonfinite"),
])
def test_foggy_clock_rejects_ambiguous_observations(rows, reason):
    with pytest.raises(ValueError, match=reason):
        read_tracks(StringIO("id,time\n" + rows), sind=False)


def test_sind_reads_clock_and_ids_without_position_fields():
    tracks, residual = read_tracks(StringIO(
        "track_id,frame_id,timestamp_ms,agent_type\nP0,1,100.1001001,pedestrian\n"
    ), sind=True)
    assert tracks == {"P0": {1}}
    assert residual < 1e-12
