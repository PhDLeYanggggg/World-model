import hashlib

import numpy as np
import pytest

from scripts.audit_m3w_acme_prefix_sources import audit_tables, track_clocks, verify_blob
from scripts.audit_m3w_acme_video_alignment import contained_pairs


def test_explicit_gaps_are_not_filled_by_row_ordinal():
    record = dict(traj_start=0, trajectories=[[0, 0]]*40,
                  frame_number=list(range(8)) + list(range(20, 52)))
    result = audit_tables({"human1": record, "human2": record})
    strict = result["strict"]["grids"]["1"]["disjoint_past_selected"]
    ordinal = result["ordinal_diagnostic_not_validated"]["grids"]["1"]["disjoint_past_selected"]
    assert result["verified_gapped_tracks"] == 2
    assert result["ordinal_moved_observations_among_verified_tracks"] == 64
    assert strict["queries"] == ordinal["queries"] == 2
    assert strict["complete12"] == 2
    assert ordinal["complete12"] == 4


@pytest.mark.parametrize("frames,status", [
    (None, "missing_explicit_clock"),
    ([2], "clock_length_mismatch"),
    ([2, 2], "invalid_explicit_clock"),
    ([3, 4], "start_clock_disagreement"),
])
def test_ambiguous_clock_never_reaches_strict_input(frames, status):
    assert track_clocks(dict(traj_start=2, trajectories=[[0, 0]]*2, frame_number=frames)) == (
        status, (), range(2, 4), 2,
    )


def test_lfs_and_git_objects_are_checked_against_actual_bytes():
    data = b"source sample"
    lfs = hashlib.sha256(data).hexdigest()
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    assert verify_blob(data, {"size": len(data), "lfs": {"oid": lfs}}) == lfs
    assert verify_blob(data, {"size": len(data), "oid": blob}) == lfs
    with pytest.raises(ValueError, match="Git blob hash mismatch"):
        verify_blob(b"x"*len(data), {"size": len(data), "oid": blob})


def test_box_containment_never_counts_two_labels_for_one_detection():
    points = np.array([[1., 1.], [2., 2.], [10., 10.]])
    boxes = np.array([[0., 0., 3., 3.], [9., 9., 11., 11.]])
    assert contained_pairs(points, boxes) == 2
    assert contained_pairs(points, boxes[:0]) == 0
