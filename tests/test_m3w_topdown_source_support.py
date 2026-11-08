import numpy as np
import pytest

from scripts.audit_m3w_topdown_sources import mot_tracks, opendd_record
from src.data_unification.m3w_topdown_source_support import grid_indices, support


def test_support_separates_past_eligibility_and_future():
    complete = list(range(20))
    no_future = list(range(8))
    endpoint_only = [*range(8), 19]
    r = support([complete, no_future, endpoint_only], 20, 1)
    assert r["past_eligible"] == 3
    assert r["endpoint"] == 2
    assert r["complete12"] == r["partial12"] == r["no_future"] == 1
    assert r["multiagent_queries"] == 1 and r["multiagent_complete_queries"] == 0
    assert support([np.arange(19)], 19, 1)["past_eligible"] == 0


def test_interleaved_stride_uses_exact_sample_indices():
    r = support([np.arange(0, 39, 2), np.arange(39)], 39, 2)
    assert r["complete12"] == 2
    assert r["nonoverlapping_multiagent_complete"] == 1
    with pytest.raises(ValueError, match="sorted unique"):
        support([[0, 0]], 20, 1)


def test_verified_export_clock_not_rounded_30hz():
    expected = np.arange(100)
    index, residual = grid_indices(np.round(expected * 1001./30000., 6))
    np.testing.assert_array_equal(index, expected)
    assert residual < 5.1e-7
    with pytest.raises(ValueError, match="Off-grid"):
        grid_indices(expected / 30.)


def test_mot_support_ignores_position_and_rejects_duplicate_observation():
    lines = ["2 5 x y w h 1 -1 -1", "1 5 x y w h 1 -1 -1"]
    assert mot_tracks(lines, 2) == [[0, 1]]
    with pytest.raises(ValueError, match="Duplicate"):
        mot_tracks([*lines, lines[0]], 2)


def test_opendd_uses_real_timestamps_not_row_number():
    import sqlite3

    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE rdb1_1 (TIMESTAMP REAL, OBJID INTEGER, CLASS TEXT)")
        for i in range(20):
            for obj in (1, 2):
                if i != 12 or obj == 2:
                    db.execute("INSERT INTO rdb1_1 VALUES (?, ?, 'Pedestrian')",
                               (round(i * 1001 / 30000, 6), obj))
        r = opendd_record(db, "rdb1_1")["classes"]["Pedestrian"]["stride_1"]
        assert r["past_eligible"] == 2 and r["complete12"] == 1
        assert r["multiagent_complete_queries"] == 0
