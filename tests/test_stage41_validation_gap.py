import numpy as np

from src import stage41_validation_gap as gap


def test_stage41_validation_gap_stats_empty() -> None:
    rows = {
        "candidate_fde": np.zeros((0, 3), dtype=float),
        "floor_fde": np.zeros(0, dtype=float),
        "horizon": np.zeros(0, dtype=int),
        "hard": np.zeros(0, dtype=bool),
        "easy": np.zeros(0, dtype=bool),
        "failure": np.zeros(0, dtype=bool),
        "source_file": np.asarray([], dtype="U1"),
        "scene_id": np.asarray([], dtype="U1"),
    }
    stats = gap._stats(rows, np.zeros(0, dtype=bool))
    assert stats["rows"] == 0
    assert stats["oracle_headroom_t50"] == 0.0


def test_stage41_validation_gap_stats_t50_oracle() -> None:
    rows = {
        "candidate_fde": np.asarray([[10.0, 8.0, 11.0], [10.0, 9.0, 12.0]]),
        "floor_fde": np.asarray([10.0, 10.0]),
        "horizon": np.asarray([50, 50]),
        "hard": np.asarray([True, False]),
        "easy": np.asarray([False, True]),
        "failure": np.asarray([True, False]),
        "source_file": np.asarray(["a", "a"]),
        "scene_id": np.asarray(["s", "s"]),
    }
    stats = gap._stats(rows, np.asarray([True, True]))
    assert stats["t50_rows"] == 2
    assert round(stats["oracle_headroom_t50"], 3) == 0.150
    assert stats["best_candidate_distribution_t50"]["1"] == 2
