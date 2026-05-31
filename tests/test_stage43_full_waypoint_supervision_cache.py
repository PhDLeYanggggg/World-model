from pathlib import Path

from src.stage43_full_waypoint_supervision_cache import run_full_waypoint_supervision_cache


def test_stage43_l_full_waypoint_supervision_cache_passes():
    payload = run_full_waypoint_supervision_cache()
    gate = payload["stage43_l_gate"]
    assert gate["verdict"] == "stage43_l_full_waypoint_supervision_cache_pass"
    assert gate["passed"] == gate["total"]
    assert gate["full_waypoint_supervised_training_ready"] is True


def test_stage43_l_writes_local_uncommitted_split_caches():
    payload = run_full_waypoint_supervision_cache()
    assert payload["cache_committed"] is False
    for split, row in payload["split_summaries"].items():
        assert Path(row["cache_path"]).exists(), split
        assert row["rows"] > 0
        assert row["full_waypoint_rows"] > 0
        assert row["max_endpoint_diff_last_waypoint"] <= 1e-4
    assert payload["source_overlap_counts"] == {"train_val": 0, "train_test": 0, "val_test": 0}
    assert payload["no_leakage"]["future_waypoint_input"] is False
    assert payload["no_leakage"]["future_waypoint_label_eval_only"] is True
