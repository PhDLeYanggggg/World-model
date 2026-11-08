import numpy as np
import pytest

from scripts.run_m3w_acme_held_camera_probe import association_frames, score


@pytest.mark.parametrize("factor", (1, 6))
def test_publisher_clock_mapping_preserves_detections(factor):
    frames = [dict(camera_frame=(10+4*k)*factor, rows=[[0, 1, 2, 3, 4, .9, 1]]) for k in range(8)]
    normalized = association_frames(frames, 10, factor)
    assert [row["camera_frame"] for row in normalized] == [3*(10+4*k) for k in range(8)]
    assert [row["rows"] for row in normalized] == [row["rows"] for row in frames]
    frames[-1]["camera_frame"] += 1
    with pytest.raises(ValueError, match="camera frame mismatch"):
        association_frames(frames, 10, factor)


def test_forecast_summary_weights_queries_not_rows_and_keeps_missing_out():
    labels = {}
    for key, start in [("a", 0), ("b", 0), ("c", 100), ("partial", 100)]:
        labels[key] = {start+4*(8+k): [0, 0] for k in range(11 if key == "partial" else 12)}
    paths = lambda distance: np.tile([distance, 0], (2, 12, 1))
    result = score(
        [[("a", paths(2)), ("b", paths(4))], [("c", paths(9)), ("partial", paths(80))]],
        [0, 100], labels,
    )
    assert result["complete_futures"] == 3
    assert result["scored_windows"] == 2
    assert result["query_equal_ade_fde_px"] == {"CV": [6., 6.], "OLS4": [6., 6.]}
