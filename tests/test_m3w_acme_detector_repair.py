import pytest

from scripts.run_m3w_acme_detector_repair import verify_frames


def test_detector_change_requires_identical_clock_and_pixels():
    old = [dict(camera_frame=i*4, frame_sha256=f"image{i}", pts_error_ms=0., rows=[]) for i in range(8)]
    new = [{**f, "rows": [[1, 2, 3, 4]]} for f in old]
    verify_frames(new, old)
    new[-1]["frame_sha256"] = "other"
    with pytest.raises(ValueError, match="identical"):
        verify_frames(new, old)
