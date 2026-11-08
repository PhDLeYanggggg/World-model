import pytest

from scripts.run_m3w_acme_hybrid_detector import rows_compare


def test_overlap_keeps_box_and_identity_discrepancies_separate():
    fields = [dict(camera_frame=4*i, frame_sha256=f"frame{i}", pts_error_ms=0.) for i in range(8)]
    cpu = [dict(**f, rows=[[0, 10., 20., 40., 60., .9, 1]]) for f in fields]
    mps = [dict(**f, rows=[[0, 10.001, 20., 40., 60., .90002, 1]]) for f in fields]
    mps[0]["rows"][0][0] = 1
    out = rows_compare(mps, cpu)
    assert out["frames"] == 8
    assert out["count_or_identity_mismatches"] == 1
    assert out["max_box_difference_px"] == pytest.approx(.001)
    assert out["max_confidence_difference"] == pytest.approx(.00002)
    mps[-1]["camera_frame"] += 1
    with pytest.raises(ValueError, match="identical frames"):
        rows_compare(mps, cpu)
