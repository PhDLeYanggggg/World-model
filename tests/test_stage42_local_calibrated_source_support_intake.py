from src import stage42_local_calibrated_source_support_intake as jn


def test_wildtrack_position_id_to_metric_grid():
    assert jn._wildtrack_xy(0) == (-3.0, -9.0)
    assert jn._wildtrack_xy(481) == (-2.975, -8.975)


def test_track_stats_counts_horizons():
    points = [("toy", 1, i, float(i), 0.0) for i in range(60)]
    stats = jn._track_stats(points)
    assert stats["point_rows"] == 60
    assert stats["t10_rows"] == 50
    assert stats["t50_rows"] == 10
    assert stats["t100_rows"] == 0


def test_license_excerpt_detects_missing_license():
    assert jn._license_excerpt("hello") == "no_license_section_found"
