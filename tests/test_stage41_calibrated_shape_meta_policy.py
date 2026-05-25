import numpy as np

from src import stage41_calibrated_shape_meta_policy as calibrated


def test_affine_log_calibration_recovers_scale_shift():
    pred = np.linspace(1.0, 16.0, 32, dtype=np.float64)
    true = np.expm1(0.5 + 1.2 * np.log1p(pred))
    cal = calibrated._fit_affine_log_calibration(pred, true)
    out = calibrated._apply_affine_log_calibration(pred, cal)
    assert np.corrcoef(out, true)[0, 1] > 0.99


def test_calibrator_modes_are_finite():
    pack = {"horizon": np.asarray([50, 100], dtype=np.int16)}
    pred = {
        "bridge": np.asarray([1.0, 2.0], dtype=np.float64),
        "old_shape": np.asarray([0.8, 1.9], dtype=np.float64),
        "gain_gate": np.asarray([0.9, 1.8], dtype=np.float64),
    }
    cal = {
        "global": {"a": 1.0, "b": 0.0, "rows": 2},
        "by_source": {source: {"a": 1.0, "b": 0.0, "rows": 2} for source in calibrated.SOURCES},
        "by_source_horizon": {source: {"50": {"a": 1.0, "b": 0.0, "rows": 1}, "100": {"a": 1.0, "b": 0.0, "rows": 1}} for source in calibrated.SOURCES},
    }
    out = calibrated._apply_calibrator(pack, pred, cal, "source_horizon")
    assert all(np.isfinite(v).all() for v in out.values())
