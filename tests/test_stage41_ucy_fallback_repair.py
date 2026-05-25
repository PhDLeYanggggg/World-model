import numpy as np

from src import stage41_ucy_fallback_repair as ufr


def test_calibration_mask_uses_train_only_domain_rows():
    data = {"domain": np.asarray(["UCY", "ETH_UCY", "UCY", "UCY", "TrajNet", "UCY"])}
    mask = ufr._calibration_mask(data, "UCY")
    assert mask.tolist() == [True, False, False, False, False, True]


def test_domain_counts_reports_horizon_distribution():
    data = {
        "domain": np.asarray(["A", "A", "B", "B", "B"]),
        "horizon": np.asarray([10, 50, 10, 100, 100]),
    }
    counts = ufr._domain_counts(data)
    assert counts["A"]["rows"] == 2
    assert counts["A"]["horizons"]["50"] == 1
    assert counts["B"]["horizons"]["100"] == 2
