import numpy as np

from src.stage41_route_physical_group_consistency import _delta


def test_delta_reports_positive_lift() -> None:
    out = _delta({"all_improvement": 0.2, "t50_improvement": 0.1}, {"all_improvement": 0.15, "t50_improvement": 0.2})
    assert np.isclose(out["all_delta"], 0.05)
    assert np.isclose(out["t50_delta"], -0.1)


def test_delta_defaults_missing_metrics() -> None:
    out = _delta({"hard_failure_improvement": 0.3}, {})
    assert np.isclose(out["hard_delta"], 0.3)
    assert np.isclose(out["easy_delta"], 0.0)
