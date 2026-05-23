from __future__ import annotations

from src.data_unification.stage21_sdd_converter import _horizon_counts


def test_stage21_sdd_horizon_counts_use_past_window():
    counts = _horizon_counts({1: 120, 2: 20})
    assert counts["t10"] == 101 + 1
    assert counts["t50"] == 61
    assert counts["t100"] == 11

