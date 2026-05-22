from __future__ import annotations


def consensus_passed(signals: dict) -> bool:
    return bool(signals.get("no_test_endpoint_leakage", False))

