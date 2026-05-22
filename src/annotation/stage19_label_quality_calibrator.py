from __future__ import annotations


def calibrate_quality(raw_quality: str) -> str:
    if raw_quality == "gold_human":
        return "self_audited_silver"
    return raw_quality

