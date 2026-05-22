from __future__ import annotations

from src.stage19_pipeline import verify_topdown_data


def verify_full_eth_ucy():
    return verify_topdown_data(quick=True)

