from __future__ import annotations

from src.stage19_pipeline import verify_egocentric_data


def verify_epic_kitchens():
    return verify_egocentric_data(quick=True)

