from __future__ import annotations

from src.stage19_pipeline import generate_simulation_data


def export_simulation(quick: bool = True):
    return generate_simulation_data(quick=quick)

