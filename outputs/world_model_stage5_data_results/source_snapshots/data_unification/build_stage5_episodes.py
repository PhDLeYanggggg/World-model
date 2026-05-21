from __future__ import annotations

from pathlib import Path
from typing import Dict

from src.data_unification.convert_to_world_state import convert_real_dataset, write_conversion_report
from src.data_unification.episode_builder import build_episodes_from_world_state


def build_stage5_real_dataset(dataset: str, data_path: str, quick: bool = True) -> Dict:
    table, meta = convert_real_dataset(dataset, data_path, quick=quick)
    write_conversion_report(dataset, table, meta)
    summary = build_episodes_from_world_state(table, dataset, quick=quick)
    Path("outputs/reports/stage5_data").mkdir(parents=True, exist_ok=True)
    return {"world_state_rows": int(len(table)), "episode_summary": summary, "metadata": meta}
