from __future__ import annotations

from typing import Any, Dict

from src.search.stage13_deterministic_search import run_stage13_search
from src.stage14_pipeline import write_json, write_md


def train_stage14_multimodal(max_trials_per_family: int = 1, max_iterations: int | None = 10) -> Dict[str, Any]:
    result = run_stage13_search(
        max_trials_per_family=max_trials_per_family,
        max_iterations=max_iterations,
        allow_training=True,
    )
    result["stage14_training_note"] = (
        "Deterministic bounded-residual search reused for Stage14 multimodal scaffold. "
        "Visual/raster branches are represented in data/scene packs but are not proven effective yet."
    )
    write_json("outputs/reports/stage14_multimodal_training_report.json", result)
    write_md(
        "outputs/reports/stage14_multimodal_training_report.md",
        [
            "# Stage 14 Multimodal Deterministic Training Report",
            "",
            f"- executed_training: `{result.get('executed_training')}`",
            f"- trial_count: `{result.get('trial_count')}`",
            f"- episode_count: `{result.get('episode_count')}`",
            "- latent_enabled: `False`",
            "- smc_enabled: `False`",
            "",
            result["stage14_training_note"],
        ],
    )
    return result

