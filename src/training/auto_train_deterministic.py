from __future__ import annotations

from src.models.auto_scene_goal_interaction_model import model_spec
from src.orchestrator.research_state import write_json, write_md


TRAINING_ORDER = [
    "strongest causal baselines",
    "alpha-only baseline failure gate",
    "bounded residual without scene/goal",
    "scene-only",
    "goal-only",
    "interaction-only",
    "scene+goal",
    "scene+interaction",
    "goal+interaction",
    "full scene+goal+interaction",
    "hard/failure fine-tuned",
    "long-horizon fine-tuned",
]


def write_deterministic_training_plan() -> dict:
    payload = {
        "model_spec": model_spec(),
        "training_order": TRAINING_ORDER,
        "executed_training": False,
        "reason": "Auto quick loop plans deterministic repair but does not launch heavy training by default.",
    }
    write_json("outputs/reports/auto_deterministic_training_report.json", payload)
    write_md(
        "outputs/reports/auto_deterministic_training_report.md",
        [
            "# Auto Deterministic Training Report",
            "",
            f"- executed_training: `{payload['executed_training']}`",
            f"- prediction_form: `{payload['model_spec']['prediction_form']}`",
            "",
            "## Training Order",
            "",
            *[f"- {item}" for item in TRAINING_ORDER],
        ],
    )
    write_md(
        "outputs/reports/auto_deterministic_ablation_report.md",
        ["# Auto Deterministic Ablation Report", "", "- Not run in quick loop. Use `scripts/auto_train_deterministic.py --execute` after reviewing gates."],
    )
    return payload

