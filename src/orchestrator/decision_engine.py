from __future__ import annotations

from typing import Any, Dict, List


def decide_next_actions(state: Dict[str, Any], failures: Dict[str, List[str]], mode: str = "quick") -> Dict[str, Any]:
    actions: List[Dict[str, str]] = []
    blockers: List[str] = list(failures.get("user_blockers", []))

    if not state.get("pedestrian_long_horizon_ready"):
        actions.append(
            {
                "name": "data_acquisition",
                "priority": "P0",
                "reason": "No verified pedestrian/drone t+50/t+100 source is ready.",
                "command": "python scripts/auto_find_and_prepare_datasets.py --dry-run",
            }
        )
        blockers.append("Provide local SDD/OpenTraj/full TrajNet++ path if license-gated download cannot be automated.")

    if not state.get("scene_annotation_ready"):
        actions.append(
            {
                "name": "annotation_tasks",
                "priority": "P0",
                "reason": "Scene/goal labels are insufficient or not human/silver-confirmed.",
                "command": "python scripts/auto_validate_annotations.py",
            }
        )

    deterministic_ready = bool(state.get("deterministic_ready"))
    if not deterministic_ready:
        actions.append(
            {
                "name": "deterministic_repair",
                "priority": "P1",
                "reason": "Learned deterministic model has not beaten strongest causal baselines.",
                "command": "python scripts/auto_benchmark.py --quick",
            }
        )

    if not state.get("latent_generative_ready"):
        actions.append(
            {
                "name": "latent_blocked",
                "priority": "guardrail",
                "reason": "Latent generative training remains forbidden until deterministic gates pass.",
                "command": "do_not_run_latent_generative",
            }
        )

    if mode == "data-first":
        actions.sort(key=lambda row: 0 if row["name"] == "data_acquisition" else 1)
    elif mode == "annotation-first":
        actions.sort(key=lambda row: 0 if row["name"] == "annotation_tasks" else 1)
    elif mode == "train-deterministic":
        actions.sort(key=lambda row: 0 if row["name"] == "deterministic_repair" else 1)

    return {
        "actions": actions,
        "blockers_requiring_user": sorted(set(blockers)),
        "recommended_next_action": actions[0] if actions else None,
    }

