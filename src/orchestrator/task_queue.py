from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class QueueTask:
    name: str
    priority: str
    command: str
    family: str = ""
    allow_failure: bool = True


def build_stage13_task_queue(max_trials_per_family: int = 2, allow_training: bool = True) -> List[QueueTask]:
    training_flag = "" if allow_training else " --no-training"
    return [
        QueueTask("read_current_state", "P0", "python run_auto_world_model_loop.py --mode quick --max-steps 1"),
        QueueTask("check_stage13_allowed", "P0", "python scripts/auto_run_gates.py"),
        QueueTask("run_stage13_data_audit", "P1", "python scripts/auto_convert_available_datasets.py"),
        QueueTask(
            "stage13_deterministic_search",
            "P2",
            f"python scripts/run_stage13_deterministic_search.py --max-trials-per-family {max_trials_per_family}{training_flag}",
            family="all",
        ),
        QueueTask("run_stage13_gates", "P3", "python run_stage13_gates.py"),
        QueueTask("run_stage13_failure_miner", "P4", "python run_stage13_failure_miner.py"),
        QueueTask("auto_data_annotation_fallback", "P5", "python scripts/auto_find_and_prepare_datasets.py --dry-run"),
        QueueTask("update_readme", "P6", "python scripts/auto_update_readme_results.py"),
    ]

