from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List


HEARTBEAT_PATH = Path("outputs/reports/overnight_heartbeat.md")


def write_heartbeat(
    started_at: float,
    current_task: str,
    completed: List[Dict],
    failed: List[Dict],
    best_model: str = "unknown",
    best_t100_improvement: float | None = None,
    best_hard_improvement: float | None = None,
    gates_passed: List[str] | None = None,
    next_task: str = "unknown",
) -> None:
    HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    elapsed_h = (time.time() - started_at) / 3600.0
    lines = [
        "# Overnight Stage 13 Heartbeat",
        "",
        f"- current_time_unix: `{int(time.time())}`",
        f"- elapsed_hours: `{elapsed_h:.3f}`",
        f"- current_task: `{current_task}`",
        f"- completed_tasks: `{len(completed)}`",
        f"- failed_tasks: `{len(failed)}`",
        f"- best_model_so_far: `{best_model}`",
        f"- best_eth_ucy_ewap_t100_improvement: `{best_t100_improvement}`",
        f"- best_hard_failure_improvement: `{best_hard_improvement}`",
        f"- gates_passed_so_far: `{gates_passed or []}`",
        "- latent_blocked: `True`",
        "- smc_blocked: `True`",
        f"- next_task: `{next_task}`",
    ]
    HEARTBEAT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

