from __future__ import annotations

from typing import Any, Dict, List


def build_task_plan(decision: Dict[str, Any], max_steps: int = 1) -> List[Dict[str, Any]]:
    tasks = []
    for action in decision.get("actions", []):
        if action["name"] in {"latent_blocked"}:
            continue
        tasks.append(
            {
                "task": action["name"],
                "priority": action["priority"],
                "reason": action["reason"],
                "command": action["command"],
                "status": "planned",
            }
        )
        if len(tasks) >= max_steps:
            break
    return tasks


def plan_to_markdown(tasks: List[Dict[str, Any]]) -> List[str]:
    if not tasks:
        return ["No executable task selected. Guardrails may be blocking progress."]
    lines = []
    for idx, task in enumerate(tasks, 1):
        lines += [
            f"{idx}. `{task['task']}` ({task['priority']})",
            f"   - reason: {task['reason']}",
            f"   - command: `{task['command']}`",
            f"   - status: {task['status']}",
        ]
    return lines

