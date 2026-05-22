from __future__ import annotations

from typing import Dict, List


def build_legal_download_plan(rows: List[Dict]) -> List[Dict]:
    plan = []
    for row in rows:
        can_auto = row.get("can_download_automatically") is True and not row.get("requires_login") and not row.get("requires_application")
        if can_auto:
            action = "eligible_for_manual_execute_download"
            command = f"python scripts/auto_find_and_prepare_datasets.py --dataset '{row['dataset_name']}' --execute-download"
        else:
            action = "requires_user_path_or_license_review"
            command = "prepare local path after accepting official terms"
        plan.append(
            {
                "dataset_name": row["dataset_name"],
                "action": action,
                "command_or_next_step": command,
                "legal_notes": row.get("legal_notes", ""),
            }
        )
    return plan

