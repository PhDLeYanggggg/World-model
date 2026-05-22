from __future__ import annotations

from typing import Dict, List


def encode_goals(goals: List[Dict[str, object]]) -> Dict[str, object]:
    return {"goal_count": len(goals), "has_candidate_goals": bool(goals)}

