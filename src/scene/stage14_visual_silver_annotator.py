from __future__ import annotations

from typing import Dict


def visual_silver_metadata() -> Dict[str, str]:
    return {
        "annotation_quality": "ai_visual_silver",
        "note": "AI visual/rule-assisted annotation is not human gold and must be reviewed before gold claims.",
    }

