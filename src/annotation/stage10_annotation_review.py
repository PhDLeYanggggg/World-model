from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def mark_reviewed(path: str | Path, reviewer_id: str, quality: str) -> Path:
    """Mark an annotation as human-reviewed after a person has edited/confirmed it."""
    if quality not in {"gold_human", "silver_human_confirmed"}:
        raise ValueError("Human review can only promote to gold_human or silver_human_confirmed.")
    p = Path(path)
    ann = json.loads(p.read_text(encoding="utf-8"))
    ann["annotation_quality"] = quality
    ann["reviewer_id"] = reviewer_id
    ann["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    ann["requires_human_review"] = False
    p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
    return p
