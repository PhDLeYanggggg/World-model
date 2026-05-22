from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.annotation.auto_annotation_orchestrator import run_annotation_orchestrator
from src.annotation.auto_annotation_upgrader import write_annotation_upgrade_report
from src.annotation.auto_annotation_validator import validate_latest_annotations


def main() -> None:
    orchestrator = run_annotation_orchestrator()
    validation = validate_latest_annotations()
    upgrade = write_annotation_upgrade_report()
    print({"review_queue": len(orchestrator["human_review_queue"]), "checks": len(validation["checks"])})


if __name__ == "__main__":
    main()
