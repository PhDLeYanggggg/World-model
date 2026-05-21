from __future__ import annotations

import json

from src.evaluation.hard_reliability_audit import run_audit, write_outputs


def main() -> int:
    rows = write_outputs(run_audit())
    print(json.dumps({"datasets": len(rows), "official_hard_gate_eligible": sum(1 for r in rows if r["hard_subset_is_gate_eligible"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

