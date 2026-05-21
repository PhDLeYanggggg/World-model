from __future__ import annotations

import json

from src.data.stage6_pedestrian_drone_audit import run_audit, write_outputs


def main() -> int:
    rows = write_outputs(run_audit())
    print(json.dumps({"sources": len(rows), "verified_t50_or_t100": sum(1 for r in rows if r["eligible_for_pedestrian_drone_long_horizon_gate"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

