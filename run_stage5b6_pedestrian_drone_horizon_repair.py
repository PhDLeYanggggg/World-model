from __future__ import annotations

import json

from src.data.pedestrian_drone_horizon_repair import run_horizon_repair, write_outputs


def main() -> int:
    rows = write_outputs(run_horizon_repair())
    print(json.dumps({"sources_checked": len(rows), "ped_drone_t50_verified": sum(1 for r in rows if r["t50_verified"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

