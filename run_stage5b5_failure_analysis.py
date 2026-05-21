from __future__ import annotations

import json

from src.evaluation.failure_case_miner import mine_failures, write_failure_report


def main() -> int:
    failures = mine_failures()
    write_failure_report(failures)
    print(json.dumps({"failure_count": len(failures), "failures": failures}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
