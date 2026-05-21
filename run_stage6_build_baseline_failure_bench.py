from __future__ import annotations

import json

from src.evaluation.baseline_failure_bench import build_failure_bench, write_outputs


def main() -> int:
    payload = write_outputs(build_failure_bench())
    print(json.dumps({k: v for k, v in payload.items() if k != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

