from __future__ import annotations

from src.evaluation.stage8p5_goalbench_gold_v2 import build_goalbench_gold_v2, write_goalbench_v2


if __name__ == "__main__":
    payload = build_goalbench_gold_v2()
    write_goalbench_v2(payload)
    print({k: v for k, v in payload.items() if k != "records"})
