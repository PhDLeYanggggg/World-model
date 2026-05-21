from __future__ import annotations

from src.evaluation.stage9_benchmark import run_stage9_benchmark, write_stage9_benchmark


if __name__ == "__main__":
    payload = run_stage9_benchmark()
    write_stage9_benchmark(payload)
    print("variants:", [v["variant"] for v in payload["variants"]])
