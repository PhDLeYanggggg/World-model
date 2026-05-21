from __future__ import annotations

from src.evaluation.stage9_per_agent_baselines import run_stage9_baselines, write_stage9_baselines


if __name__ == "__main__":
    payload = run_stage9_baselines()
    write_stage9_baselines(payload)
    print(payload["datasets"].keys())
