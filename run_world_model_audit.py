from __future__ import annotations

from src.evaluation.world_model_self_audit import run_world_model_self_audit


def main() -> None:
    result = run_world_model_self_audit()
    print(f"world_model_self_audit: score={result['score']}/100 verdict={result['verdict']}")
    print("report=outputs/reports/world_model_expert_self_audit.md")


if __name__ == "__main__":
    main()
