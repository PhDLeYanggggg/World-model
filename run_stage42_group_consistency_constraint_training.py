from __future__ import annotations

from src.stage42_group_consistency_constraint_training import run_stage42_group_consistency_constraint_training


def main() -> None:
    payload = run_stage42_group_consistency_constraint_training(refresh_readmes=True)
    gate = payload["stage42_eu_gate"]
    decision = payload["deployment_decision"]["decision"]
    selected = payload["training"]["selected"]
    print(
        "Stage42-EU group-consistency constraint training complete: "
        f"{gate['passed']}/{gate['total']} {gate['verdict']} "
        f"variant={selected['variant']} decision={decision}"
    )


if __name__ == "__main__":
    main()
