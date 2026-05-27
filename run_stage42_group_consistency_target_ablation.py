from __future__ import annotations

from src.stage42_group_consistency_target_ablation import run_stage42_group_consistency_target_ablation


def main() -> None:
    payload = run_stage42_group_consistency_target_ablation(refresh_readmes=True)
    gate = payload["stage42_et_gate"]
    selection = payload["group_schema_ablation"]["selection"]
    print(
        "Stage42-ET group-consistency target ablation complete: "
        f"{gate['passed']}/{gate['total']} {gate['verdict']} "
        f"target={selection['selected_target_for_next_stage']}"
    )


if __name__ == "__main__":
    main()
