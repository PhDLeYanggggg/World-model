from __future__ import annotations

from src.stage42_group_consistency_contribution_audit import run_stage42_group_consistency_contribution_audit


def main() -> None:
    payload = run_stage42_group_consistency_contribution_audit(refresh_readmes=True)
    gate = payload["stage42_ec_gate"]
    print(
        "Stage42-EC group-consistency contribution audit complete: "
        f"{gate['passed']}/{gate['total']} {gate['verdict']}"
    )


if __name__ == "__main__":
    main()
