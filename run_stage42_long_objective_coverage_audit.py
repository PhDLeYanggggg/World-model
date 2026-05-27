from __future__ import annotations

from src.stage42_long_objective_coverage_audit import run_stage42_long_objective_coverage_audit


def main() -> None:
    payload = run_stage42_long_objective_coverage_audit(refresh_readmes=True)
    gate = payload["stage42_ek_gate"]
    print(
        "Stage42-EK long objective coverage audit complete: "
        f"{gate['passed']}/{gate['total']} {gate['verdict']}"
    )


if __name__ == "__main__":
    main()
