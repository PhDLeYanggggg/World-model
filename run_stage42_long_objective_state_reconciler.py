from __future__ import annotations

from src.stage42_long_objective_state_reconciler import run_stage42_long_objective_state_reconciler


def main() -> None:
    payload = run_stage42_long_objective_state_reconciler(refresh_readmes=True)
    gate = payload["stage42_gr_gate"]
    print(
        "Stage42-GR long objective state reconciler complete: "
        f"{gate['passed']}/{gate['total']} {gate['verdict']}"
    )


if __name__ == "__main__":
    main()
