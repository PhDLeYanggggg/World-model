from __future__ import annotations

from src.stage42_constraint_aware_composer import run_stage42_constraint_aware_composer


def main() -> None:
    payload = run_stage42_constraint_aware_composer(refresh_readmes=True)
    gate = payload["stage42_ev_gate"]
    selected = payload["composer"]["selected"]
    decision = payload["deployment_decision"]["decision"]
    print(
        "Stage42-EV constraint-aware composer complete: "
        f"{gate['passed']}/{gate['total']} {gate['verdict']} "
        f"mode={selected['mode']} decision={decision}"
    )


if __name__ == "__main__":
    main()
