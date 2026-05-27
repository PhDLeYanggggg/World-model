from __future__ import annotations

from src.stage42_interaction_occupancy_target_selection import run_stage42_interaction_occupancy_target_selection


def main() -> None:
    payload = run_stage42_interaction_occupancy_target_selection(refresh_readmes=True)
    gate = payload["stage42_es_gate"]
    print(f"Stage42-ES interaction/occupancy target selection complete: {gate['passed']}/{gate['total']} {gate['verdict']}")


if __name__ == "__main__":
    main()
