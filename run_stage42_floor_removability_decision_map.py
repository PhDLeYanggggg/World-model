from __future__ import annotations

from src.stage42_floor_removability_decision_map import run_stage42_floor_removability_decision_map


def main() -> None:
    payload = run_stage42_floor_removability_decision_map(refresh_readmes=True)
    gate = payload["stage42_en_gate"]
    print(f"Stage42-EN floor removability decision map complete: {gate['passed']}/{gate['total']} {gate['verdict']}")


if __name__ == "__main__":
    main()
