from __future__ import annotations

from src.stage42_source_terms_intake_validator_bridge import run_stage42_source_terms_intake_validator_bridge


def main() -> None:
    payload = run_stage42_source_terms_intake_validator_bridge(refresh_readmes=True)
    gate = payload["stage42_ei_gate"]
    print(f"Stage42-EI source terms intake validator bridge complete: {gate['passed']}/{gate['total']} {gate['verdict']}")


if __name__ == "__main__":
    main()
