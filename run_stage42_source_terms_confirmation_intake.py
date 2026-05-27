from __future__ import annotations

from src.stage42_source_terms_confirmation_intake import run_stage42_source_terms_confirmation_intake


def main() -> None:
    payload = run_stage42_source_terms_confirmation_intake(refresh_readmes=True)
    gate = payload["stage42_eh_gate"]
    print(f"Stage42-EH source terms confirmation intake complete: {gate['passed']}/{gate['total']} {gate['verdict']}")


if __name__ == "__main__":
    main()
