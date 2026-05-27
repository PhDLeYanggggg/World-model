from __future__ import annotations

from src.stage42_source_terms_gap_audit import run_stage42_source_terms_gap_audit


def main() -> None:
    payload = run_stage42_source_terms_gap_audit(refresh_readmes=True)
    gate = payload["stage42_ef_gate"]
    print(f"Stage42-EF source terms gap audit complete: {gate['passed']}/{gate['total']} {gate['verdict']}")


if __name__ == "__main__":
    main()
