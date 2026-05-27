from __future__ import annotations

from src.stage42_official_source_link_audit import run_stage42_official_source_link_audit


def main() -> None:
    payload = run_stage42_official_source_link_audit(refresh_readmes=True)
    gate = payload["stage42_em_gate"]
    print(f"Stage42-EM official source link audit complete: {gate['passed']}/{gate['total']} {gate['verdict']}")


if __name__ == "__main__":
    main()
