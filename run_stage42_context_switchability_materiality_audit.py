from __future__ import annotations

from src.stage42_context_switchability_materiality_audit import run_stage42_context_switchability_materiality_audit


def main() -> None:
    payload = run_stage42_context_switchability_materiality_audit(refresh_readmes=True)
    gate = payload["stage42_ee_gate"]
    print(
        "Stage42-EE context switchability materiality audit complete: "
        f"{gate['passed']}/{gate['total']} {gate['verdict']}"
    )


if __name__ == "__main__":
    main()
