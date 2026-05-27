from __future__ import annotations

from src.stage42_guarded_source_conversion_launcher import run_stage42_guarded_source_conversion_launcher


def main() -> None:
    payload = run_stage42_guarded_source_conversion_launcher(refresh_readmes=True)
    gate = payload["stage42_ej_gate"]
    print(
        "Stage42-EJ guarded source conversion launcher complete: "
        f"{gate['passed']}/{gate['total']} {gate['verdict']}"
    )


if __name__ == "__main__":
    main()
