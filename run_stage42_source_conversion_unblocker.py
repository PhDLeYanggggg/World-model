from __future__ import annotations

from src.stage42_source_conversion_unblocker import run_stage42_source_conversion_unblocker


def main() -> None:
    payload = run_stage42_source_conversion_unblocker(refresh_readmes=True)
    gate = payload["stage42_ed_gate"]
    print(
        "Stage42-ED source conversion unblocker complete: "
        f"{gate['passed']}/{gate['total']} {gate['verdict']}"
    )


if __name__ == "__main__":
    main()
