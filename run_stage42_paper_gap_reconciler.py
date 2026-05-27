from __future__ import annotations

from src.stage42_paper_gap_reconciler import run_stage42_paper_gap_reconciler


def main() -> None:
    payload = run_stage42_paper_gap_reconciler(refresh_readmes=True)
    gate = payload["stage42_gs_gate"]
    print(
        "Stage42-GS paper gap reconciler complete: "
        f"{gate['passed']}/{gate['total']} {gate['verdict']}"
    )


if __name__ == "__main__":
    main()
