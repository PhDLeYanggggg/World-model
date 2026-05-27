from __future__ import annotations

from src.stage42_post_ea_paper_refresh import run_stage42_post_ea_paper_refresh


def main() -> None:
    payload = run_stage42_post_ea_paper_refresh(refresh_readmes=True)
    gate = payload["stage42_eb_gate"]
    print(
        "Stage42-EB post-EA paper refresh complete: "
        f"{gate['passed']}/{gate['total']} {gate['verdict']}"
    )


if __name__ == "__main__":
    main()
