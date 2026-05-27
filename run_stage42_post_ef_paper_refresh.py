from __future__ import annotations

from src.stage42_post_ef_paper_refresh import run_stage42_post_ef_paper_refresh


def main() -> None:
    payload = run_stage42_post_ef_paper_refresh(refresh_readmes=True)
    gate = payload["stage42_eg_gate"]
    print(f"Stage42-EG post-EE/EF paper refresh complete: {gate['passed']}/{gate['total']} {gate['verdict']}")


if __name__ == "__main__":
    main()
