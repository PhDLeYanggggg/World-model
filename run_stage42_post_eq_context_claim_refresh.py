from __future__ import annotations

from src.stage42_post_eq_context_claim_refresh import run_stage42_post_eq_context_claim_refresh


def main() -> None:
    payload = run_stage42_post_eq_context_claim_refresh(refresh_readmes=True)
    gate = payload["stage42_er_gate"]
    print(f"Stage42-ER post-EQ context claim refresh complete: {gate['passed']}/{gate['total']} {gate['verdict']}")


if __name__ == "__main__":
    main()
