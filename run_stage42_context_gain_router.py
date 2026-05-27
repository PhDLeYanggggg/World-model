from __future__ import annotations

from src.stage42_context_gain_router import run_stage42_context_gain_router


def main() -> None:
    payload = run_stage42_context_gain_router(refresh_readmes=True)
    gate = payload["stage42_el_gate"]
    print(f"Stage42-EL context gain router complete: {gate['passed']}/{gate['total']} {gate['verdict']}")


if __name__ == "__main__":
    main()
