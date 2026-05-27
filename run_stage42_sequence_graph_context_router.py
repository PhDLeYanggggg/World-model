from __future__ import annotations

from src.stage42_sequence_graph_context_router import run_stage42_sequence_graph_context_router


def main() -> None:
    payload = run_stage42_sequence_graph_context_router(refresh_readmes=True)
    gate = payload["stage42_eq_gate"]
    print(f"Stage42-EQ sequence+graph context router complete: {gate['passed']}/{gate['total']} {gate['verdict']}")


if __name__ == "__main__":
    main()
