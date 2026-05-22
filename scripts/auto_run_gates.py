from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestrator.auto_gates import evaluate_auto_gates, write_auto_gate_report
from src.orchestrator.auto_loop import build_current_state


def main() -> None:
    gates = evaluate_auto_gates(build_current_state())
    write_auto_gate_report(gates)
    print({"passed": len(gates["passed"]), "failed": len(gates["failed"]), "latent_ready": gates["latent_generative_ready"]})


if __name__ == "__main__":
    main()
