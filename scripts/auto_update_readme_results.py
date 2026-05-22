from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestrator.auto_loop import build_current_state


def main() -> None:
    state = build_current_state()
    path = Path("README_RESULTS.md")
    marker = "## Auto-Orchestrator Status"
    block = "\n".join(
        [
            marker,
            "",
            "This section is maintained by `scripts/auto_update_readme_results.py`.",
            "",
            "```text",
            f"current_highest_stage = {state['current_highest_stage']}",
            f"expert_audit_score = {state['expert_audit_score']}",
            f"verdict = {state['verdict']}",
            f"latent_generative_ready = {state['latent_generative_ready']}",
            f"smc_ready = {state['smc_ready']}",
            f"learned_model_beats_strongest_baseline = {state['learned_model_beats_strongest_baseline']}",
            "```",
            "",
        ]
    )
    text = path.read_text(encoding="utf-8") if path.exists() else "# Physical World Model 2.5D Results\n"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n\n" + block
    else:
        text = text.rstrip() + "\n\n" + block
    path.write_text(text, encoding="utf-8")
    print({"updated": str(path), "stage": state["current_highest_stage"]})


if __name__ == "__main__":
    main()
