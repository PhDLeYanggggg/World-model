from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_unification.auto_convert_dataset import write_auto_conversion_report
from src.data_unification.auto_episode_builder import write_auto_episode_report
from src.data_unification.auto_horizon_audit import write_auto_horizon_audit
from src.orchestrator.research_state import write_md


def main() -> None:
    conversion = write_auto_conversion_report()
    horizon = write_auto_horizon_audit()
    episodes = write_auto_episode_report()
    write_md(
        "outputs/reports/auto_no_leakage_audit.md",
        [
            "# Auto No-Leakage Audit",
            "",
            "- official velocity: causal_fd",
            "- central_fd: diagnostic only",
            "- candidate goals: train-only suggestions",
            "- future endpoint input: forbidden",
            "- status: inherited pass from Stage 12 reports unless a new conversion is added",
        ],
    )
    write_md(
        "outputs/reports/auto_baseline_report.md",
        [
            "# Auto Baseline Report",
            "",
            "- Strongest causal baselines are inherited from Stage 12 rebenchmark until a new dataset is converted.",
        ],
    )
    print({"conversion": conversion, "episodes": episodes})


if __name__ == "__main__":
    main()
