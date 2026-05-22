from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.auto_baseline_failure_bench import write_auto_baseline_failure_report
from src.evaluation.auto_benchmark import write_auto_benchmark_report
from src.evaluation.auto_failure_miner import write_auto_failure_analysis
from src.evaluation.auto_goalbench_builder import write_auto_goalbench_report
from src.scene.auto_scene_pack_builder import write_auto_scene_pack_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto benchmark wrapper.")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    scene = write_auto_scene_pack_report()
    goal = write_auto_goalbench_report()
    failure = write_auto_baseline_failure_report()
    bench = write_auto_benchmark_report()
    analysis = write_auto_failure_analysis()
    print({"quick": bool(args.quick), "benchmark_source": bench["source"], "failure_records": failure["baseline_failure_records"]})


if __name__ == "__main__":
    main()
