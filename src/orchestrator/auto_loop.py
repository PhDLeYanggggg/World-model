from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

from .auto_gates import evaluate_auto_gates, write_auto_gate_report
from .decision_engine import decide_next_actions
from .failure_analyzer import analyze_failures, datasets_from_stage12, strongest_baseline_summary
from .gate_reader import summarize_latest_reports
from .report_writer import write_current_state_report, write_iteration_report, write_next_stage_plan
from .research_state import ResearchState, read_json, read_text, write_research_state_markdown
from .task_planner import build_task_plan


REQUESTED_FILES = [
    "README_RESULTS.md",
    "outputs/reports/report_stage13_final.md",
    "outputs/reports/world_model_gate_stage13.md",
    "outputs/reports/failure_analysis_stage13.md",
    "outputs/reports/report_stage12_final.md",
    "outputs/reports/world_model_gate_stage12.md",
    "outputs/reports/data_card_stage12.md",
    "outputs/reports/annotation_card_stage12.md",
    "outputs/reports/report_stage11_final.md",
    "outputs/reports/report_stage10_final.md",
    "outputs/reports/report_stage9_final.md",
    "outputs/world_model_stage5_data_results/data_registry/dataset_registry_stage5.json",
    "outputs/world_model_stage5_data_results/data_registry/dataset_registry_stage5.md",
]


def missing_requested_files() -> List[str]:
    missing = []
    for item in REQUESTED_FILES:
        if not Path(item).exists():
            missing.append(item)
    return missing


def stage12_counts() -> Dict[str, Any]:
    summary = read_json("outputs/reports/stage12_final_summary.json", default={}) or {}
    annotation = read_json("outputs/reports/stage12_annotation_report.json", default={}) or {}
    episodes = read_json("outputs/reports/stage12_multiagent_episode_report.json", default={}) or {}
    hard = read_json("outputs/reports/stage12_hard_failure_report.json", default=[]) or []
    goal = read_json("outputs/reports/stage12_goalbench_v4_report.json", default={}) or {}
    return {
        "summary": summary,
        "annotation": annotation,
        "episodes": episodes,
        "hard_failure_count": len(hard) if isinstance(hard, list) else hard.get("records", 0),
        "goalbench": goal,
    }


def strongest_baseline_names() -> Dict[str, str]:
    path = Path("outputs/reports/stage12_rebenchmark/stage9_per_agent_baseline_table.md")
    if not path.exists():
        return {}
    names: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "---" in line or "dataset" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) >= 10 and cells[9].lower() == "true":
            names[cells[0]] = cells[1]
    return names


def build_current_state() -> Dict[str, Any]:
    latest = summarize_latest_reports()
    failures = analyze_failures(latest["report_text"], latest["gate_text"])
    counts = stage12_counts()
    datasets = datasets_from_stage12()
    baselines = strongest_baseline_summary()
    baseline_names = strongest_baseline_names()

    summary = counts["summary"]
    annotation = counts["annotation"]
    episodes = counts["episodes"]
    goalbench = counts["goalbench"]
    human_confirmed = int(summary.get("human_confirmed_scenes", annotation.get("human_confirmed_scenes", 0)) or 0)
    silver_rule = int(summary.get("silver_rule_confirmed_scenes", annotation.get("silver_rule_confirmed_scenes", 0)) or 0)
    goalbench_official = int(summary.get("goalbench_official", goalbench.get("official_records", 0)) or 0)
    hard_failure = int(summary.get("hard_failure_total", counts["hard_failure_count"]) or 0)
    episodes_ge2 = int(summary.get("multiagent_ge2", episodes.get("episodes_ge2_agents", 0)) or 0)
    verified_sources = datasets["verified_long_horizon"]
    deterministic_gate = "deterministic_5pct_gate = false" not in latest["report_text"].lower()
    # Stage 12 summary explicitly says deterministic gate false. Keep this conservative.
    deterministic_gate = False

    state = {
        "current_highest_stage": latest["latest_stage"],
        "latest_report_path": latest["latest_report_path"],
        "latest_gate_path": latest["latest_gate_path"],
        "missing_requested_files": missing_requested_files(),
        "expert_audit_score": latest["expert_audit_score"],
        "verdict": latest["verdict"],
        "model_type": "2.5D / pseudo-3D per-agent multi-agent trajectory world-state scaffold",
        "true_3D": False,
        "large_scale_foundation_model": False,
        "stage13_ready": latest["stage13_ready"],
        "latent_generative_ready": False,
        "smc_ready": False,
        "strongest_causal_baseline": baseline_names or "per-dataset causal kinematic baseline",
        "strongest_causal_baselines": baselines,
        "learned_model_beats_strongest_baseline": "否" if not deterministic_gate else "是",
        "verified_pedestrian_drone_t50_t100": verified_sources,
        "pedestrian_long_horizon_ready": bool(verified_sources),
        "human_gold_annotations": 0,
        "human_silver_annotations": human_confirmed,
        "silver_rule_confirmed_annotations": silver_rule,
        "scene_annotation_ready": human_confirmed >= 3,
        "goalbench_beats_majority": False,
        "goalbench_official_records": goalbench_official,
        "hardbench_baselinefailure_enough": hard_failure >= 100,
        "hard_failure_records": hard_failure,
        "multi_agent_ready": episodes_ge2 >= 300,
        "multi_agent_episodes_ge2": episodes_ge2,
        "datasets_converted": datasets["loaded"],
        "datasets_registry_only": [],
        "datasets_failed": [],
        "top_failure_reasons": failures["top_failures"][:3],
        "best_auto_directions": [
            "Stage 13 deterministic repair: failure-aware bounded residual that preserves strong causal baselines.",
            "Add/verify more SDD/OpenTraj pedestrian/drone scenes with images/homography where legally available.",
            "Upgrade rule-confirmed silver annotations into human-confirmed silver/gold and measure GoalBench lift.",
        ],
        "blockers_requiring_user": failures["user_blockers"],
    }
    return state


def research_state_from_current(current: Dict[str, Any], decision: Dict[str, Any], gates: Dict[str, Any]) -> ResearchState:
    state = ResearchState.load()
    state.current_stage = f"stage{current['current_highest_stage']}"
    state.current_verdict = current["verdict"]
    state.expert_audit_score = int(current["expert_audit_score"] or 0)
    state.deterministic_ready = current["learned_model_beats_strongest_baseline"] == "是"
    state.latent_generative_ready = False
    state.smc_ready = False
    state.pedestrian_long_horizon_ready = bool(current["verified_pedestrian_drone_t50_t100"])
    state.scene_annotation_ready = bool(current["scene_annotation_ready"])
    state.multi_agent_ready = bool(current["multi_agent_ready"])
    state.strongest_causal_baselines = current["strongest_causal_baselines"]
    state.best_learned_models = current["strongest_causal_baselines"]
    state.datasets_converted = current["datasets_converted"]
    state.datasets_registry_only = current["datasets_registry_only"]
    state.datasets_failed = current["datasets_failed"]
    state.annotation_status = {
        "human_gold": current["human_gold_annotations"],
        "human_silver": current["human_silver_annotations"],
        "silver_rule_confirmed": current["silver_rule_confirmed_annotations"],
    }
    state.gates_passed = gates["passed"]
    state.gates_failed = gates["failed"]
    state.next_actions = [row["name"] for row in decision.get("actions", [])[:5]]
    state.blockers_requiring_user = decision.get("blockers_requiring_user", [])
    state.generated_reports = [
        "outputs/reports/auto_orchestrator_current_state.md",
        "outputs/reports/research_state.md",
        "outputs/reports/auto_gate_report.md",
        "outputs/reports/auto_next_stage_plan.md",
        "outputs/reports/auto_loop_iteration_report.md",
    ]
    return state


def run_auto_loop(mode: str = "quick", max_steps: int = 1) -> Dict[str, Any]:
    current = build_current_state()
    write_current_state_report(current)

    gates = evaluate_auto_gates(current)
    write_auto_gate_report(gates)

    decision = decide_next_actions(current, analyze_failures(read_text(current["latest_report_path"]), read_text(current["latest_gate_path"])), mode=mode)
    tasks = build_task_plan(decision, max_steps=max_steps)
    executed: List[Dict[str, Any]] = []

    # The quick default intentionally does not launch heavy training or large downloads.
    for task in tasks:
        task = dict(task)
        task["status"] = "planned_not_executed_in_quick_loop"
        executed.append(task)

    write_next_stage_plan(decision, tasks)
    write_iteration_report(current, tasks, executed, decision)

    research_state = research_state_from_current(current, decision, gates)
    research_state.last_successful_command = f"python run_auto_world_model_loop.py --mode {mode} --max-steps {max_steps}"
    research_state.save()
    write_research_state_markdown(research_state)

    return {
        "current_state": current,
        "gates": gates,
        "decision": decision,
        "tasks": tasks,
        "executed": executed,
        "research_state": research_state.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous world-model research loop.")
    parser.add_argument("--mode", default="quick", choices=["quick", "data-first", "annotation-first", "train-deterministic", "full-auto", "overnight-stage13", "continuous-stage14", "continuous-stage15"])
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--min-hours", type=float, default=1.0)
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--min-training-trials", type=int, default=30)
    parser.add_argument("--min-data-actions", type=int, default=3)
    parser.add_argument("--min-benchmark-runs", type=int, default=3)
    parser.add_argument("--min-oracle-runs", type=int, default=3)
    parser.add_argument("--allow-training", action="store_true")
    parser.add_argument("--allow-data-discovery", action="store_true")
    parser.add_argument("--allow-safe-download-dry-run", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--allow-git", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-latent", action="store_true", default=True)
    parser.add_argument("--no-smc", action="store_true", default=True)
    parser.add_argument("--stop-on-user-blocker", action="store_true", default=True)
    parser.add_argument("--continue-on-task-failure", action="store_true", default=True)
    parser.add_argument("--heartbeat-minutes", type=float, default=15.0)
    parser.add_argument("--max-trials-per-family", type=int, default=2)
    parser.add_argument("--max-epochs-per-trial", type=int, default=20)
    parser.add_argument("--gpu-auto-detect", action="store_true", default=True)
    parser.add_argument("--cpu-safe-mode", action="store_true", default=True)
    parser.add_argument("--dynamic-queue", action="store_true")
    args = parser.parse_args()
    if args.mode == "overnight-stage13":
        from .overnight_runner import run_overnight_stage13

        result = run_overnight_stage13(
            max_hours=args.max_hours,
            max_iterations=args.max_iterations,
            allow_training=args.allow_training,
            allow_download=args.allow_download,
            allow_git=args.allow_git,
            heartbeat_minutes=args.heartbeat_minutes,
            max_trials_per_family=args.max_trials_per_family,
            no_latent=args.no_latent,
            no_smc=args.no_smc,
            continue_on_task_failure=args.continue_on_task_failure,
        )
        print({
            "mode": args.mode,
            "completed": len(result["completed"]),
            "failed": len(result["failed"]),
            "verdict": result["final_report"]["current_verdict"],
            "latent_ready": result["final_report"]["latent_generative_ready"],
            "smc_ready": result["final_report"]["smc_ready"],
        })
        return
    if args.mode == "continuous-stage14":
        from .overnight_runner import run_continuous_stage14

        result = run_continuous_stage14(
            min_hours=args.min_hours,
            max_hours=args.max_hours,
            max_iterations=args.max_iterations,
            min_training_trials=args.min_training_trials,
            min_data_actions=args.min_data_actions,
            min_benchmark_runs=args.min_benchmark_runs,
            allow_training=args.allow_training,
            allow_data_discovery=args.allow_data_discovery,
            allow_safe_download_dry_run=args.allow_safe_download_dry_run,
            allow_git=args.allow_git,
            heartbeat_minutes=args.heartbeat_minutes,
            max_trials_per_family=args.max_trials_per_family,
            no_latent=args.no_latent,
            no_smc=args.no_smc,
            dynamic_queue=args.dynamic_queue,
            continue_on_task_failure=args.continue_on_task_failure,
        )
        print({
            "mode": args.mode,
            "elapsed_hours": result["loop_report"]["elapsed_hours"],
            "completed": len(result["completed"]),
            "failed": len(result["failed"]),
            "training_trials": result["loop_report"]["training_trials"],
            "data_actions": result["loop_report"]["data_actions"],
            "benchmark_runs": result["loop_report"]["benchmark_runs"],
            "verdict": result["final_report"]["current_verdict"],
            "latent_ready": result["final_report"]["latent_stage5c_ready"],
            "smc_ready": result["final_report"]["smc_ready"],
        })
        return
    if args.mode == "continuous-stage15":
        from .overnight_runner import run_continuous_stage15

        result = run_continuous_stage15(
            min_hours=args.min_hours,
            max_hours=args.max_hours,
            max_iterations=args.max_iterations,
            min_training_trials=args.min_training_trials,
            min_data_actions=args.min_data_actions,
            min_oracle_runs=args.min_oracle_runs,
            allow_training=args.allow_training,
            allow_data_discovery=args.allow_data_discovery,
            allow_safe_download_dry_run=args.allow_safe_download_dry_run,
            allow_git=args.allow_git,
            heartbeat_minutes=args.heartbeat_minutes,
            no_latent=args.no_latent,
            no_smc=args.no_smc,
            dynamic_queue=args.dynamic_queue,
            continue_on_task_failure=args.continue_on_task_failure,
        )
        print({
            "mode": args.mode,
            "elapsed_hours": result["loop_report"]["elapsed_hours"],
            "completed": len(result["completed"]),
            "failed": len(result["failed"]),
            "training_trials": result["loop_report"]["training_trials"],
            "data_actions": result["loop_report"]["data_actions"],
            "oracle_runs": result["loop_report"]["oracle_runs"],
            "verdict": result["final_report"]["current_verdict"],
            "latent_ready": result["final_report"]["stage5c_ready"],
            "smc_ready": result["final_report"]["smc_ready"],
        })
        return
    result = run_auto_loop(mode=args.mode, max_steps=max(1, args.max_steps))
    print({
        "stage": result["current_state"]["current_highest_stage"],
        "verdict": result["current_state"]["verdict"],
        "expert_audit_score": result["current_state"]["expert_audit_score"],
        "latent_ready": result["gates"]["latent_generative_ready"],
        "smc_ready": result["gates"]["smc_ready"],
        "planned_tasks": [task["task"] for task in result["tasks"]],
    })


if __name__ == "__main__":
    main()
