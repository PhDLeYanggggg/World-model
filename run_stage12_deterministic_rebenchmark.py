from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


def configure_env() -> None:
    os.environ["STAGE9_EP_ROOT"] = "data/stage12_multiagent_episodes"
    os.environ["STAGE9_SCENE_PACK_ROOT"] = "data/stage12_scene_packs"
    os.environ["STAGE9_REPORT_DIR"] = "outputs/reports/stage12_rebenchmark"
    os.environ["STAGE9_CKPT_DIR"] = "outputs/checkpoints/stage12"


def main() -> None:
    gates = json.loads(Path("outputs/reports/world_model_gate_stage12.json").read_text(encoding="utf-8"))
    if not gates.get("stage13_ready"):
        raise SystemExit("Stage 12 gates do not allow deterministic re-benchmark/training yet.")
    configure_env()
    from src.evaluation.stage9_data_audit import audit_stage9_data, write_stage9_data_audit
    from src.evaluation.stage9_per_agent_baselines import run_stage9_baselines, write_stage9_baselines
    from src.training.train_stage9_per_agent import train_stage9_models
    from src.evaluation.stage9_benchmark import run_stage9_benchmark, write_stage9_benchmark

    report_dir = Path(os.environ["STAGE9_REPORT_DIR"])
    report_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_stage9_data()
    write_stage9_data_audit(audit)
    baselines = run_stage9_baselines()
    write_stage9_baselines(baselines)
    training = train_stage9_models()
    benchmark = run_stage9_benchmark()
    write_stage9_benchmark(benchmark)
    summary = summarize(audit, baselines, training, benchmark)
    (report_dir / "report_stage12_deterministic_rebenchmark.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (report_dir / "report_stage12_deterministic_rebenchmark.md").write_text(markdown(summary), encoding="utf-8")
    package_rebenchmark(report_dir)
    print(json.dumps(summary, indent=2))


def summarize(audit: dict, baselines: dict, training: dict, benchmark: dict) -> dict:
    best = {}
    for variant in benchmark.get("variants", []):
        for dataset, drow in variant.get("datasets", {}).items():
            all_subset = drow.get("subsets", {}).get("all", {})
            if not all_subset.get("horizons"):
                continue
            target = "100" if "100" in all_subset["horizons"] else ("10" if "10" in all_subset["horizons"] else max(all_subset["horizons"], key=lambda x: int(x)))
            row = all_subset["horizons"][target]
            current = best.get(dataset)
            if current is None or row["FDE"] < current["FDE"]:
                best[dataset] = {
                    "variant": variant.get("variant"),
                    "horizon": target,
                    "FDE": row["FDE"],
                    "baseline_FDE": row["baseline_FDE"],
                    "improvement": row["improvement_over_strongest"],
                }
    wins = {dataset: row for dataset, row in best.items() if row["improvement"] > 0.05}
    return {
        "stage": "12",
        "model_type": "deterministic_per_agent_scene_grounded_rebenchmark",
        "datasets": sorted(baselines.get("datasets", {}).keys()),
        "episodes": audit.get("total_per_agent_multiagent_episodes"),
        "episodes_ge2": audit.get("episodes_with_ge2_agents"),
        "verified_t10": audit.get("actual_verified_t10_episodes"),
        "verified_t50": audit.get("actual_verified_t50_episodes"),
        "verified_t100": audit.get("actual_verified_t100_episodes"),
        "predicts_all_agents": training.get("predicts_all_agents"),
        "latent_enabled": False,
        "smc_enabled": False,
        "best_by_dataset": best,
        "datasets_with_5pct_win": wins,
        "deterministic_gate_passed": len(wins) >= 1,
        "limitations": [
            "This is deterministic re-benchmarking, not latent generative modeling.",
            "Only ETH/UCY EWAP currently provides verified pedestrian long-horizon t+50/t+100.",
            "AerialMPT remains pixel-space because no homography or metric scale is available.",
            "SMC remains disabled.",
        ],
    }


def markdown(summary: dict) -> str:
    lines = [
        "# Stage 12 Deterministic Re-benchmark",
        "",
        f"Datasets: `{summary['datasets']}`",
        f"Episodes >=2 agents: `{summary['episodes_ge2']}`",
        f"Verified t+10/t+50/t+100: `{summary['verified_t10']}/{summary['verified_t50']}/{summary['verified_t100']}`",
        f"Predicts all agents: `{summary['predicts_all_agents']}`",
        f"Latent enabled: `{summary['latent_enabled']}`",
        f"SMC enabled: `{summary['smc_enabled']}`",
        f"Deterministic gate passed: `{summary['deterministic_gate_passed']}`",
        "",
        "| dataset | best variant | horizon | FDE | baseline FDE | improvement |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for dataset, row in summary["best_by_dataset"].items():
        lines.append(f"| {dataset} | {row['variant']} | {row['horizon']} | {row['FDE']} | {row['baseline_FDE']} | {row['improvement']} |")
    lines += ["", "## Limitations", ""]
    lines += [f"- {item}" for item in summary["limitations"]]
    return "\n".join(lines) + "\n"


def package_rebenchmark(report_dir: Path) -> None:
    out = Path("outputs/world_model_stage12_results")
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "rebenchmark_reports"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(report_dir, dst)
    ckpt = Path("outputs/checkpoints/stage12")
    if ckpt.exists():
        target = out / "checkpoints"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(ckpt, target)


if __name__ == "__main__":
    main()
