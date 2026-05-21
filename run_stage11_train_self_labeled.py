from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


def configure_stage11_env() -> None:
    os.environ["STAGE9_EP_ROOT"] = "data/stage11_multiagent_episodes"
    os.environ["STAGE9_SCENE_PACK_ROOT"] = "data/stage11_scene_packs"
    os.environ["STAGE9_REPORT_DIR"] = "outputs/reports/stage11"
    os.environ["STAGE9_CKPT_DIR"] = "outputs/checkpoints/stage11"


def main() -> None:
    configure_stage11_env()
    from src.evaluation.stage9_data_audit import audit_stage9_data, write_stage9_data_audit
    from src.evaluation.stage9_per_agent_baselines import run_stage9_baselines, write_stage9_baselines
    from src.evaluation.stage9_benchmark import run_stage9_benchmark, write_stage9_benchmark
    from src.training.train_stage9_per_agent import train_stage9_models

    report_dir = Path(os.environ["STAGE9_REPORT_DIR"])
    report_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_stage9_data()
    write_stage9_data_audit(audit)
    baselines = run_stage9_baselines()
    write_stage9_baselines(baselines)
    train = train_stage9_models()
    bench = run_stage9_benchmark()
    write_stage9_benchmark(bench)
    summary = summarize(audit, baselines, train, bench)
    (report_dir / "report_stage11_self_labeled_training.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (report_dir / "report_stage11_self_labeled_training.md").write_text(markdown(summary), encoding="utf-8")
    package_results(report_dir)
    print(json.dumps(summary, indent=2))


def summarize(audit: dict, baselines: dict, train: dict, bench: dict) -> dict:
    best = {}
    for variant in bench.get("variants", []):
        for dataset, drow in variant.get("datasets", {}).items():
            all_subset = drow.get("subsets", {}).get("all", {})
            if not all_subset.get("horizons"):
                continue
            target = "10" if "10" in all_subset["horizons"] else max(all_subset["horizons"], key=lambda x: int(x))
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
    return {
        "stage": "11",
        "model_type": "self_labeled_per_agent_deterministic_residual",
        "datasets": sorted(baselines.get("datasets", {}).keys()),
        "episodes": audit.get("total_per_agent_multiagent_episodes"),
        "episodes_ge2": audit.get("episodes_with_ge2_agents"),
        "verified_t10": audit.get("actual_verified_t10_episodes"),
        "verified_t50": audit.get("actual_verified_t50_episodes"),
        "verified_t100": audit.get("actual_verified_t100_episodes"),
        "predicts_all_agents": train.get("predicts_all_agents"),
        "latent_enabled": False,
        "smc_enabled": False,
        "best_by_dataset": best,
        "limitations": [
            "AerialMPT visual labels are AI visual silver, not human gold.",
            "AerialMPT is pixel-space because no homography or metric scale is available.",
            "Pedestrian/drone t+50/t+100 remains unavailable.",
            "Stage 11 is deterministic; latent generative and SMC remain disabled.",
        ],
    }


def markdown(summary: dict) -> str:
    lines = [
        "# Stage 11 Self-Labeled Training Report",
        "",
        f"Model type: `{summary['model_type']}`",
        f"Datasets: `{summary['datasets']}`",
        f"Episodes >=2 agents: `{summary['episodes_ge2']}`",
        f"Verified t+10/t+50/t+100: `{summary['verified_t10']}/{summary['verified_t50']}/{summary['verified_t100']}`",
        f"Predicts all agents: `{summary['predicts_all_agents']}`",
        f"Latent enabled: `{summary['latent_enabled']}`",
        f"SMC enabled: `{summary['smc_enabled']}`",
        "",
        "| dataset | best variant | horizon | FDE | baseline FDE | improvement |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for dataset, row in summary["best_by_dataset"].items():
        lines.append(f"| {dataset} | {row['variant']} | {row['horizon']} | {row['FDE']} | {row['baseline_FDE']} | {row['improvement']} |")
    lines += ["", "## Limitations", ""]
    lines += [f"- {item}" for item in summary["limitations"]]
    return "\n".join(lines) + "\n"


def package_results(report_dir: Path) -> None:
    out = Path("outputs/world_model_stage11_results")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(report_dir, out / "reports")
    for report in Path("outputs/reports").glob("stage11_*"):
        shutil.copy2(report, out / "reports" / report.name)
    if Path("outputs/checkpoints/stage11").exists():
        shutil.copytree("outputs/checkpoints/stage11", out / "checkpoints")
    if Path("outputs/figures/stage11_visual_annotations").exists():
        shutil.copytree("outputs/figures/stage11_visual_annotations", out / "figures" / "stage11_visual_annotations")
    for src, dst in [
        ("data/stage11_visual_annotations", "visual_annotations"),
        ("data/stage11_scene_packs", "scene_packs"),
        ("data/stage11_multiagent_episodes", "multiagent_episodes"),
    ]:
        if Path(src).exists():
            shutil.copytree(src, out / dst)


if __name__ == "__main__":
    main()
