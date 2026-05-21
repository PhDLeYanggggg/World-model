from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


REPORT_DIR = Path("outputs/reports")
OFFICIAL_GATED_VARIANTS = {
    "gated_residual_all_data",
    "gated_residual_hard_weighted",
    "gated_residual_failure_classifier_aux",
}


def load_json(path: str | Path, default):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def best_target_improvements(metrics: Dict, subset: str = "all", official_only: bool = True) -> Dict[str, float]:
    best: Dict[str, float] = {}
    for variant in metrics.get("variants", []):
        if official_only and variant.get("variant") not in OFFICIAL_GATED_VARIANTS:
            continue
        for dataset, drow in variant.get("datasets", {}).items():
            srow = drow.get("subsets", {}).get(subset)
            if not srow:
                continue
            horizons = srow.get("horizons", {})
            if not horizons:
                continue
            target = "100" if "100" in horizons else max(horizons, key=lambda h: int(h))
            imp = float(horizons[target].get("improvement_over_strongest", 0.0))
            best[dataset] = max(best.get(dataset, -999.0), imp)
    return best


def verified_t100_wins(metrics: Dict) -> int:
    wins = 0
    for dataset in ["tgsim", "tgsim_i90"]:
        best = -999.0
        for variant in metrics.get("variants", []):
            if variant.get("variant") not in OFFICIAL_GATED_VARIANTS:
                continue
            hrow = variant.get("datasets", {}).get(dataset, {}).get("subsets", {}).get("all", {}).get("horizons", {}).get("100")
            if hrow:
                best = max(best, float(hrow.get("improvement_over_strongest", 0.0)))
        if best >= 0.05:
            wins += 1
    return wins


def physical_validity_ok(metrics: Dict) -> bool:
    vals = []
    for variant in metrics.get("variants", []):
        for drow in variant.get("datasets", {}).values():
            for srow in drow.get("subsets", {}).values():
                vals.append(float(srow.get("physical_validity_rate", 1.0)))
    return min(vals) >= 0.95 if vals else False


def interaction_gate(ablation_rows: List[Dict]) -> Dict:
    no = next((r for r in ablation_rows if r.get("ablation") == "no interaction"), None)
    graph = next((r for r in ablation_rows if r.get("ablation") == "graph attention interaction"), None)
    temporal = next((r for r in ablation_rows if r.get("ablation") == "graph attention + temporal neighbor history"), None)
    no_val = float(no.get("mean_hard_target_improvement", 0.0)) if no else 0.0
    graph_val = max(
        float(graph.get("mean_hard_target_improvement", 0.0)) if graph else -999.0,
        float(temporal.get("mean_hard_target_improvement", 0.0)) if temporal else -999.0,
    )
    return {"passed": graph_val > no_val + 0.02, "no_interaction": no_val, "graph": graph_val}


def evaluate_gates() -> Dict:
    reliability = load_json(REPORT_DIR / "stage5b6_hard_reliability_audit.json", [])
    horizon = load_json(REPORT_DIR / "stage5b6_pedestrian_drone_horizon_report.json", [])
    leakage = load_json(REPORT_DIR / "leakage_audit_stage5b.json", [])
    metrics = load_json(REPORT_DIR / "metrics_stage5b6.json", {"variants": []})
    oracle = load_json(REPORT_DIR / "stage5b6_baseline_failure_oracle.json", {"summary": {}})
    ablation = load_json(REPORT_DIR / "stage5b6_interaction_ablation.json", [])
    official_hard = [r for r in reliability if r.get("hard_subset_is_gate_eligible")]
    ped_long = [r for r in horizon if r.get("t50_verified") or r.get("t100_verified")]
    target_wins = sum(1 for v in best_target_improvements(metrics, "all", official_only=True).values() if v >= 0.05)
    eligible_hard_names = {r["dataset_name"] for r in official_hard}
    hard_best = best_target_improvements(metrics, "hard", official_only=True)
    hard_wins = sum(1 for dataset, imp in hard_best.items() if dataset in eligible_hard_names and imp >= 0.10)
    summary = oracle.get("summary", {})
    alpha_corr = summary.get("alpha_vs_baseline_failure_correlation")
    easy_alpha = summary.get("easy_alpha_mean")
    hard_alpha = summary.get("hard_alpha_mean")
    alpha_ok = alpha_corr is not None and alpha_corr > 0.0 and easy_alpha is not None and hard_alpha is not None and easy_alpha < hard_alpha
    interaction = interaction_gate(ablation)
    t100_wins = verified_t100_wins(metrics)
    gates = [
        gate("Hard Reliability Gate", len(official_hard) >= 2, f"{len(official_hard)} hard subsets are official gate eligible (>=50 hard episodes)", "Collect more hard episodes; do not gate on one-episode wins."),
        gate("Pedestrian / Drone Horizon Gate", bool(ped_long), f"{len(ped_long)} pedestrian/drone sources have verified t+50/t+100", "Prepare SDD/full OpenTraj/long AerialMPT with legal access."),
        gate("No Leakage Gate", bool(leakage) and all(r.get("passed") for r in leakage), f"{sum(1 for r in leakage if r.get('passed'))}/{len(leakage)} leakage audits passed", "Keep official features causal."),
        gate("Gated Residual Gate", target_wins >= 2, f"gated residual beats strongest baseline by >=5% on {target_wins} dataset target horizons", "Improve baseline failure detection and residual training."),
        gate("Hard Interaction Gate", hard_wins >= 2, f"gated residual beats strongest baseline by >=10% on {hard_wins} official hard subsets", "Need reliable hard subsets and stronger interaction model."),
        gate("Alpha Calibration Gate", alpha_ok, f"corr={alpha_corr}, easy_alpha={easy_alpha}, hard_alpha={hard_alpha}", "Alpha should rise when baseline likely fails and stay low on easy segments."),
        gate("Interaction Encoder Gate", interaction["passed"], f"hard improvement no_interaction={interaction['no_interaction']}, graph={interaction['graph']}", "Graph interaction must beat no-interaction on hard subsets."),
        gate("Verified Long-Horizon Gate", t100_wins >= 1, f"{t100_wins} verified t+100 sources beat strongest baseline by >=5%", "Need at least one robust long-horizon win."),
        gate("Physical Validity Gate", physical_validity_ok(metrics), "minimum physical validity remains >=0.95" if physical_validity_ok(metrics) else "physical validity degraded or missing", "Bound residual and add validity losses."),
    ]
    ready = gates[0]["passed"] and gates[2]["passed"] and gates[3]["passed"] and gates[4]["passed"] and gates[5]["passed"] and gates[6]["passed"] and gates[8]["passed"] and (gates[1]["passed"] or gates[7]["passed"])
    gates.append(gate("Stage 5C Readiness Gate", ready, "ready" if ready else "Do not enter Stage 5C. Deterministic gated interaction model is not strong enough.", "Pass deterministic reliability gates first."))
    passed = sum(1 for g in gates if g["passed"])
    score = 71 if passed >= 6 else 69 if passed >= 5 else 68
    return {
        "stage": "5B.6",
        "gates": gates,
        "passed": passed,
        "total": len(gates),
        "latent_stage5c_ready": ready,
        "smc_ready": False,
        "expert_audit_score": score,
        "verdict": "stage5b6_reliability_repaired_but_deterministic_gate_failed" if not ready else "stage5b6_ready_for_stage5c",
    }


def gate(name, passed, evidence, next_fix):
    return {"name": name, "passed": bool(passed), "evidence": evidence, "next_fix": next_fix}


def write_report(result: Dict, path: str | Path = REPORT_DIR / "world_model_gate_stage5b6.md") -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Stage 5B.6 Gates", "", f"Passed: {result['passed']} / {result['total']}", "", "| gate | pass | evidence | next fix |", "| --- | --- | --- | --- |"]
    for g in result["gates"]:
        lines.append(f"| {g['name']} | {g['passed']} | {g['evidence']} | {g['next_fix']} |")
    lines += [
        "",
        f"latent_stage5c_ready: `{result['latent_stage5c_ready']}`",
        f"smc_ready: `{result['smc_ready']}`",
        f"expert_audit_score: `{result['expert_audit_score']}`",
        f"verdict: `{result['verdict']}`",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
