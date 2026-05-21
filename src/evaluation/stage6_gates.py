from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


REPORT_DIR = Path("outputs/reports")
OFFICIAL_MODELS = {"failure_predictor_only_gate", "learned_alpha_gate", "hybrid_failure_predictor_plus_learned_gate"}


def load_json(path: str | Path, default):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def best_improvements(metrics: Dict, subset: str) -> Dict[str, float]:
    out = {}
    for variant in metrics.get("variants", []):
        if variant.get("variant") not in OFFICIAL_MODELS:
            continue
        for dataset, drow in variant.get("datasets", {}).items():
            srow = drow.get("subsets", {}).get(subset)
            if not srow:
                continue
            horizons = srow.get("horizons", {})
            if not horizons:
                continue
            target = "100" if "100" in horizons else max(horizons, key=lambda h: int(h))
            imp = float(horizons[target]["improvement_over_strongest"])
            out[dataset] = max(out.get(dataset, -999.0), imp)
    return out


def alpha_stats(metrics: Dict) -> Dict:
    best = None
    for variant in metrics.get("variants", []):
        if variant.get("variant") == "hybrid_failure_predictor_plus_learned_gate":
            best = variant
            break
    best = best or (metrics.get("variants", [{}])[0] if metrics.get("variants") else {})
    rows = best.get("alpha_rows", [])
    easy = [r["alpha"] for r in rows if r.get("subset") == "easy"]
    hard = [r["alpha"] for r in rows if r.get("subset") == "hard"]
    failure = [r["alpha"] for r in rows if r.get("baseline_failure")]
    labels = np.asarray([float(r.get("baseline_failure", False)) for r in rows], dtype=float)
    alpha = np.asarray([float(r.get("alpha", 0.0)) for r in rows], dtype=float)
    corr = None
    if len(alpha) > 1 and len(set(labels.tolist())) > 1:
        corr = float(np.corrcoef(labels, alpha)[0, 1])
    return {
        "easy": float(np.mean(easy)) if easy else None,
        "hard": float(np.mean(hard)) if hard else None,
        "failure": float(np.mean(failure)) if failure else None,
        "corr": corr,
    }


def interaction_gain() -> Dict:
    rows = load_json(REPORT_DIR / "stage6_interaction_ablation.json", [])
    by = {r["model"]: r for r in rows}
    no = by.get("no_interaction_ablation", {})
    scalar = by.get("scalar_interaction_ablation", {})
    graph = by.get("graph_interaction_ablation", {})
    no_val = max(float(no.get("mean_hard_improvement", 0.0)), float(no.get("mean_failure_improvement", 0.0)))
    best_int = max(
        float(scalar.get("mean_hard_improvement", -999.0)),
        float(scalar.get("mean_failure_improvement", -999.0)),
        float(graph.get("mean_hard_improvement", -999.0)),
        float(graph.get("mean_failure_improvement", -999.0)),
    )
    return {"no": no_val, "interaction": best_int, "passed": best_int > no_val + 0.02}


def evaluate_gates() -> Dict:
    ped = load_json(REPORT_DIR / "stage6_pedestrian_drone_audit.json", [])
    hardbench = load_json(REPORT_DIR / "hardbench_v1_summary.json", {})
    failure_bench = load_json(REPORT_DIR / "baseline_failure_bench_summary.json", {})
    predictor = load_json(REPORT_DIR / "baseline_failure_predictor_metrics.json", {})
    metrics = load_json(REPORT_DIR / "metrics_stage6.json", {"variants": []})
    ped_ok = any(r.get("eligible_for_pedestrian_drone_long_horizon_gate") for r in ped)
    hard_total = int(hardbench.get("total_hard_episodes", 0))
    hard_ok = hard_total >= 50
    fail_ok = bool(failure_bench.get("enough_samples_for_training")) and bool(failure_bench.get("enough_samples_for_evaluation"))
    test = predictor.get("test", {})
    auroc = float(test.get("AUROC", 0.0))
    auprc = float(test.get("AUPRC", 0.0))
    pos_rate = float(test.get("positive_rate", 0.0))
    predictor_ok = auroc >= 0.70 or (auprc > max(pos_rate + 0.1, 0.2) and auroc > 0.55)
    astats = alpha_stats(metrics)
    alpha_ok = astats["easy"] is not None and astats["hard"] is not None and astats["failure"] is not None and astats["easy"] < astats["hard"] < astats["failure"] and astats["corr"] is not None and astats["corr"] > 0
    failure_imps = best_improvements(metrics, "baseline_failure")
    failure_model_ok = any(v >= 0.10 for v in failure_imps.values())
    easy_imps = best_improvements(metrics, "easy")
    easy_ok = all(v >= -0.05 for v in easy_imps.values()) if easy_imps else False
    long_imps = {**best_improvements(metrics, "verified_t50"), **best_improvements(metrics, "verified_t100")}
    long_ok = any(v >= 0.05 for v in long_imps.values())
    igain = interaction_gain()
    gates = [
        gate("Pedestrian/Drone Long-Horizon Gate", ped_ok, f"{sum(1 for r in ped if r.get('eligible_for_pedestrian_drone_long_horizon_gate'))} actual pedestrian/drone sources support verified t+50/t+100", "No pedestrian long-horizon claim without this."),
        gate("HardBench Reliability Gate", hard_ok, f"HardBench-v1 hard episodes={hard_total}, eligibility={hardbench.get('gate_eligibility')}", "Need at least 50 official hard episodes."),
        gate("BaselineFailureBench Gate", fail_ok, f"failure_samples={failure_bench.get('failure_samples')}, train_ok={failure_bench.get('enough_samples_for_training')}, eval_ok={failure_bench.get('enough_samples_for_evaluation')}", "Need enough failure samples for train/test."),
        gate("Failure Predictor Gate", predictor_ok, f"AUROC={auroc}, AUPRC={auprc}, positive_rate={pos_rate}", "Improve causal failure predictor."),
        gate("Alpha Calibration Gate", alpha_ok, f"easy={astats['easy']}, hard={astats['hard']}, failure={astats['failure']}, corr={astats['corr']}", "Alpha should increase from easy to hard to failure."),
        gate("Failure-Aware Improvement Gate", failure_model_ok, f"best failure subset improvements={failure_imps}", "Need >=10% improvement on BaselineFailureBench."),
        gate("Easy Preservation Gate", easy_ok, f"best easy improvements={easy_imps}", "Do not degrade easy cases."),
        gate("Verified Long-Horizon Gate", long_ok, f"verified long-horizon improvements={long_imps}", "Need >=5% improvement on verified t+50/t+100."),
        gate("Interaction Gate", igain["passed"], f"no_interaction={igain['no']}, interaction={igain['interaction']}", "Interaction features must help hard/failure subsets."),
    ]
    ready = gates[1]["passed"] and gates[2]["passed"] and gates[3]["passed"] and gates[4]["passed"] and gates[5]["passed"] and gates[6]["passed"] and gates[8]["passed"] and (gates[0]["passed"] or gates[7]["passed"])
    gates.append(gate("Stage 5C Readiness Gate", ready, "ready" if ready else "Do not enter Stage 5C. Deterministic failure-aware gates did not pass.", "Pass Stage 6 deterministic gates first."))
    passed = sum(g["passed"] for g in gates)
    return {
        "stage": "6",
        "gates": gates,
        "passed": passed,
        "total": len(gates),
        "latent_stage5c_ready": ready,
        "smc_ready": False,
        "expert_audit_score": 70 if passed >= 5 else 68,
        "verdict": "stage6_failure_bench_built_but_not_stage5c_ready" if not ready else "stage6_ready_for_stage5c",
    }


def gate(name, passed, evidence, next_fix):
    return {"name": name, "passed": bool(passed), "evidence": evidence, "next_fix": next_fix}


def write_report(result: Dict, path: str | Path = REPORT_DIR / "world_model_gate_stage6.md") -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Stage 6 Gates", "", f"Passed: {result['passed']} / {result['total']}", "", "| gate | pass | evidence | next fix |", "| --- | --- | --- | --- |"]
    for g in result["gates"]:
        lines.append(f"| {g['name']} | {g['passed']} | {g['evidence']} | {g['next_fix']} |")
    lines += ["", f"latent_stage5c_ready: `{result['latent_stage5c_ready']}`", f"smc_ready: `{result['smc_ready']}`", f"expert_audit_score: `{result['expert_audit_score']}`", f"verdict: `{result['verdict']}`"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

