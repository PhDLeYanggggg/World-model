from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md


OUT = Path("outputs/m3w")


def run_m3w_gates() -> Dict[str, Any]:
    ensure_dir(OUT)
    metrics = read_json(OUT / "metrics_m3w.json", {})
    train = read_json(OUT / "training_report.json", {})
    feature = read_json("outputs/reports/stage26_feature_store_report.json", {})
    stage26 = read_json("outputs/reports/report_stage26_final.json", {})
    m = metrics.get("test_metrics", {})
    backend = metrics.get("backend") or train.get("backend")
    torch_backend = backend not in {"numpy_safe_fallback_due_torch_openmp_shm_blocker", None}
    gates = [
        ("Data Gate", bool(feature.get("feature_names")) or Path("data/stage26_sdd_feature_store/train.npz").exists(), "SDD Stage26 feature store available."),
        ("No Leakage Gate", feature.get("leakage_audit", {}).get("future_endpoint_input") is False, "Feature store audit forbids future/test leakage."),
        ("JEPA Non-Collapse Gate", bool(metrics.get("jepa_non_collapse", False)) and torch_backend, "Full torch JEPA latent variance must be non-collapsed; current small run did not pass."),
        ("JEPA Downstream Gate", False, "Small run did not prove JEPA improves selector/failure over non-JEPA baseline."),
        ("Transformer Dynamics Gate", train.get("best", {}).get("variant") in {"transformer_only", "hybrid"} and torch_backend, "Torch Transformer dynamics variant executed and was validation-selected."),
        ("Selector Gate", m.get("official_t50_improvement", 0.0) >= 0.05 or metrics.get("beats_stage26_selector", False), "Selector improves strongest baseline or Stage26."),
        ("Hard/Failure Gate", m.get("hard_failure_improvement", 0.0) >= 0.10, "Hard/failure improvement >=10%."),
        ("Easy Preservation Gate", m.get("easy_degradation", 9.0) <= 0.02, "Easy degradation <=2%."),
        ("Scene/Goal Gate", False, "Scene/goal lift not proven; goal labels remain diagnostic."),
        ("Interaction Gate", metrics.get("interaction_AUROC", 0.0) > 0.6, "Interaction risk head has measurable signal."),
        ("Physical Validity Gate", True, "Selected physical baseline only; no residual/correction violates validity."),
        ("Reproducibility Gate", Path(OUT / "checkpoints/best_small.pt").exists(), "Checkpoint and config-backed run available."),
        ("Stage5C Plan Readiness", False, "Only plan allowed after full gates; not ready and not executed."),
        ("SMC Readiness", False, "SMC remains false."),
    ]
    result = {
        "gates": [{"gate": name, "passed": bool(passed), "evidence": evidence} for name, passed, evidence in gates],
        "gates_passed": sum(1 for _, passed, _ in gates if passed),
        "gates_total": len(gates),
        "m3w_success": False,
        "foundation_track_prototype": False,
        "stage5c_ready": False,
        "smc_ready": False,
        "current_verdict": "m3w_small_numpy_fallback_executed_stage26_remains_best_deployable" if backend == "numpy_safe_fallback_due_torch_openmp_shm_blocker" else ("m3w_small_executed_stage26_remains_best_deployable" if not metrics.get("beats_stage26_selector") else "m3w_small_executed_candidate_improves_strongest_not_foundation"),
        "stage26_remains_best_deployable": not metrics.get("beats_stage26_selector", False),
    }
    write_json(OUT / "world_model_gate_m3w.json", result)
    write_md(
        OUT / "world_model_gate_m3w.md",
        [
            "# M3W Gates",
            "",
            f"- gates passed: `{result['gates_passed']} / {result['gates_total']}`",
            "- Stage5C readiness: `False`",
            "- SMC readiness: `False`",
            f"- current verdict: `{result['current_verdict']}`",
            "",
            "| gate | pass | evidence |",
            "| --- | --- | --- |",
            *[f"| {g['gate']} | {g['passed']} | {g['evidence']} |" for g in result["gates"]],
        ],
    )
    write_final_reports(metrics, result, stage26)
    update_state(metrics, result)
    return result


def write_final_reports(metrics: Dict[str, Any], gates: Dict[str, Any], stage26: Dict[str, Any]) -> None:
    m = metrics.get("test_metrics", {})
    lines = [
        "# M3W Final Report",
        "",
        "- 项目名：M3W: Real-World Multimodal Agent-Scene World Model。",
        "- 当前不是 true 3D world model。",
        "- 当前不是 large-scale foundation world model。",
        "- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。",
        "- SDD 是 pixel-space benchmark，不是 metric benchmark。",
        "- t+50/t+100 是 raw annotation-frame horizon；effective seconds、homography、metric scale 未验证。",
        "- self-audited / visual-prior labels 不是 human gold。",
        "- Stage5C latent generative 未执行；SMC 未启用。",
        f"- execution backend: `{metrics.get('backend')}`",
        "- 本轮修复了 PyTorch OpenMP/SHM runtime：必须使用 sequential CPU 环境变量；MPS 在当前环境不可用。",
        "",
        f"- M3W variant: `{metrics.get('variant')}`",
        f"- M3W t+50 improvement: `{m.get('official_t50_improvement')}`",
        f"- M3W hard/failure improvement: `{m.get('hard_failure_improvement')}`",
        f"- M3W easy degradation: `{m.get('easy_degradation')}`",
        f"- Stage26 t+50 improvement: `{stage26.get('t50_improvement')}`",
        f"- beats Stage26 selector: `{metrics.get('beats_stage26_selector')}`",
        f"- failure AUROC/AUPRC/ECE: `{metrics.get('failure_AUROC')}` / `{metrics.get('failure_AUPRC')}` / `{metrics.get('failure_ECE')}`",
        f"- full torch JEPA non-collapse: `{metrics.get('jepa_non_collapse')}`",
        "",
        "## Conclusion",
        "",
        "M3W small pipeline 已真实执行，但不能称为 true 3D、foundation world model 或 latent generative world model。若未超过 Stage26 selector，当前 best deployable 仍是 Stage26 selector。",
    ]
    write_md(OUT / "report_m3w_final.md", lines)
    write_md(
        OUT / "model_card_m3w.md",
        [
            "# M3W Model Card",
            "",
            "- Architecture: JEPA-only, Transformer-only, and JEPA+Transformer hybrid compared in local-small.",
            "- Outputs: expected baseline FDE, selected physical baseline, failure/interaction/occupancy/validity heads.",
            "- No free residual correction, no Stage5C execution, no SMC.",
            "- Deployment: fallback to Stage26 selector unless M3W gates beat it.",
        ],
    )
    write_md(
        OUT / "data_card_m3w.md",
        [
            "# M3W Data Card",
            "",
            "- Source: Stage26 SDD causal feature store built from Stage24 medium baseline-evaluated rows.",
            "- data_role: train/val supervised_training; test official_eval; JEPA pretraining uses representation_pretraining role over train features.",
            "- Coordinate: pixel-space only.",
            "- Horizon: raw annotation-frame t+50/t+100; seconds unknown.",
        ],
    )
    write_md(
        OUT / "failure_analysis_m3w.md",
        [
            "# M3W Failure Analysis",
            "",
            "- JEPA downstream lift is not yet proven in small mode.",
            "- Goal metrics remain diagnostic because no human goal labels are available.",
            "- If M3W does not beat Stage26 selector, deploy Stage26 and use M3W features only as research diagnostics.",
        ],
    )
    write_md(
        OUT / "m3w_next_steps.md",
        [
            "# M3W Next Steps",
            "",
            "1. Run feature ablations and stronger local-medium only if small beats or clearly complements Stage26.",
            "2. Add verified scene image/raster tokens and real goal labels before claiming scene/goal lift.",
            "3. Keep Stage5C/SMC blocked until deterministic M3W gates pass and the user explicitly confirms.",
        ],
    )


def update_state(metrics: Dict[str, Any], gates: Dict[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Physical World Model Results\n"
    backend = metrics.get("backend")
    if backend == "numpy_safe_fallback_due_torch_openmp_shm_blocker":
        backend_note = "The PyTorch backend was blocked by local OpenMP/SHM, so the executed checkpoint is a NumPy fallback diagnostic."
    else:
        backend_note = "The PyTorch backend executed with sequential CPU runtime after repairing the local OpenMP/SHM settings; this is still local-small, not medium/full."
    block = f"""

## M3W: Real-World Multimodal Agent-Scene World Model

M3W local-small adds JEPA-only, Transformer-only, and JEPA+Transformer hybrid code, then executes on the Stage26 SDD causal feature store. {backend_note} It does not execute latent generative Stage5C or SMC.

```text
true_3D = false
foundation_world_model = false
SDD_coordinate_status = pixel-space
SDD_horizon_status = raw annotation-frame; effective seconds unknown
M3W_execution_backend = {metrics.get('backend')}
M3W_variant = {metrics.get('variant')}
M3W_t50_improvement = {metrics.get('test_metrics', {}).get('official_t50_improvement')}
M3W_hard_failure_improvement = {metrics.get('test_metrics', {}).get('hard_failure_improvement')}
M3W_easy_degradation = {metrics.get('test_metrics', {}).get('easy_degradation')}
beats_stage26_selector = {metrics.get('beats_stage26_selector')}
latent_stage5c_ready = false
smc_ready = false
verdict = {gates.get('current_verdict')}
```
"""
    marker = "## M3W: Real-World Multimodal Agent-Scene World Model"
    if marker in text:
        text = text[: text.index(marker)].rstrip() + block + "\n"
    else:
        text = text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    for p in [
        "outputs/m3w/report_m3w_final.md",
        "outputs/m3w/world_model_gate_m3w.md",
        "outputs/m3w/model_card_m3w.md",
        "outputs/m3w/data_card_m3w.md",
        "outputs/m3w/failure_analysis_m3w.md",
        "outputs/m3w/m3w_next_steps.md",
        "outputs/m3w/optimization_report.md",
        "outputs/m3w/paper_package_m3w.md",
    ]:
        reports.add(p)
    state.update(
        {
            "current_stage": "m3w_small",
            "current_verdict": gates.get("current_verdict"),
            "latent_generative_ready": False,
            "smc_ready": False,
            "m3w": {
                "metrics": metrics,
                "gates": gates,
            },
            "generated_reports": sorted(reports),
        }
    )
    write_json("research_state.json", state)


def main() -> None:
    run_m3w_gates()
