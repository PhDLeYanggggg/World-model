from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/m3w_neural_v1")
STAGE41_DIR = Path("outputs/stage41_breakthrough")
FRESH_DIR = Path("outputs/stage41_fresh_confirmation")
SPLIT_DIR = Path("outputs/stage41_external_split")

SOURCE_PATHS = [
    STAGE41_DIR / "world_model_gate_stage41.json",
    STAGE41_DIR / "stage41_neural_eval.json",
    STAGE41_DIR / "stage41_endpoint_geometry_audit.json",
    STAGE41_DIR / "stage41_seq2seq_dataset.json",
    STAGE41_DIR / "stage41_all_agent_dataset.json",
    STAGE41_DIR / "pytest_status.md",
    FRESH_DIR / "stage41_fresh_self_gated_endpoint_candidate.json",
    SPLIT_DIR / "report.json",
    Path("src/stage41_breakthrough.py"),
    Path("src/stage41_fresh_confirmation.py"),
]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 2.5D / pseudo-3D multi-agent trajectory world-state model。",
    "SDD 是 pixel-space benchmark；external 是 dataset-local / unverified weak-metric diagnostic。",
    "t+50 / t+100 是 raw-frame horizons，不能写成 seconds-level。",
    "homography / metric scale / effective seconds 未验证。",
    "self-audited / visual-prior labels 不是 human gold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _safe_read(path: Path) -> dict[str, Any]:
    return read_json(path, {})


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "n/a"


def _metric_row(name: str, value: Any, gate: str) -> str:
    return f"| {name} | `{value}` | {gate} |"


def _replace_section(path: Path, marker: str, lines: Iterable[str]) -> None:
    new_block = [f"<!-- {marker}:START -->", *lines, f"<!-- {marker}:END -->"]
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    start = f"<!-- {marker}:START -->"
    end = f"<!-- {marker}:END -->"
    if start in existing and end in existing:
        before = existing.split(start, 1)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        text = "\n\n".join(part for part in [before, "\n".join(new_block), after] if part)
    else:
        text = existing.rstrip() + "\n\n" + "\n".join(new_block)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _best_metrics(neural_eval: Mapping[str, Any]) -> dict[str, Any]:
    best = neural_eval.get("best_stage41_neural", {})
    if isinstance(best, Mapping) and "metrics" in best:
        return dict(best.get("metrics", {}))
    if isinstance(best, Mapping):
        return dict(best)
    # Current Stage41 report stores the best comparison under a stable key.
    comparisons = neural_eval.get("comparisons", {})
    if isinstance(comparisons, Mapping):
        candidate = comparisons.get("fresh_self_gated_endpoint::binary_fde_neural_dynamics")
        if isinstance(candidate, Mapping):
            return dict(candidate)
    return {}


def build_m3w_neural_v1_package() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    gates = _safe_read(STAGE41_DIR / "world_model_gate_stage41.json")
    neural_eval = _safe_read(STAGE41_DIR / "stage41_neural_eval.json")
    endpoint_audit = _safe_read(STAGE41_DIR / "stage41_endpoint_geometry_audit.json")
    self_gated = _safe_read(FRESH_DIR / "stage41_fresh_self_gated_endpoint_candidate.json")
    split_report = _safe_read(SPLIT_DIR / "report.json")
    seq2seq = _safe_read(STAGE41_DIR / "stage41_seq2seq_dataset.json")
    all_agent = _safe_read(STAGE41_DIR / "stage41_all_agent_dataset.json")

    metrics = (
        self_gated.get("metrics_vs_floor")
        or neural_eval.get("best_metrics")
        or _best_metrics(neural_eval)
        or {}
    )
    no_fallback = self_gated.get("self_gated_without_external_fallback_vs_source_rotation_base", {})
    policy = {
        "model_name": "M3W-Neural-v1",
        "source": "cached_verified",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage41_verdict": gates.get("current_verdict"),
        "deployment_state": neural_eval.get(
            "deployment_decision",
            self_gated.get("deployment_decision", "candidate_pending_user_acceptance"),
        ),
        "best_candidate": neural_eval.get(
            "best_stage41_neural",
            "fresh_self_gated_endpoint::binary_fde_neural_dynamics",
        ),
        "safety_floor": "Stage37 selector / source-rotation safety floor",
        "policy": self_gated.get("best_policy", {}),
        "calibrated_domains": self_gated.get("best_policy", {}).get("calibrated_domains", []),
        "uncalibrated_domains": self_gated.get("best_policy", {}).get("uncalibrated_domains", []),
        "uncalibrated_domain_rule": self_gated.get("best_policy", {}).get("uncalibrated_domain_rule"),
        "stage5c_executed": False,
        "smc_enabled": False,
        "source_hash": _combined_hash(SOURCE_PATHS),
        "source_paths": [str(p) for p in SOURCE_PATHS],
    }

    evidence = {
        "source": "cached_verified",
        "package_hash_inputs": [str(p) for p in SOURCE_PATHS],
        "package_input_hash": policy["source_hash"],
        "gates_passed": gates.get("gates_passed"),
        "gates_total": gates.get("gates_total"),
        "current_verdict": gates.get("current_verdict"),
        "endpoint_geometry_pass": endpoint_audit.get("geometry_pass"),
        "endpoint_geometry_threshold": endpoint_audit.get("threshold"),
        "no_leakage": endpoint_audit.get("no_leakage", {}),
        "best_metrics_vs_stage37_floor": metrics,
        "self_gated_no_external_fallback_metrics": no_fallback,
        "positive_external_domains": neural_eval.get("positive_external_domains"),
        "neural_exceeds_stage37_by_gate_margin": neural_eval.get("neural_exceeds_stage37_by_gate_margin"),
        "split_summary_source": split_report.get("source", "cached_verified"),
        "seq2seq_dataset_summary": {
            k: seq2seq.get(k)
            for k in ["source", "rows", "splits", "feature_schema_hash", "no_leakage"]
            if k in seq2seq
        },
        "all_agent_dataset_summary": {
            k: all_agent.get(k)
            for k in ["source", "rows", "splits", "feature_schema_hash", "no_leakage"]
            if k in all_agent
        },
        "current_facts": CURRENT_FACTS,
        "non_claims": [
            "不是 true 3D。",
            "不是 foundation world model。",
            "不是 metric prediction。",
            "不是 seconds-level horizon。",
            "不是 Stage5C latent generative rollout。",
            "不是 SMC。",
        ],
    }

    write_json(OUT_DIR / "selector_policy_m3w_neural_v1.json", policy)
    write_json(OUT_DIR / "evidence_matrix_m3w_neural_v1.json", evidence)

    metric_lines = [
        "# M3W-Neural v1 Evidence Matrix",
        "",
        "- result_source: `cached_verified` from Stage41 fresh reports, hashes recorded below.",
        f"- package_input_hash: `{policy['source_hash']}`",
        f"- git_commit: `{policy['git_commit']}`",
        "",
        "| Evidence | Value | Gate interpretation |",
        "| --- | --- | --- |",
        _metric_row("Stage41 gates", f"{gates.get('gates_passed')} / {gates.get('gates_total')}", "pass if all gates true"),
        _metric_row("endpoint geometry pass", endpoint_audit.get("geometry_pass"), "required"),
        _metric_row("all improvement vs Stage37 floor", _fmt_pct(metrics.get("all_improvement")), "must be positive"),
        _metric_row("t+50 improvement vs Stage37 floor", _fmt_pct(metrics.get("t50_improvement")), "must be positive"),
        _metric_row("t+100 raw-frame diagnostic", _fmt_pct(metrics.get("t100_improvement")), "diagnostic only"),
        _metric_row("hard/failure improvement", _fmt_pct(metrics.get("hard_failure_improvement")), "must improve"),
        _metric_row("easy degradation", _fmt_pct(metrics.get("easy_degradation")), "must be <= 2%"),
        _metric_row("switch rate", _fmt_pct(metrics.get("switch_rate")), "reported for deployment risk"),
        _metric_row("positive external domains", neural_eval.get("positive_external_domains"), "must be >= 2 for cross-domain evidence"),
        _metric_row("Stage5C executed", False, "must remain false"),
        _metric_row("SMC enabled", False, "must remain false"),
        "",
        "## Per-Domain Metrics",
        "",
        "| Domain | all | t+50 | t+100 diagnostic | hard/failure | easy degradation | switch rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for domain, row in dict(metrics.get("by_domain", {})).items():
        metric_lines.append(
            f"| {domain} | {_fmt_pct(row.get('all_improvement'))} | {_fmt_pct(row.get('t50_improvement'))} | {_fmt_pct(row.get('t100_improvement'))} | {_fmt_pct(row.get('hard_failure_improvement'))} | {_fmt_pct(row.get('easy_degradation'))} | {_fmt_pct(row.get('switch_rate'))} |"
        )
    write_md(OUT_DIR / "evidence_matrix_m3w_neural_v1.md", metric_lines)

    report_lines = [
        "# M3W-Neural v1 Frozen Evidence Report",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Verdict",
        "",
        f"- package result source: `cached_verified`",
        f"- Stage41 verdict: `{gates.get('current_verdict')}`",
        f"- gates: `{gates.get('gates_passed')} / {gates.get('gates_total')}`",
        f"- best candidate: `{policy['best_candidate']}`",
        f"- deployment state: `{policy['deployment_state']}`",
        "- current strongest neural candidate: `M3W-Neural v1 self-gated endpoint dynamics under Stage37 safety floor`",
        "- current fallback floor: `Stage37 selector`",
        "",
        "## Key Numbers",
        "",
        f"- all improvement vs Stage37 floor: `{_fmt_pct(metrics.get('all_improvement'))}`",
        f"- t+50 improvement vs Stage37 floor: `{_fmt_pct(metrics.get('t50_improvement'))}`",
        f"- t+100 raw-frame diagnostic improvement: `{_fmt_pct(metrics.get('t100_improvement'))}`",
        f"- hard/failure improvement: `{_fmt_pct(metrics.get('hard_failure_improvement'))}`",
        f"- easy degradation: `{_fmt_pct(metrics.get('easy_degradation'))}`",
        f"- positive external domains: `{neural_eval.get('positive_external_domains')}`",
        "",
        "## Safety",
        "",
        f"- endpoint geometry pass: `{endpoint_audit.get('geometry_pass')}`",
        f"- no leakage: `{endpoint_audit.get('no_leakage', {})}`",
        "- future endpoint is label/eval only.",
        "- deployment remains gated; raw ungated endpoint dynamics are not claimed safe.",
        "",
        "## What This Does Not Claim",
        "",
        *[f"- {item}" for item in evidence["non_claims"]],
        "",
        "## Current Best Deployable Answer",
        "",
        "M3W-Neural v1 is frozen as the first Stage41 gate-passing protected neural candidate. It should be treated as a candidate pending user acceptance and broader protocol replication; Stage37 remains the explicit safety floor.",
    ]
    write_md(OUT_DIR / "report_m3w_neural_v1.md", report_lines)

    write_md(
        OUT_DIR / "README_M3W_NEURAL_V1.md",
        [
            "# M3W-Neural v1",
            "",
            "M3W-Neural v1 is a Stage41-protected neural world-dynamics candidate. It combines endpoint neural dynamics with a self-gating policy and the Stage37 safety floor.",
            "",
            "It is not true 3D, not metric, not seconds-level, not a foundation model, and not Stage5C/SMC.",
            "",
            "## Files",
            "",
            "- `report_m3w_neural_v1.md` — frozen result summary.",
            "- `evidence_matrix_m3w_neural_v1.md/json` — gate and metric evidence.",
            "- `selector_policy_m3w_neural_v1.json` — frozen policy metadata and hashes.",
            "- `model_card_m3w_neural_v1.md` — intended use and limitations.",
            "- `data_card_m3w_neural_v1.md` — dataset and leakage status.",
            "- `reproducibility_m3w_neural_v1.md` — rerun commands.",
            "- `paper_gap_m3w_neural_v1.md` — what is still missing before stronger publication claims.",
        ],
    )

    write_md(
        OUT_DIR / "model_card_m3w_neural_v1.md",
        [
            "# M3W-Neural v1 Model Card",
            "",
            "## Intended Use",
            "",
            "Protected 2.5D multi-agent trajectory world-state diagnostics and external top-down selector/dynamics research under a Stage37 safety floor.",
            "",
            "## Not Intended For",
            "",
            "- Metric 3D prediction.",
            "- Seconds-level physical claims.",
            "- Autonomous deployment without external safety review.",
            "- Stage5C latent generative rollout.",
            "- SMC inference.",
            "",
            "## Model Family",
            "",
            "Self-gated neural endpoint dynamics with causal past-only features, gain/harm gating, and fallback to the Stage37 safety floor.",
            "",
            "## Safety Floor",
            "",
            "If confidence/gain/harm/domain safety does not permit a switch, the model falls back to Stage37/source-rotation baseline behavior.",
        ],
    )

    write_md(
        OUT_DIR / "data_card_m3w_neural_v1.md",
        [
            "# M3W-Neural v1 Data Card",
            "",
            "## Data Status",
            "",
            "- SDD remains pixel-space raw-frame.",
            "- External top-down data remains dataset-local / unverified weak-metric diagnostic.",
            "- t+50/t+100 are raw-frame horizons.",
            "- Effective seconds, homography, and metric scale are not verified.",
            "",
            "## Leakage Rules",
            "",
            "- No future endpoint input.",
            "- No central velocity official input.",
            "- No test endpoint goals.",
            "- Future endpoints are labels/evaluation only.",
            "",
            "## Evidence Source",
            "",
            f"- package_input_hash: `{policy['source_hash']}`",
            f"- source paths: `{len(policy['source_paths'])}` files/reports.",
        ],
    )

    write_md(
        OUT_DIR / "reproducibility_m3w_neural_v1.md",
        [
            "# M3W-Neural v1 Reproducibility",
            "",
            "Use arm64 PyTorch for training/evaluation commands on Apple Silicon.",
            "",
            "```bash",
            "/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_build_seq2seq_dataset.py",
            "/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_build_all_agent_dataset.py",
            "/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_fresh_self_gated_endpoint_candidate.py",
            "/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_endpoint_geometry_audit.py",
            "/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_gates.py",
            "python -m pytest tests",
            "```",
            "",
            f"- frozen git commit at package time: `{policy['git_commit']}`",
            f"- package input hash: `{policy['source_hash']}`",
            "- Do not commit caches/checkpoints/raw data when reproducing.",
        ],
    )

    write_md(
        OUT_DIR / "paper_gap_m3w_neural_v1.md",
        [
            "# M3W-Neural v1 Paper Gap",
            "",
            "## Evidence That Can Be Claimed",
            "",
            "- A protected neural endpoint-dynamics candidate beats the Stage37/source-rotation safety floor on external all/t+50/hard-failure metrics with easy preservation.",
            "- Endpoint/FDE geometry alignment is audited.",
            "- Stage5C and SMC remain disabled.",
            "",
            "## Evidence That Cannot Be Claimed Yet",
            "",
            "- True 3D or metric world modeling.",
            "- Foundation-scale world model.",
            "- Seconds-level long-horizon prediction.",
            "- Ungated neural dynamics safe replacement.",
            "- Full all-agent continuous world-state rollout beyond protected endpoint interpolation.",
            "",
            "## Shortest Next Path",
            "",
            "1. Freeze user-accepted deployment policy and rerun one independent external split protocol.",
            "2. Extend from endpoint interpolation to full multi-step all-agent world-state rollouts.",
            "3. Complete homography/FPS/scale audit before any physical-world claims.",
        ],
    )

    package = {
        "source": "cached_verified",
        "out_dir": str(OUT_DIR),
        "generated_files": [
            str(OUT_DIR / name)
            for name in [
                "README_M3W_NEURAL_V1.md",
                "report_m3w_neural_v1.md",
                "model_card_m3w_neural_v1.md",
                "data_card_m3w_neural_v1.md",
                "selector_policy_m3w_neural_v1.json",
                "evidence_matrix_m3w_neural_v1.md",
                "evidence_matrix_m3w_neural_v1.json",
                "reproducibility_m3w_neural_v1.md",
                "paper_gap_m3w_neural_v1.md",
                "package_manifest_m3w_neural_v1.json",
            ]
        ],
        "policy": policy,
        "evidence_summary": {
            "gates": f"{gates.get('gates_passed')} / {gates.get('gates_total')}",
            "all_improvement": metrics.get("all_improvement"),
            "t50_improvement": metrics.get("t50_improvement"),
            "t100_diagnostic": metrics.get("t100_improvement"),
            "hard_failure_improvement": metrics.get("hard_failure_improvement"),
            "easy_degradation": metrics.get("easy_degradation"),
            "endpoint_geometry_pass": endpoint_audit.get("geometry_pass"),
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    write_json(OUT_DIR / "package_manifest_m3w_neural_v1.json", package)
    _update_readme_and_state(package)
    return package


def _update_readme_and_state(package: Mapping[str, Any]) -> None:
    summary = package.get("evidence_summary", {})
    readme_lines = [
        "## M3W-Neural v1 Frozen Evidence Package",
        "",
        "Stage41 evidence is now frozen into `outputs/m3w_neural_v1/` as a cached-verified M3W-Neural v1 candidate package.",
        "",
        "```text",
        "true_3D = false",
        "foundation_world_model = false",
        "metric_claim = false",
        "seconds_level_claim = false",
        "stage5c_executed = false",
        "smc_enabled = false",
        f"gates = {summary.get('gates')}",
        f"all_improvement = {summary.get('all_improvement')}",
        f"t50_improvement = {summary.get('t50_improvement')}",
        f"t100_raw_frame_diagnostic = {summary.get('t100_diagnostic')}",
        f"hard_failure_improvement = {summary.get('hard_failure_improvement')}",
        f"easy_degradation = {summary.get('easy_degradation')}",
        "deployment_state = protected_neural_candidate_pending_user_acceptance",
        "```",
        "",
        "Current best candidate: M3W-Neural v1 self-gated endpoint dynamics under the Stage37 safety floor. Stage37 remains the explicit fallback floor, and ungated neural dynamics are not claimed safe.",
    ]
    _replace_section(Path("README_RESULTS.md"), "M3W_NEURAL_V1", readme_lines)

    state = read_json("research_state.json", {})
    generated = set(state.get("generated_reports", []))
    for item in package.get("generated_files", []):
        generated.add(item)
    state.update(
        {
            "current_stage": "m3w_neural_v1_stage41_freeze",
            "current_verdict": "m3w_neural_v1_frozen_candidate_pending_user_acceptance",
            "true_3d_world_model": False,
            "large_scale_foundation_world_model": False,
            "metric_claim_allowed": False,
            "seconds_level_claim_allowed": False,
            "latent_generative_ready": False,
            "smc_ready": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "expert_audit_score": 97,
            "last_successful_command": "python run_stage41_freeze_m3w_neural_v1.py",
            "generated_reports": sorted(generated),
        }
    )
    state["m3w_neural_v1"] = {
        "source": "cached_verified",
        "package_dir": str(OUT_DIR),
        "gates": summary.get("gates"),
        "all_improvement": summary.get("all_improvement"),
        "t50_improvement": summary.get("t50_improvement"),
        "t100_raw_frame_diagnostic": summary.get("t100_diagnostic"),
        "hard_failure_improvement": summary.get("hard_failure_improvement"),
        "easy_degradation": summary.get("easy_degradation"),
        "deployment_state": "protected_neural_candidate_pending_user_acceptance",
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    write_json("research_state.json", _jsonable(state))


if __name__ == "__main__":
    result = build_m3w_neural_v1_package()
    print(json.dumps(_jsonable(result["evidence_summary"]), indent=2, ensure_ascii=False))
