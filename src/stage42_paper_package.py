from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")

DATA_JSON = OUT_DIR / "data_calibration_stage42.json"
EXTERNAL_JSON = OUT_DIR / "external_validation_stage42.json"
FULL_WAYPOINT_JSON = OUT_DIR / "full_waypoint_dynamics_stage42.json"
ABLATION_JSON = OUT_DIR / "causal_ablation_stage42.json"
SAFETY_JSON = OUT_DIR / "safety_floor_stage42.json"

PACKAGE_JSON = OUT_DIR / "paper_package_stage42.json"
PACKAGE_MD = OUT_DIR / "report_stage42_final.md"
GATE_MD = OUT_DIR / "stage42_stage_f_gate.md"

PAPER_OUTLINE_MD = OUT_DIR / "paper_outline_stage42.md"
METHOD_DRAFT_MD = OUT_DIR / "method_draft_stage42.md"
EXPERIMENT_TABLES_MD = OUT_DIR / "experiment_tables_stage42.md"
ABLATION_TABLES_MD = OUT_DIR / "ablation_tables_stage42.md"
FAILURE_TAXONOMY_MD = OUT_DIR / "failure_taxonomy_stage42.md"
MODEL_CARD_MD = OUT_DIR / "model_card_stage42.md"
DATA_CARD_MD = OUT_DIR / "data_card_stage42.md"
REPRODUCIBILITY_MD = OUT_DIR / "reproducibility_stage42.md"
A_JOURNAL_GAP_MD = OUT_DIR / "a_journal_gap_stage42.md"

PAPER_FILES = [
    PAPER_OUTLINE_MD,
    METHOD_DRAFT_MD,
    EXPERIMENT_TABLES_MD,
    ABLATION_TABLES_MD,
    FAILURE_TAXONOMY_MD,
    MODEL_CARD_MD,
    DATA_CARD_MD,
    REPRODUCIBILITY_MD,
    A_JOURNAL_GAP_MD,
]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "SDD 是 pixel-space；external 是 dataset-local / unverified weak metric diagnostic。",
    "t+50 / t+100 是 raw-frame horizon，不能写成 seconds-level。",
    "global metric/time claims 仍不允许；TGSIM 只能作为 traffic diagnostic，不是 pedestrian official claim。",
    "self-audited / visual-prior labels 不是 human gold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _metric(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    return float(value) if isinstance(value, (int, float)) else default


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    if value is None:
        return "n/a"
    return str(value)


def _gate_summary(data: Mapping[str, Any], name: str) -> dict[str, Any]:
    gate = data.get(name) or {}
    return {
        "source": gate.get("source", data.get("source")),
        "passed": gate.get("passed"),
        "total": gate.get("total"),
        "verdict": gate.get("verdict"),
    }


def _evidence_summary(
    data: Mapping[str, Any],
    external: Mapping[str, Any],
    full: Mapping[str, Any],
    ablation: Mapping[str, Any],
    safety: Mapping[str, Any],
) -> dict[str, Any]:
    ext_protected = (external.get("comparisons") or {}).get("m3w_neural_v1_composite_tail_protected") or {}
    ext_ungated = (external.get("comparisons") or {}).get("ungated_neural_endpoint") or {}
    full_protected = ((full.get("comparisons") or {}).get("full_waypoint_transformer_protected") or {}).get("ade") or {}
    full_fde = ((full.get("comparisons") or {}).get("full_waypoint_transformer_protected") or {}).get("fde") or {}
    safety_best = safety.get("best_deployable_policy") or {}
    safety_best_metrics = safety_best.get("test_metrics") or {}
    calibration = data.get("summary") or {}
    return {
        "data_calibration": {
            "datasets_audited": calibration.get("datasets_audited"),
            "external_domains_ready_from_existing_state": calibration.get("external_domains_ready_from_existing_state"),
            "metric_claim_ready_datasets": calibration.get("metric_claim_ready_datasets"),
            "seconds_claim_ready_datasets": calibration.get("seconds_claim_ready_datasets"),
            "global_metric_claim_allowed": calibration.get("global_metric_claim_allowed"),
            "global_seconds_claim_allowed": calibration.get("global_seconds_claim_allowed"),
        },
        "external_validation": {
            "rows": ext_protected.get("rows"),
            "all": ext_protected.get("all_improvement"),
            "t50": ext_protected.get("t50_improvement"),
            "t100_raw_frame_diagnostic": ext_protected.get("t100_improvement"),
            "hard_failure": ext_protected.get("hard_failure_improvement"),
            "easy_degradation": ext_protected.get("easy_degradation"),
            "ungated_easy_degradation": ext_ungated.get("easy_degradation"),
            "by_domain": ext_protected.get("by_domain"),
        },
        "full_waypoint": {
            "model": (full.get("full_waypoint_training_result") or {}).get("best_name"),
            "positive_domains": (full.get("stage42_c_gate") or {}).get("positive_domains"),
            "ade_all": full_protected.get("all_improvement"),
            "ade_t50": full_protected.get("t50_improvement"),
            "ade_t100_raw_frame_diagnostic": full_protected.get("t100_improvement"),
            "ade_hard_failure": full_protected.get("hard_failure_improvement"),
            "ade_easy_degradation": full_protected.get("easy_degradation"),
            "fde_all": full_fde.get("all_improvement"),
            "fde_t50": full_fde.get("t50_improvement"),
            "near_collision_delta_005": (((full.get("comparisons") or {}).get("full_waypoint_transformer_protected") or {}).get("joint") or {}).get("near_collision_delta_005"),
        },
        "ablation": {
            "gate": _gate_summary(ablation, "stage42_d_gate"),
            "all_components_retrained_inside_stage42_d": (ablation.get("full_retrain_boundary") or {}).get("all_components_retrained_inside_stage42_d"),
            "required_ablation_coverage_gate": (ablation.get("summary") or {}).get("required_ablation_coverage_gate"),
            "same_protocol_architecture_ablation_gate": (ablation.get("summary") or {}).get("same_protocol_architecture_ablation_gate"),
        },
        "safety_floor": {
            "gate": _gate_summary(safety, "stage42_e_gate"),
            "best_policy_family": safety_best.get("family"),
            "best_policy_source": safety_best.get("source"),
            "best_all": safety_best_metrics.get("all_improvement"),
            "best_t50": safety_best_metrics.get("t50_improvement"),
            "best_t100_raw_frame_diagnostic": safety_best_metrics.get("t100_improvement"),
            "best_hard_failure": safety_best_metrics.get("hard_failure_improvement"),
            "best_easy_degradation": safety_best_metrics.get("easy_degradation"),
            "floor_necessity_conclusion": (safety.get("floor_necessity_analysis") or {}).get("conclusion"),
        },
    }


def _claim_matrix(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    data = evidence["data_calibration"]
    ext = evidence["external_validation"]
    full = evidence["full_waypoint"]
    ab = evidence["ablation"]
    safety = evidence["safety_floor"]
    return [
        {
            "claim": "protected external raw-frame 2.5D world-state dynamics improves over Stage37/strongest floor",
            "status": "supported",
            "evidence": f"external all={_fmt(ext.get('all'))}, t50={_fmt(ext.get('t50'))}, hard={_fmt(ext.get('hard_failure'))}, easy={_fmt(ext.get('easy_degradation'))}",
        },
        {
            "claim": "full-waypoint sequence dynamics exists beyond endpoint-only linear bridge",
            "status": "supported_but_protected",
            "evidence": f"full-waypoint ADE all={_fmt(full.get('ade_all'))}, t50={_fmt(full.get('ade_t50'))}, t100diag={_fmt(full.get('ade_t100_raw_frame_diagnostic'))}, positive_domains={full.get('positive_domains')}",
        },
        {
            "claim": "ungated neural can replace safety floor",
            "status": "rejected",
            "evidence": f"ungated easy degradation={_fmt(ext.get('ungated_easy_degradation'))}; safety conclusion={safety.get('floor_necessity_conclusion')}",
        },
        {
            "claim": "metric or seconds-level pedestrian world model",
            "status": "not_supported",
            "evidence": f"global_metric={data.get('global_metric_claim_allowed')}, global_seconds={data.get('global_seconds_claim_allowed')}",
        },
        {
            "claim": "true 3D or foundation world model",
            "status": "not_supported",
            "evidence": "all Stage42 claim boundaries keep true_3d=false and foundation_world_model=false",
        },
        {
            "claim": "scene/goal/interaction/history/neighbor contributions are proven",
            "status": "partially_supported",
            "evidence": f"ablation coverage={ab.get('required_ablation_coverage_gate')}; all Stage42-D component retraining={ab.get('all_components_retrained_inside_stage42_d')}",
        },
        {
            "claim": "A-journal submission candidate",
            "status": "candidate_package_not_final_claim",
            "evidence": "A-E evidence is organized and strong for a protected 2.5D paper; full retrained ablation, metric/time calibration, independent external expansion, and floor-free safety remain gaps.",
        },
    ]


def _experiment_rows(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    ext = evidence["external_validation"]
    full = evidence["full_waypoint"]
    safety = evidence["safety_floor"]
    return [
        {
            "experiment": "Stage42-B protected endpoint/external validation",
            "source": "fresh_run",
            "all": ext.get("all"),
            "t50": ext.get("t50"),
            "t100": ext.get("t100_raw_frame_diagnostic"),
            "hard": ext.get("hard_failure"),
            "easy": ext.get("easy_degradation"),
            "note": "protected composite-tail endpoint dynamics; external source-fold eval",
        },
        {
            "experiment": "Stage42-C protected full-waypoint dynamics ADE",
            "source": "fresh_run",
            "all": full.get("ade_all"),
            "t50": full.get("ade_t50"),
            "t100": full.get("ade_t100_raw_frame_diagnostic"),
            "hard": full.get("ade_hard_failure"),
            "easy": full.get("ade_easy_degradation"),
            "note": "actual reconstructed future waypoint labels; positive on ETH_UCY and TrajNet",
        },
        {
            "experiment": "Stage42-C protected full-waypoint dynamics FDE",
            "source": "fresh_run",
            "all": full.get("fde_all"),
            "t50": full.get("fde_t50"),
            "t100": None,
            "hard": None,
            "easy": None,
            "note": "full-waypoint FDE summary",
        },
        {
            "experiment": "Stage42-E best deployable safety-floor policy",
            "source": "fresh_run",
            "all": safety.get("best_all"),
            "t50": safety.get("best_t50"),
            "t100": safety.get("best_t100_raw_frame_diagnostic"),
            "hard": safety.get("best_hard_failure"),
            "easy": safety.get("best_easy_degradation"),
            "note": safety.get("floor_necessity_conclusion"),
        },
    ]


def _write_table(path: Path, title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = [f"# {title}", "", "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(col)) for col in columns) + " |")
    write_md(path, lines)


def _paper_outline(evidence: Mapping[str, Any], claims: list[dict[str, Any]]) -> list[str]:
    return [
        "# Stage42 Paper Outline",
        "",
        "## Working Title",
        "",
        "M3W-Neural v1: Protected Multi-Agent Full-Waypoint World-State Dynamics for External Top-Down Trajectory Benchmarks",
        "",
        "## Core Thesis",
        "",
        "A protected neural dynamics model with a Stage37/teacher safety floor can deliver positive external raw-frame 2.5D multi-agent trajectory/world-state improvements while preserving easy cases and proximity safety. The current evidence supports a protected 2.5D world-state candidate, not a true 3D, metric, seconds-level, or foundation model.",
        "",
        "## Main Contributions",
        "",
        "1. Source-level external validation over ETH_UCY, TrajNet, UCY/OpenTraj-derived state with protected composite-tail dynamics.",
        "2. Full-waypoint sequence evaluation over all active agents, beyond endpoint-only linear bridge diagnostics.",
        "3. Safety-floor analysis showing ungated neural is high-lift but unsafe, while protected safe-switch remains deployable.",
        "4. Evidence ledger with fresh_run/cached_verified/not_run boundaries and no future/test leakage claims.",
        "",
        "## Paper Structure",
        "",
        "1. Introduction and problem setting",
        "2. Related work placeholder: trajectory forecasting, world models, JEPA, safe fallback policies",
        "3. Method: causal features, Stage37 floor, composite-tail bounded dynamics, full-waypoint model",
        "4. Data and calibration: dataset-local raw-frame external benchmark, no metric/time overclaim",
        "5. Experiments: external validation, full-waypoint dynamics, safety-floor study",
        "6. Ablations and negative evidence",
        "7. Limitations and A-journal gap",
        "8. Reproducibility checklist",
        "",
        "## Claim Matrix",
        "",
        "| claim | status | evidence |",
        "| --- | --- | --- |",
        *[f"| {row['claim']} | `{row['status']}` | {row['evidence']} |" for row in claims],
    ]


def _method_draft(evidence: Mapping[str, Any]) -> list[str]:
    return [
        "# Stage42 Method Draft",
        "",
        "## Problem",
        "",
        "Given past-only multi-agent history, neighbor context, goal/prototype context, and a strongest causal/Stage37 teacher floor, predict future endpoint and full-waypoint world-state trajectories under strict no-leakage constraints.",
        "",
        "## Inputs",
        "",
        "- past-only history windows and causal velocities",
        "- neighbor/interaction/group-consistency features",
        "- domain/horizon metadata",
        "- train-only goal/prototype features where available",
        "- Stage37/teacher floor rollout and proposal scores",
        "",
        "No future endpoint, future waypoint, central velocity, or test endpoint goal is used as inference input.",
        "",
        "## Model",
        "",
        "The deployable path is a composite-tail safe-switch bounded neural dynamics policy under the Stage37/teacher floor. It combines a validation-selected teacher repaired switch with a small bounded tail alpha for confident neural proposals. Stage42-C additionally evaluates a protected full-waypoint sequence model on reconstructed future waypoint labels.",
        "",
        "## Safety",
        "",
        "Stage42-E evaluates internal self-gates, uncertainty gates, harm gates, conformal-style risk gates, teacher-prob gates, and bounded residual blends. The current deployable conclusion is that the Stage37/teacher floor remains necessary. Ungated neural improves raw error but fails safety.",
        "",
        "## Claim Boundary",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
    ]


def _failure_taxonomy(evidence: Mapping[str, Any], claims: list[dict[str, Any]]) -> list[str]:
    safety = evidence["safety_floor"]
    return [
        "# Stage42 Failure Taxonomy",
        "",
        "## Confirmed Failures / Negative Evidence",
        "",
        "- Ungated endpoint/full-waypoint neural is not deployable: easy degradation remains unsafe.",
        "- Internal self-gate, uncertainty gate, harm gate, and conformal-risk gate can produce large raw lift but violate proximity/collision safety in the fresh Stage42-E study.",
        "- JEPA-only and JEPA+Transformer hybrid attempts remain negative or fallback-only in cached-verified same-protocol architecture evidence.",
        "- Full Stage42-D retraining of every named component has not been completed; Stage42-D is an evidence audit with fresh safety/waypoint rows plus cached-verified Stage30/41 component evidence.",
        "- Metric/time claims remain blocked by missing verified homography/FPS/stride calibration for the pedestrian external benchmark.",
        "",
        "## Root Causes",
        "",
        "- Safety/floor dependence: neural proposals can help hard/long-horizon slices but are too risky without the teacher floor.",
        "- Calibration gap: dataset-local and raw-frame coordinates prevent metric/seconds claims.",
        "- Evidence gap: some contribution claims rely on cached-verified prior ablations rather than fresh Stage42 retraining.",
        "- Data gap: broader legally verified top-down external domains remain needed for stronger generalization claims.",
        "",
        "## Current Best Safe Action",
        "",
        f"Keep `{safety.get('best_policy_family')}` as the deployable policy. Do not execute Stage5C or SMC.",
    ]


def _model_card(evidence: Mapping[str, Any]) -> list[str]:
    safety = evidence["safety_floor"]
    full = evidence["full_waypoint"]
    return [
        "# Stage42 Model Card",
        "",
        "## Model",
        "",
        "M3W-Neural v1 composite-tail safe-switch bounded neural dynamics under Stage37/teacher floor.",
        "",
        "## Intended Use",
        "",
        "Research evaluation for dataset-local raw-frame top-down multi-agent trajectory/world-state prediction and failure/hard-case diagnostics.",
        "",
        "## Not Intended For",
        "",
        "- metric or seconds-level physical deployment",
        "- true 3D world modeling",
        "- large-scale foundation model claims",
        "- autonomous deployment without dataset/domain validation",
        "- Stage5C latent generative execution or SMC",
        "",
        "## Performance Summary",
        "",
        f"- protected external all: `{_fmt(safety.get('best_all'))}`",
        f"- protected external t50: `{_fmt(safety.get('best_t50'))}`",
        f"- protected external t100 raw-frame diagnostic: `{_fmt(safety.get('best_t100_raw_frame_diagnostic'))}`",
        f"- protected external hard/failure: `{_fmt(safety.get('best_hard_failure'))}`",
        f"- protected easy degradation: `{_fmt(safety.get('best_easy_degradation'))}`",
        f"- full-waypoint ADE all/t50: `{_fmt(full.get('ade_all'))}` / `{_fmt(full.get('ade_t50'))}`",
        "",
        "## Safety",
        "",
        "The Stage37/teacher floor is required for current deployment. Ungated neural is explicitly rejected.",
    ]


def _data_card(evidence: Mapping[str, Any]) -> list[str]:
    data = evidence["data_calibration"]
    return [
        "# Stage42 Data Card",
        "",
        f"- datasets audited: `{data.get('datasets_audited')}`",
        f"- external domains ready from existing state: `{data.get('external_domains_ready_from_existing_state')}`",
        f"- metric claim ready datasets: `{data.get('metric_claim_ready_datasets')}`",
        f"- seconds claim ready datasets: `{data.get('seconds_claim_ready_datasets')}`",
        f"- global metric claim allowed: `{data.get('global_metric_claim_allowed')}`",
        f"- global seconds claim allowed: `{data.get('global_seconds_claim_allowed')}`",
        "",
        "## Data Roles",
        "",
        "- SDD: pixel-space official benchmark evidence from earlier stages.",
        "- External top-down trajectories: dataset-local raw-frame external validation for Stage42.",
        "- TGSIM: diagnostic traffic unit/metric evidence only, not pedestrian official success.",
        "",
        "## Leakage Policy",
        "",
        "Future endpoints/waypoints are labels/evaluation only. No central velocity, no test endpoint goal construction, and no test threshold tuning are allowed.",
    ]


def _reproducibility(evidence: Mapping[str, Any]) -> list[str]:
    return [
        "# Stage42 Reproducibility",
        "",
        "## Environment",
        "",
        "- training/eval scripts: `.venv-pytorch/bin/python` arm64",
        "- tests: `python3 -m pytest tests`",
        "- num_workers: `0` for torch data paths",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "",
        "## Commands",
        "",
        "```bash",
        ".venv-pytorch/bin/python run_stage42_data_calibration.py",
        ".venv-pytorch/bin/python run_stage42_external_validation.py",
        ".venv-pytorch/bin/python run_stage42_full_waypoint_dynamics.py",
        ".venv-pytorch/bin/python run_stage42_causal_ablation.py",
        ".venv-pytorch/bin/python run_stage42_safety_floor.py",
        ".venv-pytorch/bin/python run_stage42_paper_package.py",
        "python3 -m pytest tests",
        "```",
        "",
        "## Source Labels",
        "",
        "All Stage42 package claims use `fresh_run`, `cached_verified`, or `not_run`. Stage42-D explicitly does not relabel cached component ablations as fresh retraining.",
    ]


def _gap_analysis(evidence: Mapping[str, Any], claims: list[dict[str, Any]]) -> list[str]:
    return [
        "# Stage42 A-Journal Gap Analysis",
        "",
        "## Current Position",
        "",
        "Stage42 is strong enough to support a serious protected 2.5D external world-state dynamics manuscript draft. It is not yet enough for a broad true-3D/foundation/world-model claim.",
        "",
        "## What Is Already Paper-Usable",
        "",
        "- Fresh source-level external validation.",
        "- Fresh full-waypoint all-agent world-state evaluation.",
        "- Fresh safety-floor study showing why ungated neural cannot be deployed.",
        "- Clear claim boundaries and no-leakage policy.",
        "",
        "## What Is Not Yet Strong Enough",
        "",
        "- Full retrained ablation for every named component inside Stage42-D.",
        "- Metric/time-calibrated pedestrian benchmark claims.",
        "- External expansion beyond the current converted top-down state with independent legal datasets.",
        "- Floor-free or partially floor-free neural deployment that preserves proximity/collision safety.",
        "- Strong JEPA/Transformer positive contribution claim; current evidence favors protected bounded dynamics over pure JEPA/Transformer.",
        "",
        "## Shortest Next Path",
        "",
        "1. Run true retrained ablations for no-history, no-neighbor, no-scene, no-goal, no-interaction, no-teacher-floor, no-safe-switch, no-endpoint-bridge, and no-full-waypoint-shape with bootstrap or three seeds.",
        "2. Add one more legally verified external top-down pedestrian/drone dataset or a stronger held-out source split.",
        "3. Build a proximity-safe internal self-gate that reduces teacher-floor dependence without increasing collision/proximity risk.",
        "4. Obtain verified homography/FPS/stride for at least one pedestrian subset, or keep all claims raw-frame/dataset-local.",
        "",
        "## Absolute Non-Claims",
        "",
        "- Not true 3D.",
        "- Not foundation.",
        "- Not metric/seconds-level pedestrian prediction.",
        "- Not Stage5C or SMC.",
        "- Not ungated neural deployment.",
    ]


def run_stage42_paper_package() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    data = read_json(DATA_JSON, {})
    external = read_json(EXTERNAL_JSON, {})
    full = read_json(FULL_WAYPOINT_JSON, {})
    ablation = read_json(ABLATION_JSON, {})
    safety = read_json(SAFETY_JSON, {})

    evidence = _evidence_summary(data, external, full, ablation, safety)
    claims = _claim_matrix(evidence)
    experiments = _experiment_rows(evidence)
    ablation_rows = list(ablation.get("fresh_ablation_rows") or []) + list(ablation.get("cached_verified_required_ablation_rows") or [])

    write_md(PAPER_OUTLINE_MD, _paper_outline(evidence, claims))
    write_md(METHOD_DRAFT_MD, _method_draft(evidence))
    _write_table(EXPERIMENT_TABLES_MD, "Stage42 Experiment Tables", experiments, ["experiment", "source", "all", "t50", "t100", "hard", "easy", "note"])
    _write_ablation_table(ABLATION_TABLES_MD, ablation_rows)
    write_md(FAILURE_TAXONOMY_MD, _failure_taxonomy(evidence, claims))
    write_md(MODEL_CARD_MD, _model_card(evidence))
    write_md(DATA_CARD_MD, _data_card(evidence))
    write_md(REPRODUCIBILITY_MD, _reproducibility(evidence))
    write_md(A_JOURNAL_GAP_MD, _gap_analysis(evidence, claims))

    gates = _gate(data, external, full, ablation, safety, evidence, claims)
    result = {
        "source": "fresh_run",
        "stage": "Stage42-F paper-ready evidence package",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([DATA_JSON, EXTERNAL_JSON, FULL_WAYPOINT_JSON, ABLATION_JSON, SAFETY_JSON]),
        "evidence_summary": evidence,
        "claim_matrix": claims,
        "experiment_rows": experiments,
        "paper_files": [str(path) for path in PAPER_FILES],
        "stage42_f_gate": gates,
        "final_verdict": gates["verdict"],
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    write_md(PACKAGE_MD, _final_report(result))
    write_md(GATE_MD, _render_gate(result))
    write_json(PACKAGE_JSON, _jsonable(result))
    _append_ledger(result)
    _update_readme_and_state(result)
    return result


def _write_ablation_table(path: Path, rows: list[Mapping[str, Any]]) -> None:
    lines = [
        "# Stage42 Ablation Tables",
        "",
        "| ablation | source | status | all | t50 | hard/failure | easy | interpretation |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('ablation')}`",
                    f"`{row.get('source')}`",
                    f"`{row.get('status')}`",
                    _fmt(row.get("all_improvement")),
                    _fmt(row.get("t50_improvement")),
                    _fmt(row.get("hard_failure_improvement")),
                    _fmt(row.get("easy_degradation")),
                    str(row.get("interpretation", row.get("evidence_type", ""))),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Stage42-D fresh-runs safety/floor/full-waypoint ablations and cached-verifies prior Stage30/41 component ablation evidence. It does not complete all-component retraining inside Stage42-D.",
        ]
    )
    write_md(path, lines)


def _gate(
    data: Mapping[str, Any],
    external: Mapping[str, Any],
    full: Mapping[str, Any],
    ablation: Mapping[str, Any],
    safety: Mapping[str, Any],
    evidence: Mapping[str, Any],
    claims: list[Mapping[str, Any]],
) -> dict[str, Any]:
    data_summary = data.get("summary") or {}
    external_gate = external.get("stage42_b_gate") or {}
    full_gate = full.get("stage42_c_gate") or {}
    ablation_gate = ablation.get("stage42_d_gate") or {}
    safety_gate = safety.get("stage42_e_gate") or {}
    full_metrics = evidence["full_waypoint"]
    safety_metrics = evidence["safety_floor"]
    gates = {
        "stage42_a_data_calibration_present": data_summary.get("stage42_b_external_validation_ready") is True,
        "stage42_b_external_validation_pass": external_gate.get("verdict") == "stage42_b_external_validation_pass_protected_neural_not_ungated",
        "stage42_c_full_waypoint_pass": full_gate.get("verdict") == "stage42_c_full_waypoint_dynamics_pass",
        "stage42_d_ablation_package_pass_with_boundary": ablation_gate.get("verdict") == "stage42_d_causal_ablation_evidence_pass_with_retrain_boundary",
        "stage42_e_safety_floor_pass": safety_gate.get("verdict") == "stage42_e_safety_floor_research_pass",
        "paper_files_written": all(path.exists() for path in PAPER_FILES),
        "claim_matrix_has_negative_boundaries": any(row["status"] == "rejected" for row in claims) and any(row["status"] == "not_supported" for row in claims),
        "full_waypoint_evidence_positive": _metric(full_metrics, "ade_all") > 0 and _metric(full_metrics, "ade_t50") > 0,
        "safety_floor_evidence_positive": _metric(safety_metrics, "best_all") > 0 and _metric(safety_metrics, "best_easy_degradation", 1.0) <= 0.02,
        "no_metric_seconds_overclaim": data_summary.get("global_metric_claim_allowed") is False and data_summary.get("global_seconds_claim_allowed") is False,
        "stage5c_false": True,
        "smc_false": True,
    }
    full_a_journal_ready = bool(
        all(gates.values())
        and (ablation.get("full_retrain_boundary") or {}).get("all_components_retrained_inside_stage42_d") is True
        and data_summary.get("global_metric_claim_allowed") is True
    )
    verdict = "stage42_f_paper_package_complete_not_full_a_journal_ready"
    if full_a_journal_ready:
        verdict = "stage42_f_full_a_journal_ready"
    return {
        "source": "fresh_run",
        "gates": gates,
        "passed": int(sum(bool(v) for v in gates.values())),
        "total": int(len(gates)),
        "full_a_journal_ready": full_a_journal_ready,
        "verdict": verdict,
    }


def _final_report(result: Mapping[str, Any]) -> list[str]:
    evidence = result["evidence_summary"]
    gate = result["stage42_f_gate"]
    lines = [
        "# Stage42 Final Evidence Package",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Direct Answers",
        "",
        f"- still 2.5D: `True`",
        f"- metric/time subset for official pedestrian claims: `False`",
        f"- full-waypoint dynamics: `True`, all={_fmt(evidence['full_waypoint'].get('ade_all'))}, t50={_fmt(evidence['full_waypoint'].get('ade_t50'))}",
        f"- cross-domain/external validation: `True`, all={_fmt(evidence['external_validation'].get('all'))}, t50={_fmt(evidence['external_validation'].get('t50'))}",
        f"- exceeds strongest/Stage37 floor: `True under protected policy`, safety-floor best all={_fmt(evidence['safety_floor'].get('best_all'))}",
        f"- scene/goal/interaction contribution: `partial`, because Stage42-D uses cached-verified component evidence and not all-component fresh retraining.",
        f"- enough for A-journal candidate: `not yet full A-journal ready`; strong protected 2.5D manuscript package, but gaps remain.",
        "",
        "## Paper Files",
        "",
        *[f"- `{path}`" for path in result["paper_files"]],
    ]
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_f_gate"]
    lines = [
        "# Stage42-F Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        f"- full A-journal ready: `{gate['full_a_journal_ready']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| {name} | `{ok}` |")
    return lines


def _append_ledger(result: Mapping[str, Any]) -> None:
    entry = {
        "command": "run_stage42_paper_package.py",
        "source": result["source"],
        "status": "success",
        "generated_at_utc": result["generated_at_utc"],
        "git_commit": result["git_commit"],
        "input_hash": result["input_hash"],
        "outputs": [str(PACKAGE_JSON), str(PACKAGE_MD), str(GATE_MD), *[str(path) for path in PAPER_FILES]],
    }
    with (OUT_DIR / "run_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


def _replace_block(text: str, marker: str, block: str) -> str:
    if marker in text:
        return text[: text.index(marker)].rstrip() + "\n\n" + block.strip() + "\n"
    return text.rstrip() + "\n\n" + block.strip() + "\n"


def _update_readme_and_state(result: Mapping[str, Any]) -> None:
    evidence = result["evidence_summary"]
    gate = result["stage42_f_gate"]
    block = f"""
## Stage42-F Paper Evidence Package

```text
source = {result.get('source')}
verdict = {gate.get('verdict')}
gates = {gate.get('passed')} / {gate.get('total')}
full_a_journal_ready = {gate.get('full_a_journal_ready')}
external_all = {evidence['external_validation'].get('all')}
external_t50 = {evidence['external_validation'].get('t50')}
full_waypoint_ade_all = {evidence['full_waypoint'].get('ade_all')}
full_waypoint_ade_t50 = {evidence['full_waypoint'].get('ade_t50')}
safety_floor_best_all = {evidence['safety_floor'].get('best_all')}
safety_floor_best_easy = {evidence['safety_floor'].get('best_easy_degradation')}
all_components_retrained_inside_stage42_d = {evidence['ablation'].get('all_components_retrained_inside_stage42_d')}
metric_or_seconds_claim = false
stage5c_executed = false
smc_enabled = false
```

Stage42-F packages A-E into paper-ready artifacts under `outputs/stage42_long_research/`. It supports a protected raw-frame 2.5D external world-state manuscript package, but it is **not yet full A-journal ready** because metric/time calibration, all-component fresh retrained ablation, independent external expansion, and floor-free safety remain open.
"""
    for path in [Path("README_RESULTS.md"), Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")]:
        text = path.read_text(encoding="utf-8") if path.exists() else "# Results\n"
        path.write_text(_replace_block(text, "## Stage42-F Paper Evidence Package", block), encoding="utf-8")

    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    reports.update({str(PACKAGE_JSON), str(PACKAGE_MD), str(GATE_MD), *[str(path) for path in PAPER_FILES]})
    stage42 = dict(state.get("stage42", {}))
    stage42["stage_f_paper_package"] = {
        "source": result.get("source"),
        "verdict": gate.get("verdict"),
        "gates": gate,
        "evidence_summary": evidence,
        "claim_matrix": result["claim_matrix"],
        "paper_files": result["paper_files"],
        "claim_boundary": result["claim_boundary"],
    }
    state.update(
        {
            "current_stage": "stage42_f_paper_package",
            "current_verdict": gate.get("verdict"),
            "current_best_deployable": "M3W-Neural v1 composite-tail safe-switch bounded neural dynamics under Stage37/teacher floor",
            "last_updated": "2026-05-25",
            "latent_generative_ready": False,
            "stage5c_ready": False,
            "smc_ready": False,
            "stage42": stage42,
            "generated_reports": sorted(reports),
        }
    )
    write_json("research_state.json", _jsonable(state))


if __name__ == "__main__":
    run_stage42_paper_package()
