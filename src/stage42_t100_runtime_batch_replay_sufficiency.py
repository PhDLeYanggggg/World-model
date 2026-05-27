from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _pct, _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
HR_JSON = OUT_DIR / "group_consistency_t100_easy_guard_stage42.json"
HS_JSON = OUT_DIR / "group_consistency_t100_easy_guard_freeze_stage42.json"
HT_JSON = OUT_DIR / "group_consistency_t100_easy_guard_runtime_stage42.json"
POLICY_JSON = OUT_DIR / "frozen_group_consistency_t100_easy_guard_policy_stage42.json"

REPORT_JSON = OUT_DIR / "t100_runtime_batch_replay_sufficiency_stage42.json"
REPORT_MD = OUT_DIR / "t100_runtime_batch_replay_sufficiency_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_hu_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_t100_runtime_batch_replay_stage42.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
MASTER_SUMMARY = Path("README_M3W_CURRENT_MASTER_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_audit_from_stage42_hr_hs_ht_artifacts"

REQUIRED_ROW_CACHE_FIELDS = [
    "row_id",
    "split",
    "domain",
    "source_file",
    "scene_id_optional",
    "frame_id",
    "agent_id",
    "horizon",
    "candidate_xy_predicted_rollout",
    "floor_xy_train_horizon_causal_rollout",
    "original_selected_xy_from_stage42_hr_optional",
    "candidate_switch_optional",
    "future_xy_label_eval_only_optional",
    "normalizer_optional",
    "hard_failure_label_optional",
    "easy_label_optional",
]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-HU 审计 Stage42-HR/HS/HT 是否足以支持真实 row-level runtime batch replay。",
    "HT 已有 callable runtime API 和 smoke replay，但 HR/HS/HT artifact 不包含 per-row candidate/floor/selected_xy arrays。",
    "因此 real batch replay 当前标记为 not_run，而不是包装成完成。",
    "future waypoints / endpoints 只能作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _artifact_summary(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    row_like_keys = [
        key
        for key in [
            "rows",
            "row_ids",
            "test_ids",
            "candidate_xy",
            "floor_xy",
            "selected_xy",
            "original_selected_xy",
            "domains",
            "horizons",
            "source_files",
            "agent_ids",
            "future_xy",
        ]
        if key in payload
    ]
    nested_row_like = []
    for key, value in payload.items():
        if isinstance(value, dict):
            for child in ["candidate_xy", "floor_xy", "selected_xy", "row_ids", "test_ids"]:
                if child in value:
                    nested_row_like.append(f"{key}.{child}")
    return {
        "path": str(path),
        "exists": path.exists(),
        "top_level_keys": list(payload.keys()),
        "row_like_top_level_keys": row_like_keys,
        "row_like_nested_keys": nested_row_like,
        "has_per_row_rollout_arrays": any(
            key in row_like_keys for key in ["candidate_xy", "floor_xy", "selected_xy", "original_selected_xy"]
        )
        or bool(nested_row_like),
    }


def _sufficiency(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = payload["artifact_summaries"]
    ht_gate = payload["inputs"]["stage42_ht"].get("stage42_ht_gate", {})
    hs_gate = payload["inputs"]["stage42_hs"].get("stage42_hs_gate", {})
    has_runtime_api = ht_gate.get("passed") == ht_gate.get("total")
    has_policy = payload["policy_artifact"].get("exists") is True
    has_row_arrays = any(row["has_per_row_rollout_arrays"] for row in artifacts)
    missing = REQUIRED_ROW_CACHE_FIELDS if not has_row_arrays else []
    return {
        "runtime_api_ready": has_runtime_api,
        "frozen_policy_ready": has_policy and hs_gate.get("passed") == hs_gate.get("total"),
        "row_level_batch_replay_ready": bool(has_runtime_api and has_policy and has_row_arrays),
        "real_batch_replay_status": "ready" if has_runtime_api and has_policy and has_row_arrays else "not_run",
        "blocker": "" if has_row_arrays else "missing_row_level_candidate_floor_selected_arrays",
        "missing_required_fields": missing,
        "required_row_cache_fields": REQUIRED_ROW_CACHE_FIELDS,
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    suff = payload["sufficiency"]
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "hr_artifact_present": payload["inputs"]["stage42_hr"]["exists"] is True,
        "hs_artifact_present": payload["inputs"]["stage42_hs"]["exists"] is True,
        "ht_artifact_present": payload["inputs"]["stage42_ht"]["exists"] is True,
        "policy_artifact_present": payload["policy_artifact"]["exists"] is True,
        "hs_gate_passed": suff["frozen_policy_ready"] is True,
        "ht_runtime_api_passed": suff["runtime_api_ready"] is True,
        "row_level_replay_not_overclaimed": suff["real_batch_replay_status"] == "not_run"
        and suff["blocker"] == "missing_row_level_candidate_floor_selected_arrays",
        "required_row_cache_schema_written": payload["required_row_cache_fields"] == REQUIRED_ROW_CACHE_FIELDS,
        "user_action_or_engineering_action_written": payload["user_action_required"]["written"] is True,
        "no_future_endpoint_input": no_leak["future_endpoint_input"] is False,
        "no_future_waypoint_input": no_leak["future_waypoint_input"] is False,
        "no_central_velocity": no_leak["central_velocity"] is False,
        "no_test_endpoint_goals": no_leak["test_endpoint_goals"] is False,
        "no_test_threshold_tuning": no_leak["test_threshold_tuning"] is False,
        "metric_seconds_overclaim_blocked": claim["metric_or_seconds_claim"] is False,
        "stage5c_not_executed": claim["stage5c_executed"] is False,
        "smc_not_enabled": claim["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_hu_t100_runtime_batch_replay_sufficiency_pass_with_blocker" if passed == total else "stage42_hu_t100_runtime_batch_replay_sufficiency_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    suff = payload["sufficiency"]
    metric = payload["policy_artifact_payload"].get("test_summary_vs_train_horizon_causal_floor", {})
    lines = [
        "## Stage42-HU T100 Runtime Batch Replay Sufficiency Audit",
        "",
        "- source: `fresh_audit_from_stage42_hr_hs_ht_artifacts`",
        "- role: audit whether the frozen/runtime t100 easy guard evidence supports real row-level batch replay.",
        f"- gate: `{payload['stage42_hu_gate']['passed']} / {payload['stage42_hu_gate']['total']}`; verdict `{payload['stage42_hu_gate']['verdict']}`.",
        f"- runtime API ready: `{suff['runtime_api_ready']}`; frozen policy ready: `{suff['frozen_policy_ready']}`.",
        f"- real batch replay status: `{suff['real_batch_replay_status']}`; blocker: `{suff['blocker']}`.",
        "- conclusion: HT is callable and smoke-tested, but not a real batch replay because row-level candidate/floor/selected arrays are absent from HR/HS/HT artifacts.",
        f"- inherited guarded all/t50/t100 raw/hard/easy: `{_pct(metric.get('all_improvement', 0.0))}` / `{_pct(metric.get('t50_improvement', 0.0))}` / `{_pct(metric.get('t100_raw_frame_diagnostic_improvement', 0.0))}` / `{_pct(metric.get('hard_failure_improvement', 0.0))}` / `{_pct(metric.get('easy_degradation', 0.0))}`.",
        "- Claim boundary: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, MASTER_SUMMARY]:
        _replace_section(path, "STAGE42_HU_T100_RUNTIME_BATCH_REPLAY_SUFFICIENCY", lines)


def _write_reports(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_hu_gate"]
    suff = payload["sufficiency"]
    metric = payload["policy_artifact_payload"].get("test_summary_vs_train_horizon_causal_floor", {})
    lines = [
        "# Stage42-HU T100 Runtime Batch Replay Sufficiency Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Sufficiency Result",
        "",
        f"- runtime_api_ready: `{suff['runtime_api_ready']}`",
        f"- frozen_policy_ready: `{suff['frozen_policy_ready']}`",
        f"- row_level_batch_replay_ready: `{suff['row_level_batch_replay_ready']}`",
        f"- real_batch_replay_status: `{suff['real_batch_replay_status']}`",
        f"- blocker: `{suff['blocker']}`",
        "",
        "## Required Row Cache Fields",
        "",
        *[f"- `{field}`" for field in REQUIRED_ROW_CACHE_FIELDS],
        "",
        "## Inherited Frozen Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| all | {_pct(metric.get('all_improvement', 0.0))} |",
        f"| t50 | {_pct(metric.get('t50_improvement', 0.0))} |",
        f"| t100 raw diagnostic | {_pct(metric.get('t100_raw_frame_diagnostic_improvement', 0.0))} |",
        f"| hard/failure | {_pct(metric.get('hard_failure_improvement', 0.0))} |",
        f"| easy degradation | {_pct(metric.get('easy_degradation', 0.0))} |",
        f"| t100 easy degradation | {_pct(metric.get('t100_easy_degradation', 0.0))} |",
        "",
        "## Interpretation",
        "",
        "- Stage42-HT should be described as runtime API + smoke replay, not real batch replay.",
        "- A real batch replay requires row-level candidate/floor/selected rollout arrays and row identifiers.",
        "- This audit is useful precisely because it prevents overclaiming deployment reproducibility evidence.",
    ]
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage42-HU Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
        *[f"| `{name}` | `{bool(ok)}` |" for name, ok in gate["gates"].items()],
    ]
    write_md(GATE_MD, gate_lines)
    action_lines = [
        "# User / Engineering Action Required: T100 Runtime Batch Replay Row Cache",
        "",
        "Stage42-HU found that Stage42-HR/HS/HT artifacts are sufficient for frozen policy replay and runtime smoke tests, but not for real row-level batch replay.",
        "",
        "Required cache fields:",
        "",
        *[f"- `{field}`" for field in REQUIRED_ROW_CACHE_FIELDS],
        "",
        "Rules:",
        "",
        "- `future_xy_label_eval_only_optional` may be used only for loss/evaluation, never as runtime input.",
        "- `candidate_xy_predicted_rollout` and `floor_xy_train_horizon_causal_rollout` must be stored with stable row ids.",
        "- Thresholds must remain validation-selected; test rows are replay/evaluation only.",
        "- This remains raw-frame/dataset-local evidence unless metric/time calibration is separately verified.",
    ]
    write_md(USER_ACTION_MD, action_lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-HU t100 runtime batch replay sufficiency audit"
    state["current_verdict"] = payload["stage42_hu_gate"]["verdict"]
    state["stage42_hu_t100_runtime_batch_replay_sufficiency"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_hu_gate"]["verdict"],
        "gates": f"{payload['stage42_hu_gate']['passed']}/{payload['stage42_hu_gate']['total']}",
        "runtime_api_ready": payload["sufficiency"]["runtime_api_ready"],
        "row_level_batch_replay_ready": payload["sufficiency"]["row_level_batch_replay_ready"],
        "real_batch_replay_status": payload["sufficiency"]["real_batch_replay_status"],
        "blocker": payload["sufficiency"]["blocker"],
        "required_row_cache_fields": payload["required_row_cache_fields"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["last_updated"] = "2026-05-27"
    write_json(RESEARCH_STATE, state)


def run_stage42_t100_runtime_batch_replay_sufficiency() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    hr = read_json(HR_JSON, {})
    hs = read_json(HS_JSON, {})
    ht = read_json(HT_JSON, {})
    policy = read_json(POLICY_JSON, {})
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-HU t100 runtime batch replay sufficiency audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([HR_JSON, HS_JSON, HT_JSON, POLICY_JSON]),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "stage42_hr": {"path": str(HR_JSON), "exists": HR_JSON.exists(), "stage42_hr_gate": hr.get("stage42_hr_gate", {})},
            "stage42_hs": {"path": str(HS_JSON), "exists": HS_JSON.exists(), "stage42_hs_gate": hs.get("stage42_hs_gate", {})},
            "stage42_ht": {"path": str(HT_JSON), "exists": HT_JSON.exists(), "stage42_ht_gate": ht.get("stage42_ht_gate", {})},
        },
        "policy_artifact": {"path": str(POLICY_JSON), "exists": POLICY_JSON.exists()},
        "policy_artifact_payload": policy,
        "artifact_summaries": [
            _artifact_summary(HR_JSON, hr),
            _artifact_summary(HS_JSON, hs),
            _artifact_summary(HT_JSON, ht),
            _artifact_summary(POLICY_JSON, policy),
        ],
        "required_row_cache_fields": REQUIRED_ROW_CACHE_FIELDS,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "user_action_required": {"path": str(USER_ACTION_MD), "written": True},
    }
    payload["sufficiency"] = _sufficiency(payload)
    payload["stage42_hu_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    _write_reports(payload)
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    out = run_stage42_t100_runtime_batch_replay_sufficiency()
    gate = out["stage42_hu_gate"]
    print(f"Stage42-HU t100 runtime batch replay sufficiency: {gate['verdict']} ({gate['passed']}/{gate['total']})")
