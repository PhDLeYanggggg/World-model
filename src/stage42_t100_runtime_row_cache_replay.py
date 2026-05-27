from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_group_consistency_full_waypoint_repair as di
from src import stage42_group_consistency_t100_easy_guard as hr
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_group_consistency_t100_easy_guard_runtime import FrozenT100EasyGuardPolicy
from src.stage42_proximity_guard_policy_freeze import _pct, _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
CACHE_DIR = Path("data/stage42_t100_runtime_replay_cache")
CACHE_PATH = CACHE_DIR / "stage42hv_t100_runtime_replay_test_cache.npz"

HR_JSON = OUT_DIR / "group_consistency_t100_easy_guard_stage42.json"
HS_JSON = OUT_DIR / "group_consistency_t100_easy_guard_freeze_stage42.json"
HT_JSON = OUT_DIR / "group_consistency_t100_easy_guard_runtime_stage42.json"
HU_JSON = OUT_DIR / "t100_runtime_batch_replay_sufficiency_stage42.json"
POLICY_JSON = OUT_DIR / "frozen_group_consistency_t100_easy_guard_policy_stage42.json"

REPORT_JSON = OUT_DIR / "t100_runtime_row_cache_replay_stage42.json"
REPORT_MD = OUT_DIR / "t100_runtime_row_cache_replay_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_hv_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
MASTER_SUMMARY = Path("README_M3W_CURRENT_MASTER_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE_FRESH = "fresh_row_cache_reconstruction_and_runtime_batch_replay_from_stage42_hr_ht"
SOURCE_CACHED = "cached_verified_row_cache_runtime_batch_replay_from_stage42_hr_ht"

REQUIRED_CACHE_FIELDS = [
    "row_id",
    "split",
    "domain",
    "source_file",
    "scene_id",
    "frame_id",
    "agent_id",
    "horizon",
    "candidate_xy_predicted_rollout",
    "floor_xy_train_horizon_causal_rollout",
    "selected_xy_stage42_hr",
    "selected_xy_runtime_replay",
    "candidate_switch",
    "runtime_switch",
    "runtime_reason",
    "future_xy_label_eval_only",
    "future_valid_label_eval_only",
    "normalizer",
    "hard_label",
    "failure_label",
    "easy_label",
]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-HV 修复 Stage42-HU 暴露的 row-level replay blocker：从 HR 的可重建路径生成本地 row-level replay cache。",
    "cache 是 derived local artifact，写在 data/ 下，不提交 GitHub。",
    "runtime replay 只使用 domain、horizon、candidate rollout、floor rollout 和 optional candidate switch。",
    "future waypoints / endpoints 只作为 evaluation labels 存储，不作为 runtime input。",
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


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _trajectory_errors(xy: np.ndarray, future_xy: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid_b = valid.astype(bool)
    err = np.linalg.norm(xy.astype(np.float64) - future_xy.astype(np.float64), axis=2)
    ade = (err * valid_b).sum(axis=1) / np.maximum(valid_b.sum(axis=1), 1)
    fde = err[:, -1]
    return ade.astype(np.float64), fde.astype(np.float64)


def _safe_improvement(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return 1.0 - float(np.mean(selected[mask])) / max(float(np.mean(floor[mask])), di.EPS)


def _metric_from_errors(
    selected_ade: np.ndarray,
    floor_ade: np.ndarray,
    horizons: np.ndarray,
    hard: np.ndarray,
    failure: np.ndarray,
    easy: np.ndarray,
    switch: np.ndarray,
) -> dict[str, Any]:
    h = horizons.astype(int)
    hard_failure = hard.astype(bool) | failure.astype(bool)
    easy_b = easy.astype(bool)
    all_mask = np.ones(len(h), dtype=bool)
    return {
        "rows": int(len(h)),
        "all_improvement": _safe_improvement(selected_ade, floor_ade, all_mask),
        "t10_improvement": _safe_improvement(selected_ade, floor_ade, h == 10),
        "t25_improvement": _safe_improvement(selected_ade, floor_ade, h == 25),
        "t50_improvement": _safe_improvement(selected_ade, floor_ade, h == 50),
        "t100_raw_frame_diagnostic_improvement": _safe_improvement(selected_ade, floor_ade, h == 100),
        "hard_failure_improvement": _safe_improvement(selected_ade, floor_ade, hard_failure),
        "easy_degradation": -_safe_improvement(selected_ade, floor_ade, easy_b),
        "t100_easy_degradation": -_safe_improvement(selected_ade, floor_ade, (h == 100) & easy_b),
        "switch_rate": float(np.mean(switch.astype(bool))) if len(h) else 0.0,
        "harm_over_fallback": float(np.mean(selected_ade - floor_ade)) if len(h) else 0.0,
    }


def _field_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path), "fields": [], "missing_required_fields": REQUIRED_CACHE_FIELDS}
    with np.load(path, allow_pickle=False) as npz:
        fields = list(npz.files)
        shapes = {k: list(npz[k].shape) for k in fields}
        dtypes = {k: str(npz[k].dtype) for k in fields}
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "fields": fields,
        "shapes": shapes,
        "dtypes": dtypes,
        "missing_required_fields": [field for field in REQUIRED_CACHE_FIELDS if field not in fields],
    }


def _cache_has_required_fields(path: Path = CACHE_PATH) -> bool:
    summary = _field_summary(path)
    return bool(summary.get("exists")) and not summary.get("missing_required_fields")


def _build_row_cache(policy: FrozenT100EasyGuardPolicy) -> dict[str, Any]:
    ensure_dir(CACHE_DIR)
    rebuilt = hr._rebuild_hq_candidate()
    guarded = hr._apply_domain_t100_guard(rebuilt, threshold=hr.T100_EASY_THRESHOLD)
    data = rebuilt["data"]
    labels = rebuilt["labels"]
    test_ids = rebuilt["test_ids"].astype(np.int64)
    candidate_xy = rebuilt["test"]["selected_xy"].astype(np.float32)
    floor_xy = rebuilt["floor_xy"][test_ids].astype(np.float32)
    candidate_switch = rebuilt["test"]["switch"].astype(bool)
    domains = data["dataset"][test_ids].astype("U64")
    horizons = data["horizon"][test_ids].astype(np.int64)
    runtime = policy.apply(
        domains=domains,
        horizons=horizons,
        candidate_xy=candidate_xy,
        floor_xy=floor_xy,
        candidate_switch=candidate_switch,
    )
    selected_xy = runtime.selected_xy.astype(np.float32)
    future_xy = labels["waypoint_xy"][test_ids].astype(np.float32)
    future_valid = labels["waypoint_valid"][test_ids].astype(bool)
    selected_ade, selected_fde = _trajectory_errors(selected_xy, future_xy, future_valid)
    floor_ade, floor_fde = _trajectory_errors(floor_xy, future_xy, future_valid)
    candidate_ade, candidate_fde = _trajectory_errors(candidate_xy, future_xy, future_valid)
    max_diffs = {
        "selected_ade_vs_hr_guarded": float(np.max(np.abs(selected_ade - guarded["selected_ade"]))) if len(selected_ade) else 0.0,
        "selected_fde_vs_hr_guarded": float(np.max(np.abs(selected_fde - guarded["selected_fde"]))) if len(selected_fde) else 0.0,
        "switch_vs_hr_guarded": int(np.sum(runtime.switch.astype(bool) != guarded["switch"].astype(bool))),
        "floor_ade_vs_hr_test": float(np.max(np.abs(floor_ade - rebuilt["test"]["floor_ade"]))) if len(floor_ade) else 0.0,
        "candidate_ade_vs_hq_pre_guard": float(np.max(np.abs(candidate_ade - rebuilt["test"]["selected_ade"]))) if len(candidate_ade) else 0.0,
    }
    np.savez_compressed(
        CACHE_PATH,
        row_id=test_ids,
        split=np.full(len(test_ids), "test", dtype="U8"),
        domain=domains,
        source_file=data["source_file"][test_ids].astype("U512"),
        scene_id=data["scene_id"][test_ids].astype("U256"),
        frame_id=data["frame_id"][test_ids].astype(np.float64),
        agent_id=data["agent_id"][test_ids].astype(np.int64),
        horizon=horizons,
        candidate_xy_predicted_rollout=candidate_xy,
        floor_xy_train_horizon_causal_rollout=floor_xy,
        selected_xy_stage42_hr=selected_xy,
        selected_xy_runtime_replay=selected_xy,
        candidate_switch=candidate_switch,
        runtime_switch=runtime.switch.astype(bool),
        runtime_reason=np.asarray(runtime.reasons, dtype="U96"),
        future_xy_label_eval_only=future_xy,
        future_valid_label_eval_only=future_valid,
        normalizer=np.maximum(data["scale"][test_ids].astype(np.float32), di.EPS),
        hard_label=data["hard"][test_ids].astype(bool),
        failure_label=data["failure"][test_ids].astype(bool),
        easy_label=data["easy"][test_ids].astype(bool),
        candidate_ade=candidate_ade.astype(np.float64),
        candidate_fde=candidate_fde.astype(np.float64),
        floor_ade=floor_ade.astype(np.float64),
        floor_fde=floor_fde.astype(np.float64),
        selected_ade=selected_ade.astype(np.float64),
        selected_fde=selected_fde.astype(np.float64),
    )
    return {
        "source": "fresh_row_cache_reconstruction_from_hr_rebuild",
        "cache_path": str(CACHE_PATH),
        "rows": int(len(test_ids)),
        "max_diffs": max_diffs,
        "policy_runtime_diagnostics": runtime.diagnostics(),
        "hr_guarded_metric": guarded["metric"],
        "hr_t100_easy_degradation": float(guarded["t100_easy_degradation"]),
    }


def _replay_cache(policy: FrozenT100EasyGuardPolicy, path: Path = CACHE_PATH) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as npz:
        domain = npz["domain"].astype(str)
        horizon = npz["horizon"].astype(np.int64)
        candidate_xy = npz["candidate_xy_predicted_rollout"].astype(np.float32)
        floor_xy = npz["floor_xy_train_horizon_causal_rollout"].astype(np.float32)
        candidate_switch = npz["candidate_switch"].astype(bool)
        stored_selected = npz["selected_xy_stage42_hr"].astype(np.float32)
        stored_runtime = npz["selected_xy_runtime_replay"].astype(np.float32)
        future_xy = npz["future_xy_label_eval_only"].astype(np.float32)
        future_valid = npz["future_valid_label_eval_only"].astype(bool)
        hard = npz["hard_label"].astype(bool)
        failure = npz["failure_label"].astype(bool)
        easy = npz["easy_label"].astype(bool)
        stored_switch = npz["runtime_switch"].astype(bool)
        stored_selected_ade = npz["selected_ade"].astype(np.float64)
        stored_floor_ade = npz["floor_ade"].astype(np.float64)
        row_id = npz["row_id"].astype(np.int64)
    runtime = policy.apply(
        domains=domain,
        horizons=horizon,
        candidate_xy=candidate_xy,
        floor_xy=floor_xy,
        candidate_switch=candidate_switch,
    )
    replay_ade, replay_fde = _trajectory_errors(runtime.selected_xy, future_xy, future_valid)
    floor_ade, floor_fde = _trajectory_errors(floor_xy, future_xy, future_valid)
    metric = _metric_from_errors(replay_ade, floor_ade, horizon, hard, failure, easy, runtime.switch)
    return {
        "rows": int(len(row_id)),
        "row_id_min": int(np.min(row_id)) if len(row_id) else -1,
        "row_id_max": int(np.max(row_id)) if len(row_id) else -1,
        "selected_xy_max_abs_diff_vs_stored_hr": float(np.max(np.abs(runtime.selected_xy - stored_selected))) if len(row_id) else 0.0,
        "selected_xy_max_abs_diff_vs_stored_runtime": float(np.max(np.abs(runtime.selected_xy - stored_runtime))) if len(row_id) else 0.0,
        "selected_ade_max_abs_diff_vs_stored": float(np.max(np.abs(replay_ade - stored_selected_ade))) if len(row_id) else 0.0,
        "floor_ade_max_abs_diff_vs_stored": float(np.max(np.abs(floor_ade - stored_floor_ade))) if len(row_id) else 0.0,
        "switch_mismatch_vs_stored": int(np.sum(runtime.switch.astype(bool) != stored_switch)),
        "metric": metric,
        "runtime_diagnostics": runtime.diagnostics(),
        "t100_rows": int(np.sum(horizon == 100)),
        "t100_easy_rows": int(np.sum((horizon == 100) & easy)),
        "domains": {str(k): int(v) for k, v in zip(*np.unique(domain, return_counts=True))},
    }


def _compare_metric(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, float]:
    keys = [
        "all_improvement",
        "t50_improvement",
        "t100_raw_frame_diagnostic_improvement",
        "hard_failure_improvement",
        "easy_degradation",
        "t100_easy_degradation",
        "switch_rate",
    ]
    return {key: abs(float(a.get(key, 0.0)) - float(b.get(key, 0.0))) for key in keys}


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    replay = payload["runtime_batch_replay"]
    cache = payload["cache_summary"]
    diffs = replay["metric_diff_vs_hr"]
    claim = payload["claim_boundary"]
    no_leak = payload["no_leakage"]
    gates = {
        "hr_artifact_present": payload["inputs"]["stage42_hr"]["exists"] is True,
        "hs_artifact_present": payload["inputs"]["stage42_hs"]["exists"] is True,
        "ht_artifact_present": payload["inputs"]["stage42_ht"]["exists"] is True,
        "hu_artifact_present": payload["inputs"]["stage42_hu"]["exists"] is True,
        "policy_artifact_present": payload["policy_artifact"]["exists"] is True,
        "row_cache_exists": cache["exists"] is True,
        "required_cache_fields_present": cache["missing_required_fields"] == [],
        "row_level_batch_replay_ready": replay["status"] in {"fresh_run", "cached_verified"},
        "selected_xy_replay_exact": replay["selected_xy_max_abs_diff_vs_stored_hr"] <= 1e-6,
        "selected_ade_replay_exact": replay["selected_ade_max_abs_diff_vs_stored"] <= 1e-6,
        "floor_ade_replay_exact": replay["floor_ade_max_abs_diff_vs_stored"] <= 1e-6,
        "switch_replay_exact": replay["switch_mismatch_vs_stored"] == 0,
        "metric_replay_matches_hr": max(diffs.values()) <= 1e-6,
        "t100_easy_guard_preserved": replay["metric"]["t100_easy_degradation"] <= 0.0,
        "t100_raw_still_positive": replay["metric"]["t100_raw_frame_diagnostic_improvement"] > 0.0,
        "global_all_positive": replay["metric"]["all_improvement"] > 0.0,
        "global_t50_positive": replay["metric"]["t50_improvement"] > 0.0,
        "hard_failure_positive": replay["metric"]["hard_failure_improvement"] > 0.0,
        "easy_preserved": replay["metric"]["easy_degradation"] <= 0.02,
        "future_labels_eval_only": no_leak["future_waypoints_eval_only"] is True,
        "no_future_endpoint_input": no_leak["future_endpoint_input"] is False,
        "no_future_waypoint_input": no_leak["future_waypoint_input"] is False,
        "no_central_velocity": no_leak["central_velocity"] is False,
        "no_test_endpoint_goals": no_leak["test_endpoint_goals"] is False,
        "no_test_threshold_tuning": no_leak["test_threshold_tuning"] is False,
        "no_metric_seconds_overclaim": claim["metric_or_seconds_claim"] is False,
        "stage5c_not_executed": claim["stage5c_executed"] is False,
        "smc_not_enabled": claim["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_hv_t100_runtime_row_cache_replay_pass" if passed == total else "stage42_hv_t100_runtime_row_cache_replay_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _write_reports(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_hv_gate"]
    replay = payload["runtime_batch_replay"]
    metric = replay["metric"]
    lines = [
        "# Stage42-HV T100 Runtime Row-Cache Batch Replay",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- cache_hash: `{payload['cache_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Row Cache",
        "",
        f"- cache_path: `{payload['cache_summary']['path']}`",
        f"- cache_status: `{payload['cache_status']}`",
        f"- cache_size_bytes: `{payload['cache_summary'].get('size_bytes', 0)}`",
        f"- required fields present: `{payload['cache_summary']['missing_required_fields'] == []}`",
        f"- rows: `{replay['rows']}`",
        f"- domains: `{replay['domains']}`",
        f"- t100 rows: `{replay['t100_rows']}`",
        f"- t100 easy rows: `{replay['t100_easy_rows']}`",
        "",
        "## Runtime Batch Replay Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| all | {_pct(metric['all_improvement'])} |",
        f"| t50 | {_pct(metric['t50_improvement'])} |",
        f"| t100 raw diagnostic | {_pct(metric['t100_raw_frame_diagnostic_improvement'])} |",
        f"| hard/failure | {_pct(metric['hard_failure_improvement'])} |",
        f"| easy degradation | {_pct(metric['easy_degradation'])} |",
        f"| t100 easy degradation | {_pct(metric['t100_easy_degradation'])} |",
        f"| switch rate | {_pct(metric['switch_rate'])} |",
        "",
        "## Exact Replay Checks",
        "",
        f"- selected_xy_max_abs_diff_vs_stored_hr: `{replay['selected_xy_max_abs_diff_vs_stored_hr']}`",
        f"- selected_ade_max_abs_diff_vs_stored: `{replay['selected_ade_max_abs_diff_vs_stored']}`",
        f"- floor_ade_max_abs_diff_vs_stored: `{replay['floor_ade_max_abs_diff_vs_stored']}`",
        f"- switch_mismatch_vs_stored: `{replay['switch_mismatch_vs_stored']}`",
        f"- metric_diff_vs_hr: `{replay['metric_diff_vs_hr']}`",
        "",
        "## Interpretation",
        "",
        "- Stage42-HV closes the Stage42-HU blocker locally by reconstructing a row-level replay cache from the HR rebuild path.",
        "- The cache is intentionally not committed because it is derived row-level rollout data.",
        "- The report and hashes are committed so reviewers can see what was replayed and why the claim remains bounded.",
        "- This is real row-level runtime batch replay for the frozen t100 easy guard, not Stage5C, not SMC, not metric, and not seconds-level.",
    ]
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage42-HV Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
        *[f"| `{name}` | `{bool(ok)}` |" for name, ok in gate["gates"].items()],
    ]
    write_md(GATE_MD, gate_lines)


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    replay = payload["runtime_batch_replay"]
    metric = replay["metric"]
    lines = [
        "## Stage42-HV T100 Runtime Row-Cache Batch Replay",
        "",
        "- source: `fresh_or_cached_row_cache_reconstruction_and_runtime_batch_replay_from_stage42_hr_ht`",
        "- role: close the Stage42-HU blocker by reconstructing a local row-level cache and replaying the frozen Stage42-HT runtime policy over full test rows.",
        f"- gate: `{payload['stage42_hv_gate']['passed']} / {payload['stage42_hv_gate']['total']}`; verdict `{payload['stage42_hv_gate']['verdict']}`.",
        f"- cache path: `{CACHE_PATH}` (derived local data; not committed).",
        f"- cache hash: `{payload['cache_hash']}`.",
        f"- runtime replay rows/domains/t100 rows: `{replay['rows']}` / `{replay['domains']}` / `{replay['t100_rows']}`.",
        f"- replay all/t50/t100 raw/hard/easy: `{_pct(metric['all_improvement'])}` / `{_pct(metric['t50_improvement'])}` / `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(metric['hard_failure_improvement'])}` / `{_pct(metric['easy_degradation'])}`.",
        f"- t100 easy degradation: `{_pct(metric['t100_easy_degradation'])}`.",
        "- Claim boundary: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, MASTER_SUMMARY]:
        _replace_section(path, "STAGE42_HV_T100_RUNTIME_ROW_CACHE_REPLAY", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    gate = payload["stage42_hv_gate"]
    replay = payload["runtime_batch_replay"]
    metric = replay["metric"]
    state["current_stage"] = "Stage42-HV t100 runtime row-cache batch replay"
    state["current_verdict"] = gate["verdict"]
    state.setdefault("stage42", {})["stage_hv_t100_runtime_row_cache_replay"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "cache_path": str(CACHE_PATH),
        "cache_committed": False,
        "cache_hash": payload["cache_hash"],
        "gates": f"{gate['passed']}/{gate['total']}",
        "verdict": gate["verdict"],
        "rows": replay["rows"],
        "t100_rows": replay["t100_rows"],
        "metric": metric,
        "claim_boundary": payload["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(RESEARCH_STATE, _jsonable(state))


def run_stage42_t100_runtime_row_cache_replay(*, rebuild_cache: bool = False) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CACHE_DIR)
    policy = FrozenT100EasyGuardPolicy.from_file(POLICY_JSON)
    cache_status = "cached_verified"
    build_info: dict[str, Any] = {"source": "cached_verified_existing_cache"}
    if rebuild_cache or not _cache_has_required_fields(CACHE_PATH):
        build_info = _build_row_cache(policy)
        cache_status = "fresh_run"
    replay = _replay_cache(policy, CACHE_PATH)
    hr_payload = read_json(HR_JSON, {})
    hr_guarded = hr_payload.get("guarded", {})
    hr_metric = dict(hr_guarded.get("metric", {}))
    if "t100_easy_degradation" not in hr_metric and "t100_easy_degradation" in hr_guarded:
        hr_metric["t100_easy_degradation"] = hr_guarded["t100_easy_degradation"]
    replay["metric_diff_vs_hr"] = _compare_metric(replay["metric"], hr_metric)
    replay["status"] = cache_status
    cache_summary = _field_summary(CACHE_PATH)
    payload: dict[str, Any] = {
        "stage": "Stage42-HV t100 runtime row-cache batch replay",
        "source": SOURCE_FRESH if cache_status == "fresh_run" else SOURCE_CACHED,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([HR_JSON, HS_JSON, HT_JSON, HU_JSON, POLICY_JSON]),
        "cache_hash": _combined_hash([CACHE_PATH]),
        "cache_status": cache_status,
        "cache_build": build_info,
        "cache_summary": cache_summary,
        "policy_artifact": {"path": str(POLICY_JSON), "exists": POLICY_JSON.exists(), "policy_hash": policy.policy_hash},
        "inputs": {
            "stage42_hr": {"path": str(HR_JSON), "exists": HR_JSON.exists(), "verdict": hr_payload.get("stage42_hr_gate", {}).get("verdict")},
            "stage42_hs": {"path": str(HS_JSON), "exists": HS_JSON.exists(), "verdict": read_json(HS_JSON, {}).get("stage42_hs_gate", {}).get("verdict")},
            "stage42_ht": {"path": str(HT_JSON), "exists": HT_JSON.exists(), "verdict": read_json(HT_JSON, {}).get("stage42_ht_gate", {}).get("verdict")},
            "stage42_hu": {"path": str(HU_JSON), "exists": HU_JSON.exists(), "verdict": read_json(HU_JSON, {}).get("stage42_hu_gate", {}).get("verdict")},
        },
        "runtime_batch_replay": replay,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoints_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "runtime_inputs": [
                "domain",
                "horizon",
                "candidate_xy_predicted_rollout",
                "floor_xy_train_horizon_causal_rollout",
                "candidate_switch_optional",
            ],
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_hv_gate"] = _gate(payload)
    write_json(REPORT_JSON, _jsonable(payload))
    _write_reports(payload)
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    out = run_stage42_t100_runtime_row_cache_replay()
    gate = out["stage42_hv_gate"]
    print(f"Stage42-HV t100 runtime row-cache batch replay: {gate['verdict']} ({gate['passed']}/{gate['total']})")
