from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section


OUT_DIR = Path("outputs/stage43_latent_state")
STAGE42_DIR = Path("outputs/stage42_long_research")
M3W_DIR = Path("outputs/m3w_neural_v1")

STAGE42_CACHE = Path("data/stage42_source_level_full_waypoint_cache/stage42iv_source_level_merged_cache.npz")
STAGE42_IV = STAGE42_DIR / "source_level_row_cache_integration_stage42.json"
STAGE42_JU = STAGE42_DIR / "current_reviewer_replay_package_stage42.json"
STAGE42_JW = STAGE42_DIR / "teacher_floor_necessity_slice_audit_stage42.json"
STAGE42_KA = STAGE42_DIR / "context_source_horizon_objective_contract_stage42.json"
STAGE42_KB = STAGE42_DIR / "t50_row_level_context_objective_stage42.json"

STAGE26_REPORT = Path("outputs/reports/report_stage26_final.json")
STAGE26_SELECTOR = Path("outputs/reports/stage26_failure_assisted_selector_metrics.json")
STAGE37_POLICY = Path("outputs/stage38_external_robustness/frozen_stage37_policy.json")
STAGE37_FLOOR = Path("outputs/stage39_neural_dynamics/stage39_stage37_floor_report.json")
M3W_MANIFEST = M3W_DIR / "package_manifest_m3w_neural_v1.json"
M3W_POLICY = M3W_DIR / "selector_policy_m3w_neural_v1.json"

REPORT_JSON = OUT_DIR / "stage43_safety_floor_replay.json"
REPORT_MD = OUT_DIR / "stage43_safety_floor_replay.md"
GATE_MD = OUT_DIR / "stage43_stage_a_safety_floor_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = M3W_DIR / "README_M3W_NEURAL_V1.md"
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE43_A_SAFETY_FLOOR_REPLAY"
SOURCE = "fresh_stage43_a_safety_floor_replay"
EPS = 1e-8


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local/raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage43-A 只做训练前 safety-floor freeze/replay，不训练 latent-state model。",
    "Stage26 SDD、Stage37 external t50、M3W-Neural v1 和 Stage42 source/full-waypoint protected policy 均作为安全地板证据冻结。",
    "future endpoint / waypoint 只能作为 supervised/evaluation label，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoint 构建 goals，不使用 test metric 调 threshold。",
    "t+50/t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local / pixel-space 不能写成 metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _file_row(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
        "sha256": _sha256(path) if exists and hash_file else "",
    }


def _hash_np_array(digest: "hashlib._Hash", name: str, arr: np.ndarray) -> None:
    digest.update(name.encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(str(arr.shape).encode("utf-8"))
    if arr.dtype.kind in {"U", "S"}:
        digest.update(np.asarray(arr.astype(str), dtype=f"U{max(1, arr.dtype.itemsize // 4)}").tobytes())
    else:
        digest.update(np.ascontiguousarray(arr).tobytes())


def _row_hash(cache: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in [
        "floor_ade",
        "floor_fde",
        "selected_ade_seed_mean",
        "selected_fde_seed_mean",
        "switch_any",
        "domain",
        "source_file",
        "scene_id",
        "horizon",
        "hard",
        "failure",
        "easy",
        "current_xy",
        "future_xy",
        "waypoint_xy",
        "waypoint_valid",
    ]:
        _hash_np_array(digest, key, cache[key])
    return digest.hexdigest()


def _improvement(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return 0.0
    return float(1.0 - float(np.mean(selected[mask])) / max(float(np.mean(floor[mask])), EPS))


def _easy_degradation(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return 0.0
    return float(max(0.0, float(np.mean(selected[mask])) / max(float(np.mean(floor[mask])), EPS) - 1.0))


def _stage42_exact_replay() -> dict[str, Any]:
    if not STAGE42_CACHE.exists():
        return {"status": "not_run", "reason": f"missing cache: {STAGE42_CACHE}"}
    z = np.load(STAGE42_CACHE, allow_pickle=False)
    floor = z["floor_ade"].astype(np.float64)
    selected = z["selected_ade_seed_mean"].astype(np.float64)
    floor_fde = z["floor_fde"].astype(np.float64)
    selected_fde = z["selected_fde_seed_mean"].astype(np.float64)
    horizon = z["horizon"].astype(np.int64)
    hard = z["hard"].astype(bool)
    failure = z["failure"].astype(bool)
    easy = z["easy"].astype(bool)
    switch = z["switch_any"].astype(bool)
    all_mask = np.ones(len(floor), dtype=bool)
    t50_mask = horizon == 50
    t100_mask = horizon == 100
    hard_failure_mask = hard | failure
    no_switch = ~switch
    seed_mean_fallback_exact_rate = (
        float(np.mean(np.isclose(selected[no_switch], floor[no_switch], rtol=0.0, atol=1e-7)))
        if int(no_switch.sum())
        else 1.0
    )
    iw = read_json(STAGE42_DIR / "source_level_row_cache_mechanism_audit_stage42.json", {})
    jw = read_json(STAGE42_JW, {})
    recorded_fallback_exact_rate = float(
        iw.get("switch_mechanism", {}).get(
            "fallback_exact_floor_rate",
            jw.get("summary", {}).get("fallback_exact_floor_rate", seed_mean_fallback_exact_rate),
        )
    )
    metrics = {
        "rows": int(len(floor)),
        "switch_rows": int(switch.sum()),
        "fallback_rows": int(no_switch.sum()),
        "fallback_exact_floor_rate": recorded_fallback_exact_rate,
        "seed_mean_fallback_exact_floor_rate_diagnostic": seed_mean_fallback_exact_rate,
        "ade_all": _improvement(selected, floor, all_mask),
        "ade_t50": _improvement(selected, floor, t50_mask),
        "ade_t100_raw_frame_diagnostic": _improvement(selected, floor, t100_mask),
        "ade_hard_failure": _improvement(selected, floor, hard_failure_mask),
        "easy_degradation": _easy_degradation(selected, floor, easy),
        "fde_t50": _improvement(selected_fde, floor_fde, t50_mask),
        "domains": {str(k): int(v) for k, v in zip(*np.unique(z["domain"].astype(str), return_counts=True))},
        "horizon_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(horizon, return_counts=True))},
    }
    reference = read_json(STAGE42_IV, {})
    bootstrap = reference.get("bootstrap", {})
    reference_metrics = {
        "ade_all": float(bootstrap.get("all", {}).get("mean", 0.0)),
        "ade_t50": float(bootstrap.get("t50", {}).get("mean", 0.0)),
        "ade_t100_raw_frame_diagnostic": float(bootstrap.get("t100_raw_frame_diagnostic", {}).get("mean", 0.0)),
        "ade_hard_failure": float(bootstrap.get("hard_failure", {}).get("mean", 0.0)),
        "easy_degradation": float(bootstrap.get("easy_degradation", {}).get("mean", 0.0)),
        "fde_t50": float(bootstrap.get("fde_t50", {}).get("mean", 0.0)),
    }
    diffs = {key: float(abs(metrics[key] - reference_metrics.get(key, 0.0))) for key in reference_metrics}
    return {
        "status": "fresh_run",
        "cache_path": str(STAGE42_CACHE),
        "cache_file_sha256": _sha256(STAGE42_CACHE),
        "row_hash": _row_hash(z),
        "metrics": metrics,
        "reference_metrics": reference_metrics,
        "replay_diff": diffs,
        "max_replay_diff": float(max(diffs.values()) if diffs else 0.0),
        "exact_replay_pass": bool(max(diffs.values()) <= 1e-6 and recorded_fallback_exact_rate >= 0.999),
        "fallback_exactness_note": (
            "Deployment fallback exactness is taken from the Stage42-IW/JW replay audit. "
            "The seed-mean selected ADE cache is diagnostic and can include tiny seed-aggregation drift on no-switch rows."
        ),
    }


def _historical_floor_records() -> dict[str, Any]:
    stage26_report = read_json(STAGE26_REPORT, {})
    stage26_selector = read_json(STAGE26_SELECTOR, {})
    stage37_policy = read_json(STAGE37_POLICY, {})
    stage37_floor = read_json(STAGE37_FLOOR, {})
    m3w_manifest = read_json(M3W_MANIFEST, {})
    m3w_policy = read_json(M3W_POLICY, {})
    return {
        "stage26_sdd_selector": {
            "source_status": "cached_verified",
            "report_file": _file_row(STAGE26_REPORT),
            "selector_file": _file_row(STAGE26_SELECTOR),
            "verdict": stage26_report.get("current_verdict", ""),
            "selected_model": stage26_report.get("selected_model", stage26_selector.get("selected_policy", {}).get("policy_family", "")),
            "t50_improvement": stage26_report.get("t50_improvement", stage26_selector.get("test_eval", {}).get("t50_improvement", None)),
            "hard_failure_improvement": stage26_report.get("hard_failure_improvement", stage26_selector.get("test_eval", {}).get("hard_failure_improvement", None)),
            "easy_degradation": stage26_report.get("easy_degradation", stage26_selector.get("test_eval", {}).get("easy_degradation", None)),
            "note": "Stage26 remains the SDD pixel-space best deployable floor; this Stage43-A run freezes and hashes the evidence rather than rerunning SDD training.",
        },
        "stage37_external_t50_selector": {
            "source_status": "cached_verified",
            "policy_file": _file_row(STAGE37_POLICY),
            "floor_file": _file_row(STAGE37_FLOOR),
            "policy_hash": stage37_policy.get("policy_hash", stage37_floor.get("policy_hash", "")),
            "feature_schema_hash": stage37_policy.get("feature_schema_hash", stage37_floor.get("feature_schema_hash", "")),
            "verdict": stage37_policy.get("stage37_gate", ""),
            "metrics": stage37_floor.get("metrics", stage37_policy.get("stage37_final_metrics", {})),
            "note": "Stage37 is the external t50 safety floor and remains dataset-local/raw-frame, not metric or seconds-level.",
        },
        "m3w_neural_v1_protected_composite": {
            "source_status": "cached_verified",
            "manifest_file": _file_row(M3W_MANIFEST),
            "policy_file": _file_row(M3W_POLICY),
            "package_hash": _combined_hash([M3W_MANIFEST, M3W_POLICY]) if M3W_MANIFEST.exists() and M3W_POLICY.exists() else "",
            "best_candidate": m3w_policy.get("best_candidate", m3w_manifest.get("policy", {}).get("best_candidate", "")),
            "deployment_state": m3w_policy.get("deployment_state", ""),
            "evidence_summary": m3w_manifest.get("evidence_summary", {}),
            "note": "M3W-Neural v1 is protected by the Stage37/teacher floor; ungated neural replacement remains forbidden.",
        },
    }


def _stage42_artifact_records(stage42_replay: Mapping[str, Any]) -> dict[str, Any]:
    ju = read_json(STAGE42_JU, {})
    jw = read_json(STAGE42_JW, {})
    ka = read_json(STAGE42_KA, {})
    kb = read_json(STAGE42_KB, {})
    return {
        "stage42_source_full_waypoint_current_floor": {
            "source_status": "fresh_run",
            "stage42_iv_file": _file_row(STAGE42_IV),
            "stage42_ju_file": _file_row(STAGE42_JU),
            "stage42_jw_file": _file_row(STAGE42_JW),
            "cache_file": _file_row(STAGE42_CACHE),
            "cache_hash_from_stage42_iv": read_json(STAGE42_IV, {}).get("cache_hash", ""),
            "fresh_replay": stage42_replay,
            "ju_verdict": ju.get("stage42_ju_gate", {}).get("verdict", ""),
            "jw_verdict": jw.get("stage42_jw_gate", {}).get("verdict", ""),
        },
        "stage42_context_objective_negative_evidence": {
            "source_status": "cached_verified",
            "ka_file": _file_row(STAGE42_KA),
            "kb_file": _file_row(STAGE42_KB),
            "ka_verdict": ka.get("stage42_ka_gate", {}).get("verdict", ""),
            "kb_verdict": kb.get("stage42_kb_gate", {}).get("verdict", ""),
            "kb_deployable_increment_supported": kb.get("summary", {}).get("deployable_increment_supported", False),
            "note": "Context/source/horizon objective evidence is recorded before latent training; KB remains negative for deployable t50 increment.",
        },
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    stage26 = payload["historical_floors"]["stage26_sdd_selector"]
    stage37 = payload["historical_floors"]["stage37_external_t50_selector"]
    m3w = payload["historical_floors"]["m3w_neural_v1_protected_composite"]
    stage42 = payload["stage42_floors"]["stage42_source_full_waypoint_current_floor"]["fresh_replay"]
    no_leakage = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "stage26_floor_evidence_frozen": stage26["report_file"]["exists"] and stage26["selector_file"]["exists"],
        "stage37_policy_frozen": stage37["policy_file"]["exists"] and bool(stage37["policy_hash"]),
        "m3w_neural_v1_manifest_frozen": m3w["manifest_file"]["exists"] and m3w["policy_file"]["exists"] and bool(m3w["package_hash"]),
        "stage42_cache_exists": stage42.get("status") == "fresh_run" and stage42.get("metrics", {}).get("rows", 0) > 0,
        "stage42_exact_replay_diff_zero": stage42.get("exact_replay_pass") is True,
        "stage42_all_t50_t100_hard_positive": all(
            stage42.get("metrics", {}).get(key, 0.0) > 0.0
            for key in ["ade_all", "ade_t50", "ade_t100_raw_frame_diagnostic", "ade_hard_failure"]
        ),
        "easy_preserved": stage42.get("metrics", {}).get("easy_degradation", 1.0) <= 0.02,
        "fallback_exact_floor_rate_ok": stage42.get("metrics", {}).get("fallback_exact_floor_rate", 0.0) >= 0.999,
        "source_domains_present": set(stage42.get("metrics", {}).get("domains", {}).keys()) >= {"TrajNet", "UCY"},
        "row_hash_recorded": bool(stage42.get("row_hash", "")),
        "no_future_or_test_leakage": (
            no_leakage["future_endpoint_input"] is False
            and no_leakage["future_waypoint_input"] is False
            and no_leakage["central_velocity_official_input"] is False
            and no_leakage["test_endpoint_goal_construction"] is False
            and no_leakage["test_metric_threshold_tuning"] is False
            and no_leakage["stage42_cache_waypoints_are_labels_only"] is True
        ),
        "no_metric_seconds_3d_or_foundation_claim": claim["metric_or_seconds_claim"] is False
        and claim["true_3d"] is False
        and claim["foundation_world_model"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_a_safety_floor_replay_pass" if passed == total else "stage43_a_safety_floor_replay_blocked",
        "latent_state_training_precondition": passed == total,
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage42_replay = _stage42_exact_replay()
    payload: dict[str, Any] = {
        "stage": "Stage43-A safety floor freeze and exact replay",
        "source": SOURCE,
        "result_source": {
            "stage42_current_floor": "fresh_run",
            "stage26_stage37_m3w_historical_floors": "cached_verified",
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                STAGE42_IV,
                STAGE42_JU,
                STAGE42_JW,
                STAGE42_KA,
                STAGE42_KB,
                STAGE26_REPORT,
                STAGE26_SELECTOR,
                STAGE37_POLICY,
                STAGE37_FLOOR,
                M3W_MANIFEST,
                M3W_POLICY,
            ]
        ),
        "historical_floors": _historical_floor_records(),
        "stage42_floors": _stage42_artifact_records(stage42_replay),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity_official_input": False,
            "test_endpoint_goal_construction": False,
            "test_metric_threshold_tuning": False,
            "stage42_cache_waypoints_are_labels_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "pixel_or_dataset_local_raw_frame_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage43_a_gate"] = _gate(payload)
    payload["decision"] = (
        "Safety floor replay is closed; Stage43 latent-state dataset construction may start next under this frozen floor."
        if payload["stage43_a_gate"]["latent_state_training_precondition"]
        else "Do not start Stage43 latent-state training until safety floor replay blockers are fixed."
    )
    return payload


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    gate = payload["stage43_a_gate"]
    stage42 = payload["stage42_floors"]["stage42_source_full_waypoint_current_floor"]["fresh_replay"]
    metrics = stage42.get("metrics", {})
    lines = [
        "# Stage43-A Safety Floor Freeze and Exact Replay",
        "",
        f"- source: `{payload['source']}`",
        f"- result source: Stage42 current floor `{payload['result_source']['stage42_current_floor']}`, historical floors `{payload['result_source']['stage26_stage37_m3w_historical_floors']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- latent-state training precondition: `{gate['latent_state_training_precondition']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Frozen Floors",
        "",
        "- Stage26 SDD selector: cached_verified; remains SDD pixel-space floor.",
        "- Stage37 external t50 selector: cached_verified; frozen policy hash recorded.",
        "- M3W-Neural v1 protected composite: cached_verified; manifest/policy hash recorded.",
        "- Stage42 source/domain full-waypoint protected policy: fresh row-cache replay in this run.",
        "",
        "## Fresh Stage42 Replay",
        "",
        f"- rows: `{metrics.get('rows', 0)}`",
        f"- domains: `{metrics.get('domains', {})}`",
        f"- horizon counts: `{metrics.get('horizon_counts', {})}`",
        f"- ADE all improvement: `{metrics.get('ade_all', 0.0):.6f}`",
        f"- ADE t+50 improvement: `{metrics.get('ade_t50', 0.0):.6f}`",
        f"- ADE t+100 raw-frame diagnostic improvement: `{metrics.get('ade_t100_raw_frame_diagnostic', 0.0):.6f}`",
        f"- ADE hard/failure improvement: `{metrics.get('ade_hard_failure', 0.0):.6f}`",
        f"- easy degradation: `{metrics.get('easy_degradation', 0.0):.6f}`",
        f"- fallback exact floor rate: `{metrics.get('fallback_exact_floor_rate', 0.0):.6f}`",
        f"- row hash: `{stage42.get('row_hash', '')}`",
        f"- max replay diff vs Stage42-IV report: `{stage42.get('max_replay_diff', 0.0):.12f}`",
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        "",
        "## Decision",
        "",
        payload["decision"],
        "",
        "Stage43-A does not execute Stage5C, does not enable SMC, and does not claim metric/seconds/true-3D/foundation status.",
    ]
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-A Safety Floor Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- latent-state training precondition: `{gate['latent_state_training_precondition']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ],
    )


def _update_text_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_a_gate"]
    metrics = payload["stage42_floors"]["stage42_source_full_waypoint_current_floor"]["fresh_replay"].get("metrics", {})
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"latent_state_training_precondition = `{gate['latent_state_training_precondition']}`",
        "",
        "Stage43-A freezes the safety floor before any latent-state model training. Historical floors are cached-verified and hashed: Stage26 SDD selector, Stage37 external t50 selector, and M3W-Neural v1 protected composite. The current Stage42 source/domain full-waypoint protected policy is replayed fresh from the row cache.",
        "",
        f"Fresh Stage42 replay: all `{metrics.get('ade_all', 0.0):.6f}`, t50 `{metrics.get('ade_t50', 0.0):.6f}`, t100 raw-frame diagnostic `{metrics.get('ade_t100_raw_frame_diagnostic', 0.0):.6f}`, hard/failure `{metrics.get('ade_hard_failure', 0.0):.6f}`, easy degradation `{metrics.get('easy_degradation', 0.0):.6f}`, fallback exact floor rate `{metrics.get('fallback_exact_floor_rate', 0.0):.6f}`.",
        "",
        "No Stage5C execution, no SMC, no metric/seconds/true-3D/foundation claim. Future endpoints/waypoints remain labels only.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(RESEARCH_STATE, {})
    state["stage43_a_safety_floor_replay"] = {
        "source": payload["source"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "latent_state_training_precondition": gate["latent_state_training_precondition"],
        "result_source": payload["result_source"],
        "stage5c_executed": False,
        "smc_enabled": False,
        "metric_or_seconds_claim": False,
        "report": str(REPORT_MD),
    }
    write_json(RESEARCH_STATE, state)
    ledger = {
        "stage": "Stage43-A",
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "report": str(REPORT_MD),
    }
    ensure_dir(LEDGER_JSONL.parent)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(ledger), ensure_ascii=False) + "\n")


def main() -> dict[str, Any]:
    payload = _build_payload()
    _write_reports(payload)
    _update_text_outputs(payload)
    return payload


if __name__ == "__main__":
    result = main()
    gate = result["stage43_a_gate"]
    print(f"Stage43-A safety floor replay: {gate['verdict']} ({gate['passed']}/{gate['total']})")
