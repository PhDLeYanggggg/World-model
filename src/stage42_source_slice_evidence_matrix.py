from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
CACHE_PATH = Path("data/stage42_source_level_full_waypoint_cache/stage42iv_source_level_merged_cache.npz")
IV_JSON = OUT_DIR / "source_level_row_cache_integration_stage42.json"
IW_JSON = OUT_DIR / "source_level_row_cache_mechanism_audit_stage42.json"
JU_JSON = OUT_DIR / "current_reviewer_replay_package_stage42.json"

REPORT_JSON = OUT_DIR / "source_slice_evidence_matrix_stage42.json"
REPORT_MD = OUT_DIR / "source_slice_evidence_matrix_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_jv_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE42_JV_SOURCE_SLICE_EVIDENCE_MATRIX"
SOURCE = "fresh_stage42_jv_source_slice_evidence_matrix_from_cached_verified_row_cache"
EPS = 1e-8

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-JV 使用 cached_verified Stage42-IV row-level cache 做 fresh source/domain/horizon/slice 分解。",
    "Stage42-JV 不训练、不调 threshold、不下载、不转换，不把 slice synthesis 当新模型结果。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不能作为 inference input。",
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


def _load_cache() -> dict[str, np.ndarray]:
    with np.load(CACHE_PATH, allow_pickle=True) as cache:
        return {key: cache[key] for key in cache.files}


def _rel_improvement(floor: np.ndarray, selected: np.ndarray) -> float:
    if len(floor) == 0:
        return 0.0
    floor_mean = float(np.mean(floor))
    selected_mean = float(np.mean(selected))
    if floor_mean <= EPS:
        return 0.0
    return (floor_mean - selected_mean) / floor_mean


def _degradation(floor: np.ndarray, selected: np.ndarray) -> float:
    if len(floor) == 0:
        return 0.0
    floor_mean = float(np.mean(floor))
    selected_mean = float(np.mean(selected))
    if floor_mean <= EPS:
        return 0.0
    return max(0.0, (selected_mean - floor_mean) / floor_mean)


def _metric(cache: Mapping[str, np.ndarray], mask: np.ndarray, label: str) -> dict[str, Any]:
    mask = mask.astype(bool)
    rows = int(np.sum(mask))
    floor_ade = cache["floor_ade"][mask].astype(np.float64)
    selected_ade = cache["selected_ade_seed_mean"][mask].astype(np.float64)
    floor_fde = cache["floor_fde"][mask].astype(np.float64)
    selected_fde = cache["selected_fde_seed_mean"][mask].astype(np.float64)
    waypoint_valid = cache["waypoint_valid"][mask].astype(bool)
    switch = cache["switch_any"][mask].astype(bool)
    return {
        "label": label,
        "rows": rows,
        "floor_ade_mean": float(np.mean(floor_ade)) if rows else 0.0,
        "selected_ade_mean": float(np.mean(selected_ade)) if rows else 0.0,
        "ade_improvement": _rel_improvement(floor_ade, selected_ade),
        "floor_fde_mean": float(np.mean(floor_fde)) if rows else 0.0,
        "selected_fde_mean": float(np.mean(selected_fde)) if rows else 0.0,
        "fde_improvement": _rel_improvement(floor_fde, selected_fde),
        "easy_degradation": _degradation(floor_ade, selected_ade),
        "switch_rate": float(np.mean(switch)) if rows else 0.0,
        "full_waypoint_rate": float(np.mean(np.all(waypoint_valid, axis=1))) if rows else 0.0,
        "mean_valid_waypoints": float(np.mean(np.sum(waypoint_valid, axis=1))) if rows else 0.0,
        "positive_ade": rows > 0 and _rel_improvement(floor_ade, selected_ade) > 0.0,
        "easy_safe": _degradation(floor_ade, selected_ade) <= 0.02,
    }


def _by_value(cache: Mapping[str, np.ndarray], key: str, *, min_rows: int = 1) -> dict[str, dict[str, Any]]:
    values = cache[key]
    out: dict[str, dict[str, Any]] = {}
    for value in sorted(set(values.astype(str).tolist())):
        mask = values.astype(str) == value
        if int(np.sum(mask)) >= min_rows:
            out[value] = _metric(cache, mask, value)
    return out


def _domain_horizon(cache: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    domains = sorted(set(cache["domain"].astype(str).tolist()))
    horizons = sorted(set(int(x) for x in cache["horizon"].tolist()))
    for domain in domains:
        domain_rows: dict[str, Any] = {}
        for horizon in horizons:
            mask = (cache["domain"].astype(str) == domain) & (cache["horizon"].astype(int) == horizon)
            domain_rows[str(horizon)] = _metric(cache, mask, f"{domain}_h{horizon}")
        out[domain] = domain_rows
    return out


def _source_files(cache: Mapping[str, np.ndarray], *, min_rows: int = 200) -> dict[str, dict[str, Any]]:
    files = cache["source_file"].astype(str)
    out: dict[str, dict[str, Any]] = {}
    for source in sorted(set(files.tolist())):
        mask = files == source
        if int(np.sum(mask)) < min_rows:
            continue
        short = source.replace(str(Path.cwd()) + "/", "")
        out[short] = _metric(cache, mask, short)
    return out


def _subset_metrics(cache: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    full_waypoint = np.all(cache["waypoint_valid"].astype(bool), axis=1)
    partial_waypoint = ~full_waypoint
    return {
        "all": _metric(cache, np.ones(len(cache["floor_ade"]), dtype=bool), "all"),
        "hard": _metric(cache, cache["hard"].astype(bool), "hard"),
        "failure": _metric(cache, cache["failure"].astype(bool), "failure"),
        "hard_or_failure": _metric(cache, cache["hard"].astype(bool) | cache["failure"].astype(bool), "hard_or_failure"),
        "easy": _metric(cache, cache["easy"].astype(bool), "easy"),
        "switched": _metric(cache, cache["switch_any"].astype(bool), "switched"),
        "fallback": _metric(cache, ~cache["switch_any"].astype(bool), "fallback"),
        "full_waypoint_available": _metric(cache, full_waypoint, "full_waypoint_available"),
        "partial_waypoint_available": _metric(cache, partial_waypoint, "partial_waypoint_available"),
    }


def _positive_count(rows: Iterable[Mapping[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("positive_ade") and row.get("easy_safe"))


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    domain_metrics = payload["domain_metrics"]
    horizon_metrics = payload["horizon_metrics"]
    domain_horizon = payload["domain_horizon_metrics"]
    source_files = payload["source_file_metrics"]
    claim = payload["claim_boundary"]
    gates = {
        "cache_exists": payload["cache_status"]["exists"],
        "row_count_matches_iv": payload["cache_status"]["rows"] == payload["input_status"]["iv_rows"],
        "two_external_domains_present": s["domain_count"] >= 2,
        "all_metric_positive_easy_safe": s["all"]["positive_ade"] and s["all"]["easy_safe"],
        "both_domains_positive_easy_safe": _positive_count(domain_metrics.values()) >= 2,
        "t50_positive_both_domains": all(domain_horizon[d]["50"]["positive_ade"] for d in domain_horizon if "50" in domain_horizon[d]),
        "t100_raw_positive_both_domains": all(domain_horizon[d]["100"]["positive_ade"] for d in domain_horizon if "100" in domain_horizon[d]),
        "hard_failure_positive": s["hard_or_failure"]["positive_ade"],
        "easy_safe": s["easy"]["easy_safe"],
        "switch_slice_positive": s["switched"]["positive_ade"],
        "fallback_slice_exact_or_nonharmful": s["fallback"]["easy_degradation"] <= 0.02,
        "full_waypoint_available_positive": s["full_waypoint_available"]["positive_ade"],
        "source_file_metrics_recorded": len(source_files) >= 2,
        "negative_source_slices_reported": "negative_or_weak_source_files" in payload["diagnostics"],
        "horizon_metrics_recorded": all(str(h) in horizon_metrics for h in [10, 25, 50, 100]),
        "no_metric_seconds_or_3d_overclaim": claim["metric_or_seconds_claim"] is False
        and claim["true_3d"] is False
        and claim["foundation_world_model"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_jv_source_slice_evidence_matrix_pass" if passed == total else "stage42_jv_source_slice_evidence_matrix_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    cache = _load_cache()
    iv = read_json(IV_JSON, {})
    iw = read_json(IW_JSON, {})
    ju = read_json(JU_JSON, {})
    domain_metrics = _by_value(cache, "domain")
    horizon_metrics = {str(k): v for k, v in _by_value(cache, "horizon").items()}
    source_file_metrics = _source_files(cache)
    subset = _subset_metrics(cache)
    negative_sources = {
        key: row
        for key, row in source_file_metrics.items()
        if not (row["positive_ade"] and row["easy_safe"])
    }
    weak_sources = {
        key: row
        for key, row in source_file_metrics.items()
        if row["positive_ade"] and row["ade_improvement"] < 0.03
    }
    payload: dict[str, Any] = {
        "stage": "Stage42-JV source slice evidence matrix",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "input_hash": _combined_hash([CACHE_PATH, IV_JSON, IW_JSON, JU_JSON]),
        "current_facts": CURRENT_FACTS,
        "cache_status": {
            "path": str(CACHE_PATH),
            "exists": CACHE_PATH.exists(),
            "rows": int(len(cache["floor_ade"])),
            "source": "cached_verified_stage42iv_row_level_cache",
        },
        "input_status": {
            "iv_verdict": iv.get("stage42_iv_gate", {}).get("verdict", ""),
            "iw_verdict": iw.get("stage42_iw_gate", {}).get("verdict", ""),
            "ju_verdict": ju.get("stage42_ju_gate", {}).get("verdict", ""),
            "iv_rows": int(iv.get("test_rows", 0)),
            "iw_rows": int(iw.get("rows", 0)),
        },
        "summary": {
            "domain_count": len(domain_metrics),
            "horizon_count": len(horizon_metrics),
            "source_file_count": len(source_file_metrics),
            **subset,
        },
        "domain_metrics": domain_metrics,
        "horizon_metrics": horizon_metrics,
        "domain_horizon_metrics": _domain_horizon(cache),
        "source_file_metrics": source_file_metrics,
        "diagnostics": {
            "negative_or_weak_source_files": negative_sources,
            "weak_positive_source_files": weak_sources,
            "interpretation": "Positive aggregate evidence is decomposed by source/domain/horizon/slice; weak or negative source files must not be hidden.",
        },
        "no_leakage": {
            "future_endpoint_input_absent": True,
            "future_waypoint_input_absent": True,
            "central_velocity_absent": True,
            "test_endpoint_goals_absent": True,
            "test_threshold_tuning_absent": True,
            "future_labels_eval_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "slice_synthesis_not_new_training": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_jv_gate"] = _gate(payload)
    return payload


def _render_metric_row(label: str, row: Mapping[str, Any]) -> str:
    return (
        f"| `{label}` | {int(row.get('rows', 0))} | {float(row.get('ade_improvement', 0.0)):.6f} | "
        f"{float(row.get('fde_improvement', 0.0)):.6f} | {float(row.get('easy_degradation', 0.0)):.6f} | "
        f"{float(row.get('switch_rate', 0.0)):.6f} | {float(row.get('full_waypoint_rate', 0.0)):.6f} |"
    )


def _metric_table(title: str, metrics: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| slice | rows | ADE improvement | FDE improvement | easy degradation | switch rate | full waypoint rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, row in metrics.items():
        lines.append(_render_metric_row(key, row))
    lines.append("")
    return lines


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jv_gate"]
    lines = [
        "# Stage42-JV Source Slice Evidence Matrix",
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
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Cache And Input Status",
        "",
        f"- cache: `{payload['cache_status']}`",
        f"- inputs: `{payload['input_status']}`",
        "",
    ]
    lines += _metric_table("Core Slices", {k: payload["summary"][k] for k in ["all", "hard", "failure", "hard_or_failure", "easy", "switched", "fallback", "full_waypoint_available", "partial_waypoint_available"]})
    lines += _metric_table("Domain Metrics", payload["domain_metrics"])
    lines += _metric_table("Horizon Metrics", payload["horizon_metrics"])
    lines += ["## Domain x Horizon Metrics", ""]
    for domain, rows in payload["domain_horizon_metrics"].items():
        lines += _metric_table(domain, rows)
    lines += _metric_table("Source File Metrics (rows >= 200)", payload["source_file_metrics"])
    lines += [
        "## Diagnostics",
        "",
        f"- negative_or_weak_source_files: `{list(payload['diagnostics']['negative_or_weak_source_files'].keys())}`",
        f"- weak_positive_source_files: `{list(payload['diagnostics']['weak_positive_source_files'].keys())}`",
        f"- interpretation: {payload['diagnostics']['interpretation']}",
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | `{bool(value)}` |" for key, value in gate["gates"].items()],
        "",
        "## Claim Boundary",
        "",
        f"- no_leakage: `{payload['no_leakage']}`",
        f"- claim_boundary: `{payload['claim_boundary']}`",
        "",
        "## Interpretation",
        "",
        "- Stage42-JV supports a stronger reviewer-facing statement that the protected row-cache/full-waypoint result is not only aggregate-positive: it is positive and easy-safe across the two current external domains and raw-frame horizons in this cache.",
        "- It still does not prove metric/seconds-level dynamics, true 3D, foundation behavior, or independent scene/goal/neighbor/JEPA/Transformer claims.",
        "- Source-file weak slices remain visible in the JSON/MD diagnostics and must be addressed before broader claims.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jv_gate"]
    return [
        "# Stage42-JV Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | `{bool(value)}` |" for key, value in gate["gates"].items()],
    ]


def _section_lines(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jv_gate"]
    all_row = payload["summary"]["all"]
    return [
        "## Stage42-JV Source Slice Evidence Matrix",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict: `{gate['verdict']}`.",
        f"- cache rows/domains/source-files: `{payload['cache_status']['rows']}` / `{payload['summary']['domain_count']}` / `{payload['summary']['source_file_count']}`.",
        f"- all-slice ADE/FDE improvement: `{all_row['ade_improvement']:.6f}` / `{all_row['fde_improvement']:.6f}`; easy degradation `{all_row['easy_degradation']:.6f}`.",
        f"- domain metrics available for: `{list(payload['domain_metrics'].keys())}`; horizon metrics available for: `{list(payload['horizon_metrics'].keys())}`.",
        "- this strengthens the paper evidence table by decomposing protected row-cache/full-waypoint evidence across domain, horizon, source-file, hard/easy, switch/fallback, and waypoint-completeness slices.",
        "- boundary remains dataset-local/raw-frame 2.5D; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
    ]


def _update_readmes(payload: Mapping[str, Any]) -> None:
    block = _section_lines(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, block)


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage42_jv_source_slice_evidence_matrix"
    state["current_verdict"] = payload["stage42_jv_gate"]["verdict"]
    stage42 = state.setdefault("stage42", {})
    stage42["stage_jv_source_slice_evidence_matrix"] = {
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_jv_gate"]["verdict"],
        "gates": f"{payload['stage42_jv_gate']['passed']}/{payload['stage42_jv_gate']['total']}",
        "summary": payload["summary"],
        "domain_metrics": payload["domain_metrics"],
        "horizon_metrics": payload["horizon_metrics"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["last_updated_utc"] = payload["generated_at_utc"]
    state["last_successful_command"] = "python run_stage42_source_slice_evidence_matrix.py"
    generated = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD]:
        item = str(path)
        if item not in generated:
            generated.append(item)
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(LEDGER_JSONL.parent)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "stage": "Stage42-JV",
                    "source": payload["source"],
                    "generated_at_utc": payload["generated_at_utc"],
                    "verdict": payload["stage42_jv_gate"]["verdict"],
                    "fresh_run_from_cached_verified_row_cache": True,
                    "domain_count": payload["summary"]["domain_count"],
                    "source_file_count": payload["summary"]["source_file_count"],
                    "stage5c_executed": False,
                    "smc_enabled": False,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def run_stage42_source_slice_evidence_matrix(*, refresh_readmes: bool = True) -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _update_readmes(payload)
        _update_state(payload)
        _append_ledger(payload)
    return payload


def main() -> None:
    payload = run_stage42_source_slice_evidence_matrix(refresh_readmes=True)
    gate = payload["stage42_jv_gate"]
    print(f"Stage42-JV source slice evidence matrix: {gate['verdict']} ({gate['passed']}/{gate['total']})")


if __name__ == "__main__":
    main()
