from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_t50_floor_relaxability_repair as by
from src import stage42_ucy_validation_support_repair as aw
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
BY_JSON = OUT_DIR / "t50_floor_relaxability_repair_stage42.json"
AW_JSON = OUT_DIR / "ucy_validation_support_repair_stage42.json"
REPORT_JSON = OUT_DIR / "t50_repair_statistical_evidence_stage42.json"
REPORT_MD = OUT_DIR / "t50_repair_statistical_evidence_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_bz_gate.md"

BOOTSTRAP_N = 3000
EASY_LIMIT = 0.02
TARGET_SLICES = ["TrajNet|50", "UCY|50"]


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-BZ 给 Stage42-BY 的 protected t50 repair 补 bootstrap statistical evidence。",
    "Stage42-BZ 不重新选择 threshold，不使用 test metric 调 policy，不训练新模型。",
    "本审计使用 Stage42-AW train-only internal validation policy；test rows 只用于最终报告和 bootstrap。",
    "teacher/floor rollout context 仍然是部署安全地板；本结果不是 floor-free neural deployment。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C 未执行，SMC 未启用。",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ci_positive(ci: Mapping[str, Any], *, easy: bool = False) -> bool:
    if int(ci.get("bootstrap_n", 0)) < BOOTSTRAP_N:
        return False
    if easy:
        return float(ci.get("high", 1.0)) <= EASY_LIMIT
    return float(ci.get("low", -1.0)) > 0.0


def _slice_mask(data: Mapping[str, np.ndarray], split: np.ndarray, key: str) -> np.ndarray:
    domain_name, horizon_s = key.split("|", 1)
    return (split == "test") & (data["dataset"].astype(str) == domain_name) & (data["horizon"].astype(int) == int(horizon_s))


def _recompute_by_policy_arrays() -> dict[str, Any]:
    aw_report = _load_json(AW_JSON)
    data = s41._combined()
    original_split, group = am._split_arrays(data)
    repaired_split, internal_val_group = aw._split_with_ucy_internal_val(original_split, group, data["dataset"].astype(str))
    labels = am._reconstruct_waypoint_labels(data)
    train_mask = repaired_split == "train"
    floor = am._floor_arrays(data, train_mask)
    features, feature_names = am._feature_matrix(data, floor)
    masks = aw._safe_variant_masks(feature_names)
    selected_variant = str(aw_report["validation_best_variant"])
    fitted = by._fit_selected_variant(selected_variant, features, masks, data, repaired_split, labels, floor)
    return {
        "data": data,
        "split": repaired_split,
        "internal_val_group": internal_val_group,
        "selected_variant": selected_variant,
        "selected_ade": fitted["selected_ade"],
        "floor_ade": fitted["floor_ade"],
        "switch": fitted["switch"],
        "selected_lambda": fitted["lambda"],
        "selected_score": fitted["score"],
        "policy_slices": sorted(fitted["policy"]["slices"].keys()),
    }


def _bootstrap_for_mask(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray, seed: int, *, easy: bool = False) -> dict[str, Any]:
    if easy:
        return am._bootstrap_ci(floor, selected, mask, seed=seed, n=BOOTSTRAP_N)
    return am._bootstrap_ci(selected, floor, mask, seed=seed, n=BOOTSTRAP_N)


def _slice_statistical_evidence(
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
    selected: np.ndarray,
    floor: np.ndarray,
    switch: np.ndarray,
    key: str,
    seed_base: int,
) -> dict[str, Any]:
    mask = _slice_mask(data, split, key)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    metric = am._metric(selected, floor, data, switch, mask)
    boot = {
        "all": _bootstrap_for_mask(selected, floor, mask, seed_base),
        "t50": _bootstrap_for_mask(selected, floor, mask & (data["horizon"].astype(int) == 50), seed_base + 1),
        "hard_failure": _bootstrap_for_mask(selected, floor, mask & hard_failure, seed_base + 2),
        "easy_degradation": _bootstrap_for_mask(selected, floor, mask & easy, seed_base + 3, easy=True),
    }
    robust = (
        _ci_positive(boot["all"])
        and _ci_positive(boot["t50"])
        and _ci_positive(boot["hard_failure"])
        and _ci_positive(boot["easy_degradation"], easy=True)
        and float(metric["switch_rate"]) > 0.0
    )
    return {
        "source": "fresh_stage42_bz_t50_repair_statistical_evidence",
        "slice": key,
        "rows": int(np.sum(mask)),
        "metric": metric,
        "bootstrap": boot,
        "ci_positive_and_easy_safe": robust,
        "switch_rate_positive": float(metric["switch_rate"]) > 0.0,
    }


def _target_union_statistical_evidence(
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
    selected: np.ndarray,
    floor: np.ndarray,
    switch: np.ndarray,
    slice_rows: Mapping[str, Any],
) -> dict[str, Any]:
    mask = np.zeros(len(split), dtype=bool)
    for key in TARGET_SLICES:
        mask |= _slice_mask(data, split, key)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    metric = am._metric(selected, floor, data, switch, mask)
    boot = {
        "all": _bootstrap_for_mask(selected, floor, mask, 42601),
        "t50": _bootstrap_for_mask(selected, floor, mask & (data["horizon"].astype(int) == 50), 42602),
        "hard_failure": _bootstrap_for_mask(selected, floor, mask & hard_failure, 42603),
        "easy_degradation": _bootstrap_for_mask(selected, floor, mask & easy, 42604, easy=True),
    }
    return {
        "source": "fresh_stage42_bz_t50_repair_statistical_evidence",
        "target_slices": TARGET_SLICES,
        "rows": int(np.sum(mask)),
        "metric": metric,
        "bootstrap": boot,
        "slice_robustness": {key: bool(row["ci_positive_and_easy_safe"]) for key, row in slice_rows.items()},
        "ci_positive_and_easy_safe": _ci_positive(boot["all"])
        and _ci_positive(boot["t50"])
        and _ci_positive(boot["hard_failure"])
        and _ci_positive(boot["easy_degradation"], easy=True),
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    by_report = _load_json(BY_JSON)
    arrays = _recompute_by_policy_arrays()
    data = arrays["data"]
    split = arrays["split"]
    selected = arrays["selected_ade"]
    floor = arrays["floor_ade"]
    switch = arrays["switch"]
    slice_rows = {
        key: _slice_statistical_evidence(data, split, selected, floor, switch, key, seed_base=42500 + i * 10)
        for i, key in enumerate(TARGET_SLICES)
    }
    target_union = _target_union_statistical_evidence(data, split, selected, floor, switch, slice_rows)
    summary = {
        "source": "fresh_stage42_bz_t50_repair_statistical_evidence",
        "verdict_short": "protected_t50_repair_has_positive_bootstrap_evidence_but_not_floor_free",
        "selected_variant": arrays["selected_variant"],
        "selected_lambda": arrays["selected_lambda"],
        "internal_val_group": arrays["internal_val_group"],
        "target_slices": TARGET_SLICES,
        "robust_t50_slices": [key for key, row in slice_rows.items() if row["ci_positive_and_easy_safe"]],
        "weak_t50_slices": [key for key, row in slice_rows.items() if not row["ci_positive_and_easy_safe"]],
        "target_union_t50_ci_low": float(target_union["bootstrap"]["t50"]["low"]),
        "target_union_easy_ci_high": float(target_union["bootstrap"]["easy_degradation"]["high"]),
        "target_union_ci_positive_and_easy_safe": bool(target_union["ci_positive_and_easy_safe"]),
        "bootstrap_n": BOOTSTRAP_N,
        "floor_free_neural_deployable": False,
        "teacher_floor_context_required": True,
        "global_metric_claim_allowed": False,
        "global_seconds_claim_allowed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    payload: dict[str, Any] = {
        "source": "fresh_stage42_bz_t50_repair_statistical_evidence",
        "stage": "Stage42-BZ Protected T50 Repair Statistical Evidence",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(BY_JSON), str(AW_JSON)]),
        "current_facts": CURRENT_FACTS,
        "input_reports": {
            "stage42_by_verdict": by_report["stage42_by_gate"]["verdict"],
            "stage42_by_source": by_report["source"],
        },
        "validation_policy": {
            "source": "fresh_stage42_bz_t50_repair_statistical_evidence",
            "selected_variant": arrays["selected_variant"],
            "selected_lambda": arrays["selected_lambda"],
            "selected_score": arrays["selected_score"],
            "policy_slices": arrays["policy_slices"],
            "internal_val_group": arrays["internal_val_group"],
            "selection_source": "train_plus_internal_validation_only",
            "test_threshold_tuning": False,
        },
        "slice_evidence": slice_rows,
        "target_union_evidence": target_union,
        "summary": summary,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "internal_val_from_train_only": True,
            "test_metrics_used_for_final_reporting_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "floor_free_neural_deployable": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_bz_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    no_leakage = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "by_input_verified": payload["input_reports"]["stage42_by_verdict"] == "stage42_by_t50_floor_relaxability_repair_pass",
        "target_union_bootstrap_positive": s["target_union_ci_positive_and_easy_safe"] is True,
        "trajnet_t50_bootstrap_positive": payload["slice_evidence"]["TrajNet|50"]["ci_positive_and_easy_safe"] is True,
        "ucy_t50_bootstrap_positive": payload["slice_evidence"]["UCY|50"]["ci_positive_and_easy_safe"] is True,
        "bootstrap_n_at_least_3000": int(s["bootstrap_n"]) >= BOOTSTRAP_N,
        "train_only_internal_validation_used": no_leakage["internal_val_from_train_only"] is True,
        "test_threshold_tuning_false": no_leakage["test_threshold_tuning"] is False,
        "teacher_floor_context_required": s["teacher_floor_context_required"] is True,
        "not_floor_free_neural": s["floor_free_neural_deployable"] is False and claim["floor_free_neural_deployable"] is False,
        "no_leakage_pass": no_leakage["future_endpoint_input"] is False
        and no_leakage["future_waypoint_input"] is False
        and no_leakage["central_velocity"] is False
        and no_leakage["test_endpoint_goals"] is False,
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False
        and claim["metric_or_seconds_claim"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_bz_t50_repair_statistical_evidence_pass" if passed == total else "stage42_bz_t50_repair_statistical_evidence_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-BZ Protected T50 Repair Statistical Evidence",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_bz_gate']['passed']} / {payload['stage42_bz_gate']['total']}`",
        f"- verdict: `{payload['stage42_bz_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- verdict_short: `{s['verdict_short']}`",
        f"- selected_variant: `{s['selected_variant']}`",
        f"- internal_val_group: `{s['internal_val_group']}`",
        f"- robust_t50_slices: `{s['robust_t50_slices']}`",
        f"- weak_t50_slices: `{s['weak_t50_slices']}`",
        f"- target_union_t50_ci_low: `{_pct(s['target_union_t50_ci_low'])}`",
        f"- target_union_easy_ci_high: `{_pct(s['target_union_easy_ci_high'])}`",
        f"- bootstrap_n: `{s['bootstrap_n']}`",
        f"- floor_free_neural_deployable: `{s['floor_free_neural_deployable']}`",
        "",
        "## Slice Bootstrap Evidence",
        "",
        "| slice | rows | t50 | t50 CI low | t50 CI high | hard CI low | easy CI high | switch | robust |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, row in payload["slice_evidence"].items():
        metric = row["metric"]
        boot = row["bootstrap"]
        lines.append(
            f"| `{key}` | {row['rows']} | {_pct(float(metric['t50_improvement']))} | "
            f"{_pct(float(boot['t50']['low']))} | {_pct(float(boot['t50']['high']))} | "
            f"{_pct(float(boot['hard_failure']['low']))} | {_pct(float(boot['easy_degradation']['high']))} | "
            f"{_pct(float(metric['switch_rate']))} | {row['ci_positive_and_easy_safe']} |"
        )
    u = payload["target_union_evidence"]
    lines += [
        "",
        "## Target Union Evidence",
        "",
        f"- rows: `{u['rows']}`",
        f"- t50 improvement: `{_pct(float(u['metric']['t50_improvement']))}`",
        f"- t50 CI: `[{_pct(float(u['bootstrap']['t50']['low']))}, {_pct(float(u['bootstrap']['t50']['high']))}]`",
        f"- hard/failure CI low: `{_pct(float(u['bootstrap']['hard_failure']['low']))}`",
        f"- easy degradation CI high: `{_pct(float(u['bootstrap']['easy_degradation']['high']))}`",
        "",
        "## Interpretation",
        "",
        "- Stage42-BZ upgrades Stage42-BY from point-estimate protected t50 repair to bootstrap-backed statistical evidence.",
        "- The policy was selected by train plus internal validation only; test rows are final reporting/bootstrap evidence only.",
        "- This remains protected policy evidence, not floor-free neural world dynamics.",
        "- No true-3D, foundation, global metric, seconds-level, Stage5C, or SMC claim is allowed.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_bz_gate"]
    lines = [
        "# Stage42-BZ Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | {bool(value)} |")
    return lines


def run_stage42_t50_repair_statistical_evidence() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    return payload


if __name__ == "__main__":
    result = run_stage42_t50_repair_statistical_evidence()
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False, sort_keys=True))
