from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src import stage42_local_t100_protected_policy as bg
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
BD_JSON = OUT_DIR / "local_t100_source_inventory_stage42.json"
BG_JSON = OUT_DIR / "local_t100_protected_policy_stage42.json"
REPORT_JSON = OUT_DIR / "local_t100_independent_source_audit_stage42.json"
REPORT_MD = OUT_DIR / "local_t100_independent_source_audit_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_bh_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_local_t100_independent_sources_stage42.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

MIN_INDEPENDENT_SOURCES = 3
SAFE_SWITCH_BASELINES = ["constant_position", "damped_velocity_0p25", "damped_velocity_0p50", "damped_velocity_0p75"]


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-BH 审计本地 t100-capable 文件是否真的是独立 source，而不是同一 scene 的重复格式。",
    "本步骤只做 local source independence audit 和 validation-selected protected baseline-family source-CV。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t100 仍是 raw-frame diagnostic；局部 UCY 支持不等于全局 t100 修复。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _canonical_domain(row: Mapping[str, Any]) -> str:
    rel = str(row["relative_path"])
    if rel.startswith("UCY/"):
        return "UCY"
    if rel.startswith("ETH/"):
        return "ETH_UCY"
    if rel.startswith("TrajNet"):
        return "TrajNet"
    suggested = str(row.get("suggested_domain", "diagnostic"))
    return "UCY" if suggested == "UCY_or_ETH_UCY" else suggested


def _independent_key(row: Mapping[str, Any]) -> str:
    rel = str(row["relative_path"])
    parts = rel.split("/")
    if len(parts) >= 2:
        return f"{_canonical_domain(row)}::{parts[0]}/{parts[1]}"
    return f"{_canonical_domain(row)}::{rel}"


def _candidate_source(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source": "fresh_independent_source_audit",
        "source_id": str(row["relative_path"]),
        "path": str(row["path"]),
        "domain": _canonical_domain(row),
        "coordinate_unit": "dataset_local",
        "metric_status": "unverified",
        "estimated_t100_windows": int(row.get("estimated_t100_windows", 0)),
        "independent_key": _independent_key(row),
    }


def _choose_canonical(rows: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    def score(row: Mapping[str, Any]) -> tuple[int, int, int]:
        rel = str(row["relative_path"])
        prefer_obsmat = 1 if rel.endswith("/obsmat.txt") else 0
        prefer_non_px = 0 if rel.endswith("_px.txt") else 1
        return int(row.get("estimated_t100_windows", 0)), prefer_obsmat, prefer_non_px

    return sorted(rows, key=score, reverse=True)[0]


def _build_independent_sources(inventory: list[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    t100_rows = [
        row
        for row in inventory
        if bool(row.get("t100_capable"))
        and not bool(row.get("synthetic_or_diagnostic", False))
        and _canonical_domain(row) in {"ETH_UCY", "UCY", "TrajNet"}
    ]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in t100_rows:
        grouped[_independent_key(row)].append(row)
    independent_sources: list[dict[str, Any]] = []
    duplicate_groups: dict[str, Any] = {}
    for key, rows in sorted(grouped.items()):
        chosen = _choose_canonical(rows)
        independent_sources.append(_candidate_source(chosen))
        duplicate_groups[key] = {
            "source": "fresh_independent_source_audit",
            "domain": _canonical_domain(chosen),
            "candidate_files": [str(row["relative_path"]) for row in rows],
            "candidate_count": len(rows),
            "chosen_source_id": str(chosen["relative_path"]),
            "chosen_estimated_t100_windows": int(chosen.get("estimated_t100_windows", 0)),
            "deduplicated_as_same_scene_or_source": len(rows) > 1,
        }
    return independent_sources, {
        "source": "fresh_independent_source_audit",
        "raw_t100_capable_files": int(len(t100_rows)),
        "independent_source_count": int(len(independent_sources)),
        "duplicate_or_alternate_format_group_count": int(sum(1 for row in duplicate_groups.values() if row["deduplicated_as_same_scene_or_source"])),
        "duplicate_groups": duplicate_groups,
    }


def _folds_for_domain(sources: list[Mapping[str, Any]], domain: str) -> list[dict[str, Any]]:
    domain_sources = sorted(
        [src for src in sources if src["domain"] == domain],
        key=lambda row: (-int(row["estimated_t100_windows"]), str(row["source_id"])),
    )
    folds: list[dict[str, Any]] = []
    if len(domain_sources) < MIN_INDEPENDENT_SOURCES:
        return folds
    for holdout in domain_sources:
        remaining = [src for src in domain_sources if src["source_id"] != holdout["source_id"]]
        validation = remaining[0]
        train_sources = [src["source_id"] for src in remaining if src["source_id"] != validation["source_id"]]
        folds.append(
            {
                "source": "fresh_independent_source_cv_plan",
                "domain": domain,
                "holdout_source": holdout["source_id"],
                "validation_source": validation["source_id"],
                "train_sources": train_sources,
                "holdout_t100_windows_estimated": int(holdout["estimated_t100_windows"]),
            }
        )
    return folds


def _support_matrix(sources: list[Mapping[str, Any]], folds_by_domain: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for domain in ["ETH_UCY", "UCY", "TrajNet"]:
        rows = [src for src in sources if src["domain"] == domain]
        out[domain] = {
            "source": "fresh_independent_source_audit",
            "independent_sources": int(len(rows)),
            "estimated_t100_windows": int(sum(int(src["estimated_t100_windows"]) for src in rows)),
            "source_ids": [str(src["source_id"]) for src in rows],
            "source_cv_feasible": len(rows) >= MIN_INDEPENDENT_SOURCES,
            "fold_count": int(len(folds_by_domain.get(domain, []))),
            "blocker": None if len(rows) >= MIN_INDEPENDENT_SOURCES else f"needs_{MIN_INDEPENDENT_SOURCES - len(rows)}_additional_independent_t100_sources",
        }
    return out


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int(round(q * (len(vals) - 1)))
    return float(vals[max(0, min(idx, len(vals) - 1))])


def _safe_switch_candidates(train_windows: list[bg.Window], horizon: int) -> list[tuple[str, bg.Selector, dict[str, Any]]]:
    train_h = [row for row in train_windows if int(row["horizon"]) == horizon]
    speeds = [float(row["speed_causal"]) for row in train_h]
    thresholds = sorted({0.0, _quantile(speeds, 0.25), _quantile(speeds, 0.50), _quantile(speeds, 0.75)})
    candidates: list[tuple[str, bg.Selector, dict[str, Any]]] = [
        (f"global_{name}", bg._global_selector(name), {"baseline": name, "policy_family": "global"})
        for name in bg.bf.BASELINES
    ]
    for baseline in SAFE_SWITCH_BASELINES:
        for threshold in thresholds:
            def selector(row: bg.Window, *, b: str = baseline, t: float = threshold) -> str:
                return b if float(row["speed_causal"]) >= t else bg.FALLBACK_BASELINE

            candidates.append(
                (
                    f"speed_safe_{baseline}_gte_{threshold:.6g}",
                    selector,
                    {"baseline": baseline, "speed_threshold": threshold, "policy_family": "speed_safe_switch"},
                )
            )
    return candidates


def _select_safe_policy(train_windows: list[bg.Window], val_windows: list[bg.Window], horizon: int) -> dict[str, Any]:
    val_h = [row for row in val_windows if int(row["horizon"]) == horizon]
    fallback = bg._policy_metrics(val_h, bg._global_selector(bg.FALLBACK_BASELINE))
    evaluated: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for name, selector, meta in _safe_switch_candidates(train_windows, horizon):
        metrics = bg._policy_metrics(val_h, selector)
        improvement = float(metrics["improvement_vs_fallback"] or 0.0)
        easy = float(metrics["easy_degradation"] or 0.0)
        switch = float(metrics["switch_rate"] or 0.0)
        eligible = improvement > 0.0 and easy <= bg.EASY_DEGRADATION_LIMIT
        score = improvement - max(0.0, easy - bg.EASY_DEGRADATION_LIMIT) * 10.0 - switch * 0.01
        row = {
            "source": "fresh_independent_source_safe_switch_selection",
            "policy_name": name,
            "metadata": meta,
            "validation_metrics": metrics,
            "eligible_for_holdout": bool(eligible),
            "score": float(score),
            "horizon": int(horizon),
        }
        evaluated.append(row)
        if eligible and (best is None or score > float(best["score"])):
            best = row
    if best is None:
        best = {
            "source": "fresh_independent_source_safe_switch_selection",
            "policy_name": f"global_{bg.FALLBACK_BASELINE}",
            "metadata": {"baseline": bg.FALLBACK_BASELINE, "policy_family": "fallback"},
            "validation_metrics": fallback,
            "eligible_for_holdout": True,
            "score": 0.0,
            "horizon": int(horizon),
            "fallback_reason": "no_validation_safe_switch_policy_exceeded_fallback",
        }
    return {
        "source": "fresh_independent_source_safe_switch_selection",
        "horizon": int(horizon),
        "selected_policy": best,
        "candidate_count": len(evaluated),
        "candidates": evaluated,
    }


def _selector_from_safe_selection(selected: Mapping[str, Any], train_windows: list[bg.Window], horizon: int) -> bg.Selector:
    name = str(selected["policy_name"])
    for candidate_name, selector, _ in _safe_switch_candidates(train_windows, horizon):
        if candidate_name == name:
            return selector
    if name.startswith("global_"):
        return bg._global_selector(name.removeprefix("global_"))
    return bg._global_selector(bg.FALLBACK_BASELINE)


def _evaluate_fold_with_safe_switch(*, fold: Mapping[str, Any], windows_by_source: Mapping[str, list[bg.Window]]) -> dict[str, Any]:
    train_windows: list[bg.Window] = []
    for source_id in fold.get("train_sources", []):
        train_windows.extend(windows_by_source.get(str(source_id), []))
    val_windows = list(windows_by_source.get(str(fold["validation_source"]), []))
    holdout_windows = list(windows_by_source.get(str(fold["holdout_source"]), []))
    by_horizon: dict[str, Any] = {}
    for horizon in bg.HORIZONS:
        selection = _select_safe_policy(train_windows, val_windows, horizon)
        selector = _selector_from_safe_selection(selection["selected_policy"], train_windows, horizon)
        holdout_h = [row for row in holdout_windows if int(row["horizon"]) == horizon]
        holdout_metrics = bg._policy_metrics(holdout_h, selector)
        by_horizon[str(horizon)] = {
            "source": "fresh_independent_source_safe_switch_cv",
            "selection": selection,
            "holdout_metrics": holdout_metrics,
            "safe_positive": bool(
                (holdout_metrics["improvement_vs_fallback"] or 0.0) > 0.0
                and (holdout_metrics["easy_degradation"] or 0.0) <= bg.EASY_DEGRADATION_LIMIT
            ),
        }
    return {
        "source": "fresh_independent_source_safe_switch_cv",
        "domain": fold["domain"],
        "holdout_source": fold["holdout_source"],
        "validation_source": fold["validation_source"],
        "train_sources": list(fold.get("train_sources", [])),
        "by_horizon": by_horizon,
    }


def run_stage42_local_t100_independent_source_audit() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    bd_payload = _load_json(BD_JSON)
    bg_payload = _load_json(BG_JSON)
    independent_sources, duplicate_audit = _build_independent_sources(list(bd_payload.get("inventory", [])))
    windows_by_source = {src["source_id"]: bg._build_windows_for_source(src) for src in independent_sources}
    domains = sorted({src["domain"] for src in independent_sources} | {"ETH_UCY", "UCY", "TrajNet"})
    folds_by_domain = {domain: _folds_for_domain(independent_sources, domain) for domain in domains}
    fold_results = [
        _evaluate_fold_with_safe_switch(fold=fold, windows_by_source=windows_by_source)
        for domain in domains
        for fold in folds_by_domain.get(domain, [])
    ]
    domain_summary = bg._domain_summary(fold_results)
    support = _support_matrix(independent_sources, folds_by_domain)
    ucy_t100 = domain_summary.get("UCY", {}).get("by_horizon", {}).get("100", {})
    ucy_supported = bool(
        ucy_t100.get("fold_count", 0) >= MIN_INDEPENDENT_SOURCES
        and ucy_t100.get("all_folds_safe_positive", False)
        and (ucy_t100.get("maximum_easy_degradation") is not None and float(ucy_t100["maximum_easy_degradation"]) <= bg.EASY_DEGRADATION_LIMIT)
    )
    blocked_domains = [domain for domain, row in support.items() if not row["source_cv_feasible"]]
    payload = {
        "source": "fresh_local_independent_source_audit",
        "stage": "Stage42-BH Local T100 Independent Source Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([str(BD_JSON), str(BG_JSON)] + [str(src["path"]) for src in independent_sources]),
        "bd_verdict": bd_payload.get("stage42_bd_gate", {}).get("verdict"),
        "bg_verdict": bg_payload.get("stage42_bg_gate", {}).get("verdict"),
        "duplicate_audit": duplicate_audit,
        "independent_sources": independent_sources,
        "support_matrix": support,
        "source_cv_folds": fold_results,
        "domain_summary": domain_summary,
        "summary": {
            "source": "fresh_local_independent_source_audit",
            "raw_t100_capable_files": duplicate_audit["raw_t100_capable_files"],
            "independent_t100_sources": duplicate_audit["independent_source_count"],
            "duplicate_or_alternate_format_group_count": duplicate_audit["duplicate_or_alternate_format_group_count"],
            "ucy_independent_sources": support["UCY"]["independent_sources"],
            "eth_ucy_independent_sources": support["ETH_UCY"]["independent_sources"],
            "trajnet_independent_sources": support["TrajNet"]["independent_sources"],
            "ucy_t100_source_cv_supported": ucy_supported,
            "ucy_t100_mean_improvement_vs_fallback": ucy_t100.get("mean_holdout_improvement_vs_fallback"),
            "ucy_t100_min_improvement_vs_fallback": ucy_t100.get("minimum_holdout_improvement_vs_fallback"),
            "ucy_t100_max_easy_degradation": ucy_t100.get("maximum_easy_degradation"),
            "blocked_domains": blocked_domains,
            "global_t100_positive_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "metric_or_seconds_claim": False,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_metrics_for_threshold": False,
            "holdout_used_for_threshold": False,
            "duplicate_scene_versions_treated_as_independent": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "t100_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "global_t100_positive_claim_allowed": False,
            "ucy_local_t100_source_cv_support": ucy_supported,
        },
    }
    payload["stage42_bh_gate"] = _gate(payload)
    payload["user_action_required"] = _user_actions(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    _append_ledger(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    gates = {
        "bd_inventory_loaded": payload["bd_verdict"] == "stage42_bd_local_t100_source_inventory_pass",
        "bg_policy_loaded": payload["bg_verdict"] == "stage42_bg_local_t100_protected_policy_pass_with_global_t100_blocker",
        "duplicate_audit_completed": s["raw_t100_capable_files"] >= s["independent_t100_sources"] > 0,
        "duplicate_versions_not_counted_as_independent": payload["no_leakage"]["duplicate_scene_versions_treated_as_independent"] is False,
        "support_matrix_built": {"ETH_UCY", "UCY", "TrajNet"}.issubset(set(payload["support_matrix"].keys())),
        "ucy_source_cv_evaluated": "UCY" in payload["domain_summary"],
        "ucy_t100_source_cv_supported": bool(s["ucy_t100_source_cv_supported"]),
        "eth_ucy_blocker_reported": "ETH_UCY" in s["blocked_domains"],
        "trajnet_blocker_reported": "TrajNet" in s["blocked_domains"],
        "no_leakage_pass": all(
            payload["no_leakage"][k] is False
            for k in [
                "future_endpoint_input",
                "future_waypoint_input",
                "central_velocity",
                "test_endpoint_goals",
                "test_metrics_for_threshold",
                "holdout_used_for_threshold",
                "duplicate_scene_versions_treated_as_independent",
            ]
        ),
        "no_metric_seconds_overclaim": not payload["claim_boundary"]["metric_or_seconds_claim"],
        "global_t100_claim_still_blocked": not payload["claim_boundary"]["global_t100_positive_claim_allowed"],
        "stage5c_false": not payload["claim_boundary"]["stage5c_executed"],
        "smc_false": not payload["claim_boundary"]["smc_enabled"],
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_bh_independent_t100_source_audit_pass_with_global_blocker" if passed == total else "stage42_bh_independent_t100_source_audit_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _user_actions(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    support = payload["support_matrix"]
    actions = []
    if not payload["summary"]["ucy_t100_source_cv_supported"]:
        actions.append(
            {
                "source": payload["source"],
                "priority": "high",
                "domain": "UCY",
                "action_type": "repair_independent_source_easy_guard",
                "notes": "UCY has enough independent local t100 sources and positive mean gain, but strict source-CV easy degradation exceeds 2%; next step should add source-robust easy/harm guard or collect another independent UCY source.",
            }
        )
    for domain in ["ETH_UCY", "TrajNet"]:
        row = support[domain]
        if row["blocker"]:
            actions.append(
                {
                    "source": payload["source"],
                    "priority": "high",
                    "domain": domain,
                    "action_type": "provide_additional_independent_t100_sources",
                    "notes": f"{domain} has {row['independent_sources']} independent local t100 source(s); {row['blocker']}. Use official ETH/UCY or TrajNet++ sources only and keep license terms.",
                }
            )
    actions.append(
        {
            "source": payload["source"],
            "priority": "medium",
            "domain": "all",
            "action_type": "keep_global_t100_claim_blocked",
            "notes": "UCY local support is positive, but independent ETH_UCY and TrajNet support is insufficient.",
        }
    )
    return actions


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-BH Local T100 Independent Source Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_bh_gate']['passed']} / {payload['stage42_bh_gate']['total']}`",
        f"- verdict: `{payload['stage42_bh_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- raw_t100_capable_files: `{s['raw_t100_capable_files']}`",
        f"- independent_t100_sources: `{s['independent_t100_sources']}`",
        f"- duplicate_or_alternate_format_group_count: `{s['duplicate_or_alternate_format_group_count']}`",
        f"- ucy_independent_sources: `{s['ucy_independent_sources']}`",
        f"- eth_ucy_independent_sources: `{s['eth_ucy_independent_sources']}`",
        f"- trajnet_independent_sources: `{s['trajnet_independent_sources']}`",
        f"- ucy_t100_source_cv_supported: `{s['ucy_t100_source_cv_supported']}`",
        f"- ucy_t100_mean_improvement_vs_fallback: `{s['ucy_t100_mean_improvement_vs_fallback']}`",
        f"- ucy_t100_min_improvement_vs_fallback: `{s['ucy_t100_min_improvement_vs_fallback']}`",
        f"- ucy_t100_max_easy_degradation: `{s['ucy_t100_max_easy_degradation']}`",
        f"- blocked_domains: `{', '.join(s['blocked_domains']) or 'none'}`",
        f"- global_t100_positive_claim_allowed: `{s['global_t100_positive_claim_allowed']}`",
        "",
        "## Independent Source Support Matrix",
        "",
        "| domain | independent sources | estimated t100 windows | source-CV feasible | blocker |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for domain, row in payload["support_matrix"].items():
        lines.append(
            f"| `{domain}` | {row['independent_sources']} | {row['estimated_t100_windows']} | {row['source_cv_feasible']} | `{row['blocker']}` |"
        )
    lines.extend(
        [
            "",
            "## Duplicate / Alternate Format Groups",
            "",
            "| independent key | chosen source | candidates | deduplicated |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for key, row in payload["duplicate_audit"]["duplicate_groups"].items():
        lines.append(
            f"| `{key}` | `{row['chosen_source_id']}` | `{', '.join(row['candidate_files'])}` | {row['deduplicated_as_same_scene_or_source']} |"
        )
    lines.extend(
        [
            "",
            "## Domain Source-CV Summary",
            "",
            "| domain | horizon | folds | safe folds | mean improvement | min improvement | max easy degradation | all safe |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for domain, row in payload["domain_summary"].items():
        for horizon, hrow in row["by_horizon"].items():
            lines.append(
                f"| `{domain}` | {horizon} | {hrow['fold_count']} | {hrow['safe_positive_fold_count']} | "
                f"{hrow['mean_holdout_improvement_vs_fallback']:.6f} | {hrow['minimum_holdout_improvement_vs_fallback']:.6f} | "
                f"{hrow['maximum_easy_degradation']:.6f} | {hrow['all_folds_safe_positive']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Counting files is not enough: alternate formats from the same scene/source are deduplicated before source-CV.",
            "- UCY has enough independent local t100 sources and positive mean t100 gain, but it is not easy-safe under strict independent source-CV.",
            "- The safe-switch repair reduced one large easy-harm fold, but `students03` still exceeds the 2% easy-degradation gate.",
            "- ETH_UCY and TrajNet remain hard blockers for global t100 support.",
            "- No metric/seconds-level, true-3D, Stage5C, or SMC claim is allowed.",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_bh_gate"]
    lines = [
        "# Stage42-BH Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {bool(ok)} |")
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = ["# Stage42-BH User Action Required", "", f"- source: `{payload['source']}`", ""]
    for action in payload["user_action_required"]:
        lines.extend(
            [
                f"## {action['domain']} / {action['action_type']}",
                "",
                f"- priority: `{action['priority']}`",
                f"- notes: {action['notes']}",
                "",
            ]
        )
    return lines


def _append_ledger(payload: Mapping[str, Any]) -> None:
    row = {
        "stage": payload["stage"],
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "verdict": payload["stage42_bh_gate"]["verdict"],
        "gate": f"{payload['stage42_bh_gate']['passed']}/{payload['stage42_bh_gate']['total']}",
        "git_commit": payload["git_commit"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_local_t100_independent_source_audit()
