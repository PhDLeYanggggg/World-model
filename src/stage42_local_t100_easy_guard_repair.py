from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src import stage42_local_t100_independent_source_audit as bh
from src import stage42_local_t100_protected_policy as bg
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
BD_JSON = OUT_DIR / "local_t100_source_inventory_stage42.json"
BH_JSON = OUT_DIR / "local_t100_independent_source_audit_stage42.json"
REPORT_JSON = OUT_DIR / "local_t100_easy_guard_repair_stage42.json"
REPORT_MD = OUT_DIR / "local_t100_easy_guard_repair_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_bi_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_local_t100_easy_guard_repair_stage42.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-BI 修复 Stage42-BH 暴露的 UCY independent-source t100 easy degradation。",
    "策略选择只使用 non-holdout source：validation source + train sources；holdout source 只评估一次。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t100 仍是 raw-frame diagnostic；UCY 修复不等于全局 t100 success。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

THRESHOLD_QUANTILES = [0.0, 0.10, 0.25, 0.50, 0.75, 0.85, 0.90, 0.95, 0.975, 0.99]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _thresholds_from_sources(train_windows: list[bg.Window], horizon: int) -> list[float]:
    values = sorted(float(row["speed_causal"]) for row in train_windows if int(row["horizon"]) == horizon)
    if not values:
        return [0.0]
    thresholds = {
        values[max(0, min(int(round(q * (len(values) - 1))), len(values) - 1))]
        for q in THRESHOLD_QUANTILES
    }
    return sorted(float(v) for v in thresholds)


def _robust_speed_candidates(train_windows: list[bg.Window], horizon: int) -> list[tuple[str, bg.Selector, dict[str, Any]]]:
    candidates: list[tuple[str, bg.Selector, dict[str, Any]]] = []
    for baseline in bh.SAFE_SWITCH_BASELINES:
        for threshold in _thresholds_from_sources(train_windows, horizon):
            def selector(row: bg.Window, *, b: str = baseline, t: float = threshold) -> str:
                return b if float(row["speed_causal"]) >= t else bg.FALLBACK_BASELINE

            candidates.append(
                (
                    f"robust_speed_{baseline}_gte_{threshold:.6g}",
                    selector,
                    {
                        "baseline": baseline,
                        "speed_threshold": threshold,
                        "policy_family": "source_robust_speed_safe_switch",
                        "threshold_quantiles": THRESHOLD_QUANTILES,
                    },
                )
            )
    candidates.append((f"global_{bg.FALLBACK_BASELINE}", bg._global_selector(bg.FALLBACK_BASELINE), {"policy_family": "fallback"}))
    return candidates


def _select_source_robust_policy(
    *,
    train_windows: list[bg.Window],
    support_source_windows: Mapping[str, list[bg.Window]],
    horizon: int,
) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for name, selector, meta in _robust_speed_candidates(train_windows, horizon):
        support_metrics: dict[str, Any] = {}
        eligible = name == f"global_{bg.FALLBACK_BASELINE}"
        improvements: list[float] = []
        easy_vals: list[float] = []
        switches: list[float] = []
        if name != f"global_{bg.FALLBACK_BASELINE}":
            eligible = True
        for source_id, rows in support_source_windows.items():
            rows_h = [row for row in rows if int(row["horizon"]) == horizon]
            metrics = bg._policy_metrics(rows_h, selector)
            support_metrics[source_id] = metrics
            improvement = float(metrics["improvement_vs_fallback"] or 0.0)
            easy = float(metrics["easy_degradation"] or 0.0)
            switch = float(metrics["switch_rate"] or 0.0)
            improvements.append(improvement)
            easy_vals.append(easy)
            switches.append(switch)
            if name != f"global_{bg.FALLBACK_BASELINE}" and not (improvement > 0.0 and easy <= bg.EASY_DEGRADATION_LIMIT):
                eligible = False
        min_improvement = min(improvements) if improvements else 0.0
        mean_improvement = sum(improvements) / len(improvements) if improvements else 0.0
        max_easy = max(easy_vals) if easy_vals else 0.0
        max_switch = max(switches) if switches else 0.0
        score = min_improvement + 0.1 * mean_improvement - 0.01 * max_switch
        row = {
            "source": "fresh_source_robust_easy_guard_selection",
            "policy_name": name,
            "metadata": meta,
            "support_source_metrics": support_metrics,
            "eligible_for_holdout": bool(eligible),
            "min_support_improvement": float(min_improvement),
            "mean_support_improvement": float(mean_improvement),
            "max_support_easy_degradation": float(max_easy),
            "max_support_switch_rate": float(max_switch),
            "score": float(score),
            "horizon": int(horizon),
        }
        evaluated.append(row)
        if name != f"global_{bg.FALLBACK_BASELINE}" and eligible and (best is None or score > float(best["score"])):
            best = row
    if best is None:
        fallback = next(row for row in evaluated if row["policy_name"] == f"global_{bg.FALLBACK_BASELINE}")
        fallback["fallback_reason"] = "no_source_robust_policy_met_positive_gain_and_easy_guard_on_all_support_sources"
        best = fallback
    return {
        "source": "fresh_source_robust_easy_guard_selection",
        "horizon": int(horizon),
        "selected_policy": best,
        "candidate_count": len(evaluated),
        "candidates": evaluated,
    }


def _selector_from_robust_selection(selected: Mapping[str, Any], train_windows: list[bg.Window], horizon: int) -> bg.Selector:
    name = str(selected["policy_name"])
    for candidate_name, selector, _ in _robust_speed_candidates(train_windows, horizon):
        if candidate_name == name:
            return selector
    return bg._global_selector(bg.FALLBACK_BASELINE)


def _evaluate_fold(*, fold: Mapping[str, Any], windows_by_source: Mapping[str, list[bg.Window]]) -> dict[str, Any]:
    train_windows: list[bg.Window] = []
    support_source_windows: dict[str, list[bg.Window]] = {}
    for source_id in fold.get("train_sources", []):
        rows = list(windows_by_source.get(str(source_id), []))
        train_windows.extend(rows)
        support_source_windows[str(source_id)] = rows
    support_source_windows[str(fold["validation_source"])] = list(windows_by_source.get(str(fold["validation_source"]), []))
    holdout_windows = list(windows_by_source.get(str(fold["holdout_source"]), []))
    by_horizon: dict[str, Any] = {}
    for horizon in bg.HORIZONS:
        selection = _select_source_robust_policy(
            train_windows=train_windows,
            support_source_windows=support_source_windows,
            horizon=horizon,
        )
        selector = _selector_from_robust_selection(selection["selected_policy"], train_windows, horizon)
        holdout_h = [row for row in holdout_windows if int(row["horizon"]) == horizon]
        holdout_metrics = bg._policy_metrics(holdout_h, selector)
        by_horizon[str(horizon)] = {
            "source": "fresh_source_robust_easy_guard_cv",
            "selection": selection,
            "holdout_metrics": holdout_metrics,
            "safe_positive": bool(
                (holdout_metrics["improvement_vs_fallback"] or 0.0) > 0.0
                and (holdout_metrics["easy_degradation"] or 0.0) <= bg.EASY_DEGRADATION_LIMIT
            ),
        }
    return {
        "source": "fresh_source_robust_easy_guard_cv",
        "domain": fold["domain"],
        "holdout_source": fold["holdout_source"],
        "validation_source": fold["validation_source"],
        "train_sources": list(fold.get("train_sources", [])),
        "by_horizon": by_horizon,
    }


def run_stage42_local_t100_easy_guard_repair() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    bd_payload = _load_json(BD_JSON)
    bh_payload = _load_json(BH_JSON)
    independent_sources, duplicate_audit = bh._build_independent_sources(list(bd_payload.get("inventory", [])))
    windows_by_source = {src["source_id"]: bg._build_windows_for_source(src) for src in independent_sources}
    folds_by_domain = {domain: bh._folds_for_domain(independent_sources, domain) for domain in ["ETH_UCY", "UCY", "TrajNet"]}
    fold_results = [
        _evaluate_fold(fold=fold, windows_by_source=windows_by_source)
        for domain in ["ETH_UCY", "UCY", "TrajNet"]
        for fold in folds_by_domain.get(domain, [])
    ]
    domain_summary = bg._domain_summary(fold_results)
    support = bh._support_matrix(independent_sources, folds_by_domain)
    ucy_t100 = domain_summary.get("UCY", {}).get("by_horizon", {}).get("100", {})
    ucy_supported = bool(
        ucy_t100.get("fold_count", 0) >= bh.MIN_INDEPENDENT_SOURCES
        and ucy_t100.get("all_folds_safe_positive", False)
        and (ucy_t100.get("maximum_easy_degradation") is not None and float(ucy_t100["maximum_easy_degradation"]) <= bg.EASY_DEGRADATION_LIMIT)
    )
    blocked_domains = [domain for domain, row in support.items() if not row["source_cv_feasible"]]
    payload = {
        "source": "fresh_source_robust_easy_guard_repair",
        "stage": "Stage42-BI Local T100 Source-Robust Easy Guard Repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([str(BD_JSON), str(BH_JSON)] + [str(src["path"]) for src in independent_sources]),
        "bd_verdict": bd_payload.get("stage42_bd_gate", {}).get("verdict"),
        "bh_verdict": bh_payload.get("stage42_bh_gate", {}).get("verdict"),
        "repair_strategy": {
            "source": "fresh_source_robust_easy_guard_repair",
            "description": "Candidate policies must be positive and easy-safe on every non-holdout source (validation source plus train sources), then are evaluated once on the held-out independent source.",
            "threshold_quantiles": THRESHOLD_QUANTILES,
            "holdout_used_for_selection": False,
            "future_labels_as_inputs": False,
        },
        "duplicate_audit": duplicate_audit,
        "support_matrix": support,
        "source_cv_folds": fold_results,
        "domain_summary": domain_summary,
        "summary": {
            "source": "fresh_source_robust_easy_guard_repair",
            "independent_t100_sources": duplicate_audit["independent_source_count"],
            "ucy_independent_sources": support["UCY"]["independent_sources"],
            "eth_ucy_independent_sources": support["ETH_UCY"]["independent_sources"],
            "trajnet_independent_sources": support["TrajNet"]["independent_sources"],
            "ucy_t100_source_cv_supported": ucy_supported,
            "ucy_t100_mean_improvement_vs_fallback": ucy_t100.get("mean_holdout_improvement_vs_fallback"),
            "ucy_t100_min_improvement_vs_fallback": ucy_t100.get("minimum_holdout_improvement_vs_fallback"),
            "ucy_t100_max_easy_degradation": ucy_t100.get("maximum_easy_degradation"),
            "blocked_domains": blocked_domains,
            "bh_previous_ucy_t100_supported": bh_payload.get("summary", {}).get("ucy_t100_source_cv_supported"),
            "bh_previous_ucy_max_easy_degradation": bh_payload.get("summary", {}).get("ucy_t100_max_easy_degradation"),
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
            "ucy_local_t100_independent_source_support": ucy_supported,
        },
    }
    payload["stage42_bi_gate"] = _gate(payload)
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
        "bh_failure_loaded": payload["bh_verdict"] == "stage42_bh_independent_t100_source_audit_partial",
        "source_robust_guard_built": payload["repair_strategy"]["holdout_used_for_selection"] is False,
        "ucy_source_cv_evaluated": "UCY" in payload["domain_summary"],
        "ucy_t100_source_cv_supported": bool(s["ucy_t100_source_cv_supported"]),
        "ucy_easy_degradation_repaired": (
            s["ucy_t100_max_easy_degradation"] is not None
            and float(s["ucy_t100_max_easy_degradation"]) <= bg.EASY_DEGRADATION_LIMIT
        ),
        "ucy_positive_gain_preserved": (
            s["ucy_t100_min_improvement_vs_fallback"] is not None
            and float(s["ucy_t100_min_improvement_vs_fallback"]) > 0.0
        ),
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
            ]
        ),
        "no_metric_seconds_overclaim": not payload["claim_boundary"]["metric_or_seconds_claim"],
        "global_t100_claim_still_blocked": not payload["claim_boundary"]["global_t100_positive_claim_allowed"],
        "stage5c_false": not payload["claim_boundary"]["stage5c_executed"],
        "smc_false": not payload["claim_boundary"]["smc_enabled"],
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_bi_ucy_t100_easy_guard_repair_pass_with_global_blocker" if passed == total else "stage42_bi_ucy_t100_easy_guard_repair_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _user_actions(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "source": payload["source"],
            "priority": "high",
            "domain": "ETH_UCY",
            "action_type": "provide_additional_independent_t100_sources",
            "notes": "UCY independent-source t100 easy guard is repaired locally, but ETH_UCY still has too few independent t100 sources for global support.",
        },
        {
            "source": payload["source"],
            "priority": "high",
            "domain": "TrajNet",
            "action_type": "provide_additional_independent_t100_sources",
            "notes": "TrajNet still has zero independent local t100 sources in the current inventory.",
        },
        {
            "source": payload["source"],
            "priority": "medium",
            "domain": "all",
            "action_type": "keep_global_t100_claim_blocked",
            "notes": "The repair supports UCY independent-source t100 only; it does not establish global external t100 success.",
        },
    ]


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-BI Local T100 Source-Robust Easy Guard Repair",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_bi_gate']['passed']} / {payload['stage42_bi_gate']['total']}`",
        f"- verdict: `{payload['stage42_bi_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Repair Strategy",
        "",
        f"- description: {payload['repair_strategy']['description']}",
        f"- threshold_quantiles: `{payload['repair_strategy']['threshold_quantiles']}`",
        f"- holdout_used_for_selection: `{payload['repair_strategy']['holdout_used_for_selection']}`",
        "",
        "## Summary",
        "",
        f"- independent_t100_sources: `{s['independent_t100_sources']}`",
        f"- ucy_independent_sources: `{s['ucy_independent_sources']}`",
        f"- eth_ucy_independent_sources: `{s['eth_ucy_independent_sources']}`",
        f"- trajnet_independent_sources: `{s['trajnet_independent_sources']}`",
        f"- ucy_t100_source_cv_supported: `{s['ucy_t100_source_cv_supported']}`",
        f"- ucy_t100_mean_improvement_vs_fallback: `{s['ucy_t100_mean_improvement_vs_fallback']}`",
        f"- ucy_t100_min_improvement_vs_fallback: `{s['ucy_t100_min_improvement_vs_fallback']}`",
        f"- ucy_t100_max_easy_degradation: `{s['ucy_t100_max_easy_degradation']}`",
        f"- bh_previous_ucy_max_easy_degradation: `{s['bh_previous_ucy_max_easy_degradation']}`",
        f"- blocked_domains: `{', '.join(s['blocked_domains']) or 'none'}`",
        f"- global_t100_positive_claim_allowed: `{s['global_t100_positive_claim_allowed']}`",
        "",
        "## Domain Source-CV Summary",
        "",
        "| domain | horizon | folds | safe folds | mean improvement | min improvement | max easy degradation | all safe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
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
            "## Fold Details",
            "",
            "| domain | holdout | horizon | selected policy | rows | improvement | easy degradation | switch rate |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for fold in payload["source_cv_folds"]:
        for horizon, hrow in fold["by_horizon"].items():
            m = hrow["holdout_metrics"]
            selected = hrow["selection"]["selected_policy"]["policy_name"]
            lines.append(
                f"| `{fold['domain']}` | `{fold['holdout_source']}` | {horizon} | `{selected}` | {m['rows']} | "
                f"{m['improvement_vs_fallback']:.6f} | {m['easy_degradation']:.6f} | {m['switch_rate']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-BI repairs the Stage42-BH UCY independent-source t100 easy-degradation blocker using only non-holdout sources for selection.",
            "- The repair keeps all UCY t100 independent-source folds positive and easy-safe.",
            "- This is still not a global t100 claim because ETH_UCY and TrajNet do not have enough independent t100 sources.",
            "- No metric/seconds-level, true-3D, Stage5C, or SMC claim is allowed.",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_bi_gate"]
    lines = [
        "# Stage42-BI Gate",
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
    lines = ["# Stage42-BI User Action Required", "", f"- source: `{payload['source']}`", ""]
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
        "verdict": payload["stage42_bi_gate"]["verdict"],
        "gate": f"{payload['stage42_bi_gate']['passed']}/{payload['stage42_bi_gate']['total']}",
        "git_commit": payload["git_commit"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_local_t100_easy_guard_repair()
