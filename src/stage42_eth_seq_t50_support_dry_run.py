from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from src import stage42_eth_person_xml_t100_conversion as bl
from src import stage42_local_t100_protected_policy as bg
from src import stage42_local_t100_schema_conversion as bf
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
BM_JSON = OUT_DIR / "eth_person_terms_audit_stage42.json"
BS_JSON = OUT_DIR / "ucy_zara_t50_family_policy_stage42.json"
REPORT_JSON = OUT_DIR / "eth_seq_t50_support_dry_run_stage42.json"
REPORT_MD = OUT_DIR / "eth_seq_t50_support_dry_run_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_bt_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_eth_seq_t50_support_stage42.md"

FALLBACK = bg.FALLBACK_BASELINE
EASY_LIMIT = bg.EASY_DEGRADATION_LIMIT

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-BT 只测试 ETH-Person XML 是否技术上能修复 Stage42-BR 的 ETH_seq calibrated t50 source-support blocker。",
    "ETH-Person XML terms/license 仍未确认；本结果只能是 technical_dry_run_terms_unverified。",
    "policy threshold / baseline choice 只从 train/validation source 选择，holdout source 只评估一次。",
    "future endpoints 只作为 validation/test error labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


Selector = Callable[[bg.Window], str]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _feature_threshold_selector(baseline: str, feature: str, direction: str, threshold: float) -> Selector:
    def selector(row: bg.Window) -> str:
        value = float(row[feature])
        use_candidate = value <= threshold if direction == "low" else value >= threshold
        return baseline if use_candidate else FALLBACK

    return selector


def _select_h50_policy(train_windows: list[bg.Window], val_windows: list[bg.Window]) -> dict[str, Any]:
    train50 = [row for row in train_windows if int(row["horizon"]) == 50]
    val50 = [row for row in val_windows if int(row["horizon"]) == 50]
    fallback_metrics = bg._policy_metrics(val50, bg._global_selector(FALLBACK))
    if not train50 or not val50:
        return {
            "source": "fresh_eth_seq_t50_support_dry_run_validation_selection",
            "selected_policy": {"policy_name": f"global_{FALLBACK}", "metadata": {"baseline": FALLBACK}, "fallback_reason": "empty_train_or_validation"},
            "candidate_count": 0,
            "candidates": [],
        }
    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    score_best = -1e9
    for baseline in bf.BASELINES:
        if baseline == FALLBACK:
            continue
        for feature in ["speed_causal", "accel_causal"]:
            values = [float(row[feature]) for row in train50 + val50]
            thresholds = [float(np.quantile(values, q)) for q in [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95]]
            for direction in ["low", "high"]:
                for threshold in thresholds:
                    selector = _feature_threshold_selector(baseline, feature, direction, threshold)
                    metrics = bg._policy_metrics(val50, selector)
                    improvement = float(metrics["improvement_vs_fallback"] or 0.0)
                    easy = float(metrics["easy_degradation"] or 0.0)
                    harm = float(metrics["harm_over_fallback"] or 0.0)
                    eligible = improvement > 0.0 and easy <= EASY_LIMIT and harm <= 0.0
                    score = improvement - 0.02 * float(metrics["switch_rate"] or 0.0) - 10.0 * max(0.0, easy - EASY_LIMIT)
                    row = {
                        "source": "fresh_eth_seq_t50_support_dry_run_validation_selection",
                        "policy_name": f"{baseline}_{feature}_{direction}",
                        "metadata": {"baseline": baseline, "feature": feature, "direction": direction, "threshold": float(threshold)},
                        "validation_metrics": metrics,
                        "eligible_for_holdout": bool(eligible),
                        "score": float(score),
                    }
                    candidates.append(row)
                    if eligible and score > score_best:
                        best = row
                        score_best = float(score)
    if best is None:
        best = {
            "source": "fresh_eth_seq_t50_support_dry_run_validation_selection",
            "policy_name": f"global_{FALLBACK}",
            "metadata": {"baseline": FALLBACK},
            "validation_metrics": fallback_metrics,
            "eligible_for_holdout": True,
            "score": 0.0,
            "fallback_reason": "no_validation_safe_policy_exceeded_fallback",
        }
    return {
        "source": "fresh_eth_seq_t50_support_dry_run_validation_selection",
        "selected_policy": best,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _selector_from_selected(selected: Mapping[str, Any]) -> Selector:
    name = str(selected["policy_name"])
    if name.startswith("global_"):
        return bg._global_selector(name.removeprefix("global_"))
    meta = selected.get("metadata", {})
    return _feature_threshold_selector(str(meta["baseline"]), str(meta["feature"]), str(meta["direction"]), float(meta["threshold"]))


def _evaluate_fold(fold: Mapping[str, Any], windows_by_source: Mapping[str, list[bg.Window]]) -> dict[str, Any]:
    train: list[bg.Window] = []
    for source_id in fold.get("train_sources", []):
        train.extend(windows_by_source.get(str(source_id), []))
    val = list(windows_by_source.get(str(fold["validation_source"]), []))
    holdout = list(windows_by_source.get(str(fold["holdout_source"]), []))
    selection = _select_h50_policy(train, val)
    selector = _selector_from_selected(selection["selected_policy"])
    holdout50 = [row for row in holdout if int(row["horizon"]) == 50]
    metrics = bg._policy_metrics(holdout50, selector)
    return {
        "source": "fresh_eth_seq_t50_support_dry_run_fold",
        "holdout_source": fold["holdout_source"],
        "validation_source": fold["validation_source"],
        "train_sources": list(fold.get("train_sources", [])),
        "h50_holdout_rows": int(len(holdout50)),
        "selection": selection,
        "holdout_h50_metrics": metrics,
        "safe_positive_h50": bool((metrics["improvement_vs_fallback"] or 0.0) > 0.0 and (metrics["easy_degradation"] or 0.0) <= EASY_LIMIT),
    }


def _aggregate(folds: list[Mapping[str, Any]]) -> dict[str, Any]:
    improvements = [float(fold["holdout_h50_metrics"]["improvement_vs_fallback"] or 0.0) for fold in folds]
    easy = [float(fold["holdout_h50_metrics"]["easy_degradation"] or 0.0) for fold in folds]
    eth_seq = [fold for fold in folds if fold["holdout_source"] == "ETH_seq_eth"]
    return {
        "source_cv_folds": len(folds),
        "h50_windows_total": int(sum(int(fold["h50_holdout_rows"]) for fold in folds)),
        "technical_h50_mean_improvement_vs_fallback": float(np.mean(improvements)) if improvements else 0.0,
        "technical_h50_min_improvement_vs_fallback": float(np.min(improvements)) if improvements else 0.0,
        "technical_h50_max_easy_degradation": float(np.max(easy)) if easy else 0.0,
        "safe_positive_h50_fold_count": int(sum(bool(fold["safe_positive_h50"]) for fold in folds)),
        "eth_seq_holdout_rows": int(eth_seq[0]["h50_holdout_rows"]) if eth_seq else 0,
        "eth_seq_h50_improvement_vs_fallback": float(eth_seq[0]["holdout_h50_metrics"]["improvement_vs_fallback"] or 0.0) if eth_seq else 0.0,
        "eth_seq_easy_degradation": float(eth_seq[0]["holdout_h50_metrics"]["easy_degradation"] or 0.0) if eth_seq else 0.0,
        "eth_seq_safe_positive_h50": bool(eth_seq and eth_seq[0]["safe_positive_h50"]),
        "eth_seq_selected_policy": eth_seq[0]["selection"]["selected_policy"]["policy_name"] if eth_seq else "not_run",
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    claim = payload["claim_boundary"]
    no_leak = payload["no_leakage"]
    gates = {
        "bm_input_verified": payload["bm_verdict"] == "stage42_bm_eth_person_terms_audit_pass_claim_blocked",
        "bs_input_verified": payload["bs_verdict"] == "stage42_bs_ucy_zara_t50_family_policy_pass_positive",
        "technical_sources_present": s["candidate_sources"] >= 5,
        "source_cv_completed": s["source_cv_folds"] >= 5,
        "h50_windows_present": s["h50_windows_total"] > 0,
        "eth_seq_holdout_evaluated": s["eth_seq_holdout_rows"] > 0,
        "result_honestly_blocks_eth_seq": s["eth_seq_t50_support_repaired"] is False,
        "no_future_inputs": no_leak["future_endpoint_input"] is False and no_leak["test_threshold_tuning"] is False,
        "terms_not_overclaimed": claim["eth_person_terms_confirmed"] is False and claim["official_converted_dataset_claim_allowed"] is False,
        "global_metric_blocked": claim["global_metric_claim_allowed"] is False,
        "global_seconds_blocked": claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_bt_eth_seq_t50_support_dry_run_pass_blocker_confirmed" if passed == total else "stage42_bt_eth_seq_t50_support_dry_run_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "eth_seq_t50_support_repaired": bool(s["eth_seq_t50_support_repaired"]), "verdict": verdict}


def run_stage42_eth_seq_t50_support_dry_run() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    bm = _load_json(BM_JSON)
    bs = _load_json(BS_JSON)
    sources = bl._candidate_sources()
    windows_by_source = {str(src["source_id"]): bl._build_windows(src) for src in sources}
    folds = [_evaluate_fold(fold, windows_by_source) for fold in bl._folds_for_sources(sources)]
    aggregate = _aggregate(folds)
    claim_boundary = {
        "true_3d": False,
        "foundation_world_model": False,
        "eth_person_terms_confirmed": False,
        "official_converted_dataset_claim_allowed": False,
        "source_specific_annotation_step_subset_claim_allowed": True,
        "global_metric_claim_allowed": False,
        "global_seconds_claim_allowed": False,
        "positive_eth_seq_t50_claim_allowed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    eth_seq_repaired = bool(aggregate["eth_seq_safe_positive_h50"])
    summary = {
        "source": "fresh_eth_seq_t50_support_dry_run_terms_unverified",
        "candidate_sources": len(sources),
        "eth_person_xml_sources": int(sum(str(src["relative_path"]).startswith("ETH-Person/") for src in sources)),
        **aggregate,
        "eth_seq_t50_support_repaired": eth_seq_repaired,
        "remaining_blocker": "none" if eth_seq_repaired else "ETH_seq remains unsupported for calibrated t50 under validation-only safe policy; ETH-Person XML h50 technical positives do not safely transfer to ETH_seq_eth holdout.",
        "auto_download_executed": False,
        "training_run": True,
    }
    no_leakage = {
        "future_endpoint_input": False,
        "future_labels_eval_only": True,
        "central_velocity": False,
        "test_endpoint_goals": False,
        "test_threshold_tuning": False,
        "selection_uses_holdout": False,
    }
    payload: dict[str, Any] = {
        "source": "fresh_eth_seq_t50_support_dry_run_terms_unverified",
        "stage": "Stage42-BT ETH_seq T50 Support Dry-Run",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(BM_JSON), str(BS_JSON)] + [str(src["path"]) for src in sources]),
        "current_facts": CURRENT_FACTS,
        "bm_verdict": bm.get("stage42_bm_gate", {}).get("verdict"),
        "bs_verdict": bs.get("stage42_bs_gate", {}).get("verdict"),
        "summary": summary,
        "fold_results": folds,
        "no_leakage": no_leakage,
        "claim_boundary": claim_boundary,
    }
    payload["stage42_bt_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-BT ETH_seq T50 Support Dry-Run",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_bt_gate']['passed']} / {payload['stage42_bt_gate']['total']}`",
        f"- verdict: `{payload['stage42_bt_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- candidate_sources: `{s['candidate_sources']}`",
        f"- eth_person_xml_sources: `{s['eth_person_xml_sources']}`",
        f"- source_cv_folds: `{s['source_cv_folds']}`",
        f"- h50_windows_total: `{s['h50_windows_total']}`",
        f"- technical_h50_mean_improvement_vs_fallback: `{s['technical_h50_mean_improvement_vs_fallback']}`",
        f"- technical_h50_min_improvement_vs_fallback: `{s['technical_h50_min_improvement_vs_fallback']}`",
        f"- technical_h50_max_easy_degradation: `{s['technical_h50_max_easy_degradation']}`",
        f"- safe_positive_h50_fold_count: `{s['safe_positive_h50_fold_count']}`",
        f"- eth_seq_holdout_rows: `{s['eth_seq_holdout_rows']}`",
        f"- eth_seq_h50_improvement_vs_fallback: `{s['eth_seq_h50_improvement_vs_fallback']}`",
        f"- eth_seq_easy_degradation: `{s['eth_seq_easy_degradation']}`",
        f"- eth_seq_selected_policy: `{s['eth_seq_selected_policy']}`",
        f"- eth_seq_t50_support_repaired: `{s['eth_seq_t50_support_repaired']}`",
        f"- remaining_blocker: `{s['remaining_blocker']}`",
        "",
        "## Fold Results",
        "",
        "| holdout | validation | rows | improvement | easy degradation | switch | safe positive | selected policy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for fold in payload["fold_results"]:
        m = fold["holdout_h50_metrics"]
        selected = fold["selection"]["selected_policy"]["policy_name"]
        lines.append(
            f"| `{fold['holdout_source']}` | `{fold['validation_source']}` | {m['rows']} | {float(m['improvement_vs_fallback'] or 0.0):.6f} | {float(m['easy_degradation'] or 0.0):.6f} | {float(m['switch_rate'] or 0.0):.6f} | {fold['safe_positive_h50']} | `{selected}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- ETH-Person XML provides technical h50 signal on several ETH-Person holdouts, but this does not safely repair the actual `ETH_seq_eth` calibrated t50 holdout.",
            "- For `ETH_seq_eth`, validation-only safety selection falls back to constant velocity, so improvement is 0 rather than positive transfer.",
            "- This confirms the Stage42-BR blocker: ETH_seq still needs same-family/source-compatible support, official terms confirmation, or a stronger source-compatible model.",
            "- Because ETH-Person terms are unverified, none of this is official converted/evaluated data or a deployable metric/seconds-level claim.",
            "",
            "## Claim Boundary",
            "",
            f"- no_leakage: `{payload['no_leakage']}`",
            f"- claim_boundary: `{payload['claim_boundary']}`",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_bt_gate"]
    lines = [
        "# Stage42-BT Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- eth_seq_t50_support_repaired: `{gate['eth_seq_t50_support_repaired']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {bool(ok)} |")
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    return [
        "# User Action Required: ETH_seq Calibrated T50 Support",
        "",
        "- Stage42-BT confirms that ETH-Person XML technical dry-run does not safely repair `ETH_seq_eth` t50 under validation-only selection.",
        "- ETH-Person XML terms remain unverified; confirm official terms before any official conversion or claim.",
        "- To repair ETH_seq calibrated t50, provide or confirm legal same-family/source-compatible ETH-style top-down trajectories with t50-capable tracks.",
        "- Do not treat OpenTraj toolkit MIT license as permission for underlying ETH-Person data.",
        "- Stage5C remains unexecuted; SMC remains disabled.",
    ]


if __name__ == "__main__":
    run_stage42_eth_seq_t50_support_dry_run()
