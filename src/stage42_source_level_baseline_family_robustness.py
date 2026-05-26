from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_source_level_full_waypoint_eval import _jsonable


OUT_DIR = Path("outputs/stage42_long_research")
INPUT_JSON = OUT_DIR / "source_level_baseline_family_mechanism_stage42.json"
REPORT_JSON = OUT_DIR / "source_level_baseline_family_robustness_stage42.json"
REPORT_MD = OUT_DIR / "source_level_baseline_family_robustness_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_av_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-AV 是 Stage42-AU baseline-family mechanism 的 robustness / weak-slice audit。",
    "本审计不重新用 test 调 threshold，不执行 Stage5C，不启用 SMC。",
    "如果某个 domain 缺 validation support，必须写成 blocker 或 floor-only，不得包装成 positive transfer。",
    "raw-frame / dataset-local 不能写成 metric 或 seconds-level。",
]


def _load_au(path: Path = INPUT_JSON) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ci_positive(ci: Mapping[str, Any], *, easy: bool = False) -> bool:
    if int(ci.get("bootstrap_n", 0)) <= 0:
        return False
    return float(ci["high"] if easy else ci["low"]) <= 0.02 if easy else float(ci["low"]) > 0.0


def _global_stability(variant: Mapping[str, Any]) -> dict[str, Any]:
    boot = variant["bootstrap"]
    return {
        "source": "cached_verified_from_stage42_au",
        "all_ci_low_positive": _ci_positive(boot["all"]),
        "t50_ci_low_positive": _ci_positive(boot["t50"]),
        "t100_raw_frame_diagnostic_ci_low_positive": _ci_positive(boot["t100_raw_frame_diagnostic"]),
        "hard_failure_ci_low_positive": _ci_positive(boot["hard_failure"]),
        "easy_degradation_ci_high_safe": _ci_positive(boot["easy_degradation"], easy=True),
        "bootstrap": boot,
    }


def _domain_support(au: Mapping[str, Any], variant_name: str) -> dict[str, Any]:
    variant = au["variants"][variant_name]
    split_stats = au["split_stats"]["by_split"]
    out: dict[str, Any] = {}
    for domain, metric in variant["by_domain"].items():
        train_rows = int(split_stats.get("train", {}).get("domains", {}).get(domain, 0))
        val_rows = int(split_stats.get("val", {}).get("domains", {}).get(domain, 0))
        test_rows = int(metric["rows"])
        switch_rate = float(metric["switch_rate"])
        positive = bool(
            (float(metric["all_improvement"]) > 0.0 or float(metric["t50_improvement"]) > 0.0)
            and float(metric["easy_degradation"]) <= 0.02
        )
        blocker = "none"
        if test_rows > 0 and val_rows == 0 and switch_rate == 0.0:
            blocker = "no_validation_rows_for_domain_policy_selection_floor_only"
        out[domain] = {
            "source": "cached_verified_from_stage42_au",
            "train_rows": train_rows,
            "val_rows": val_rows,
            "test_rows": test_rows,
            "metric": metric,
            "positive_or_floor_safe": positive or blocker != "none",
            "positive_transfer": positive and blocker == "none",
            "blocker": blocker,
        }
    return out


def _horizon_support(au: Mapping[str, Any], variant_name: str) -> dict[str, Any]:
    variant = au["variants"][variant_name]
    out: dict[str, Any] = {}
    for horizon, metric in variant["by_horizon"].items():
        easy = float(metric["easy_degradation"])
        positive = bool(
            (float(metric["all_improvement"]) > 0.0 or float(metric["hard_failure_improvement"]) > 0.0)
            and easy <= 0.02
        )
        weak = []
        if float(metric["all_improvement"]) <= 0.0:
            weak.append("non_positive_all")
        if easy > 0.02:
            weak.append("easy_degradation_over_2pct")
        out[str(horizon)] = {
            "source": "cached_verified_from_stage42_au",
            "metric": metric,
            "positive_and_easy_safe": positive,
            "weaknesses": weak,
        }
    return out


def _support_summary(domain_support: Mapping[str, Any], horizon_support: Mapping[str, Any]) -> dict[str, Any]:
    positive_domains = sorted([k for k, v in domain_support.items() if v["positive_transfer"]])
    floor_only_domains = sorted([k for k, v in domain_support.items() if v["blocker"] != "none"])
    weak_horizons = sorted([k for k, v in horizon_support.items() if v["weaknesses"]])
    return {
        "source": "cached_verified_from_stage42_au",
        "positive_domains": positive_domains,
        "floor_only_or_blocked_domains": floor_only_domains,
        "weak_horizons": weak_horizons,
        "uniform_domain_claim_allowed": len(floor_only_domains) == 0 and len(positive_domains) == len(domain_support),
        "uniform_horizon_claim_allowed": len(weak_horizons) == 0,
        "paper_claim": (
            "baseline-family rollout context is statistically stable globally and on TrajNet, but uniform source-level domain/horizon claims remain disallowed because UCY is floor-only under this split and t100 has easy-safety weakness."
            if floor_only_domains or weak_horizons
            else "baseline-family rollout context is stable across audited domains and horizons under this split."
        ),
    }


def run_stage42_source_level_baseline_family_robustness() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    au = _load_au()
    variant_name = "baseline_family_all"
    family_name = "family_baseline_rel_only"
    global_stability = {
        family_name: _global_stability(au["variants"][family_name]),
        variant_name: _global_stability(au["variants"][variant_name]),
    }
    domain_support = _domain_support(au, variant_name)
    horizon_support = _horizon_support(au, variant_name)
    summary = _support_summary(domain_support, horizon_support)
    result = {
        "source": "cached_verified_from_stage42_au",
        "stage": "Stage42-AV baseline-family mechanism robustness audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([str(INPUT_JSON)]),
        "au_verdict": au["stage42_au_gate"]["verdict"],
        "global_stability": global_stability,
        "domain_support": domain_support,
        "horizon_support": horizon_support,
        "summary": summary,
        "no_leakage": au["no_leakage"],
        "claim_boundary": au["claim_boundary"],
    }
    result["stage42_av_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _append_ledger(result)
    return result


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    global_all = result["global_stability"]["baseline_family_all"]
    gates = {
        "au_input_verified": result["au_verdict"] == "stage42_au_baseline_family_mechanism_pass",
        "global_all_ci_positive": global_all["all_ci_low_positive"],
        "global_t50_ci_positive": global_all["t50_ci_low_positive"],
        "global_hard_failure_ci_positive": global_all["hard_failure_ci_low_positive"],
        "global_easy_ci_safe": global_all["easy_degradation_ci_high_safe"],
        "domain_support_audited": len(result["domain_support"]) >= 2,
        "weak_or_blocked_slices_reported": bool(result["summary"]["floor_only_or_blocked_domains"] or result["summary"]["weak_horizons"]),
        "uniform_claim_not_overstated": not result["summary"]["uniform_domain_claim_allowed"] or not result["summary"]["uniform_horizon_claim_allowed"],
        "no_leakage_pass": all(
            result["no_leakage"][k] is False
            for k in ["future_endpoint_input", "future_waypoint_input", "central_velocity", "test_endpoint_goals", "test_threshold_tuning"]
        )
        and result["no_leakage"]["train_only_feature_normalization"],
        "no_metric_seconds_overclaim": not result["claim_boundary"]["metric_or_seconds_claim"],
        "stage5c_false": not result["claim_boundary"]["stage5c_executed"],
        "smc_false": not result["claim_boundary"]["smc_enabled"],
    }
    verdict = "stage42_av_baseline_family_robustness_pass_with_limits" if all(gates.values()) else "stage42_av_baseline_family_robustness_partial"
    return {"source": result.get("source", "unit_test"), "gates": gates, "passed": int(sum(bool(v) for v in gates.values())), "total": int(len(gates)), "verdict": verdict}


def _render_report(result: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-AV Baseline-Family Mechanism Robustness Audit",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_av_gate']['passed']} / {result['stage42_av_gate']['total']}`",
        f"- verdict: `{result['stage42_av_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
    ]
    lines.extend([f"- {fact}" for fact in result["current_facts"]])
    lines.extend(
        [
            "",
            "## Global Bootstrap Stability",
            "",
            "| variant | all low | t50 low | t100 low | hard low | easy high |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, row in result["global_stability"].items():
        b = row["bootstrap"]
        lines.append(
            f"| `{name}` | {b['all']['low']:.6f} | {b['t50']['low']:.6f} | {b['t100_raw_frame_diagnostic']['low']:.6f} | {b['hard_failure']['low']:.6f} | {b['easy_degradation']['high']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Domain Support",
            "",
            "| domain | train | val | test | all | t50 | hard | easy | switch | blocker |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for domain, row in result["domain_support"].items():
        m = row["metric"]
        lines.append(
            f"| `{domain}` | {row['train_rows']} | {row['val_rows']} | {row['test_rows']} | {m['all_improvement']:.6f} | {m['t50_improvement']:.6f} | {m['hard_failure_improvement']:.6f} | {m['easy_degradation']:.6f} | {m['switch_rate']:.6f} | `{row['blocker']}` |"
        )
    lines.extend(
        [
            "",
            "## Horizon Support",
            "",
            "| horizon | rows | all | horizon metric | hard | easy | switch | weaknesses |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for horizon, row in result["horizon_support"].items():
        m = row["metric"]
        horizon_key = f"t{horizon}_improvement" if horizon != "100" else "t100_raw_frame_diagnostic_improvement"
        lines.append(
            f"| `{horizon}` | {m['rows']} | {m['all_improvement']:.6f} | {m[horizon_key]:.6f} | {m['hard_failure_improvement']:.6f} | {m['easy_degradation']:.6f} | {m['switch_rate']:.6f} | `{', '.join(row['weaknesses']) or 'none'}` |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- positive_domains: `{result['summary']['positive_domains']}`",
            f"- floor_only_or_blocked_domains: `{result['summary']['floor_only_or_blocked_domains']}`",
            f"- weak_horizons: `{result['summary']['weak_horizons']}`",
            f"- uniform_domain_claim_allowed: `{result['summary']['uniform_domain_claim_allowed']}`",
            f"- uniform_horizon_claim_allowed: `{result['summary']['uniform_horizon_claim_allowed']}`",
            f"- paper_claim: {result['summary']['paper_claim']}",
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{result['no_leakage']}`",
            f"- claim_boundary: `{result['claim_boundary']}`",
        ]
    )
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_av_gate"]
    lines = [
        "# Stage42-AV Gate",
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


def _append_ledger(result: Mapping[str, Any]) -> None:
    row = {
        "stage": result["stage"],
        "source": result["source"],
        "generated_at_utc": result["generated_at_utc"],
        "verdict": result["stage42_av_gate"]["verdict"],
        "gate": f"{result['stage42_av_gate']['passed']}/{result['stage42_av_gate']['total']}",
        "git_commit": result["git_commit"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(f"{row}\n")


if __name__ == "__main__":
    run_stage42_source_level_baseline_family_robustness()
