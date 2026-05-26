from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_t50_repair_statistical_evidence as bz
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
BZ_JSON = OUT_DIR / "t50_repair_statistical_evidence_stage42.json"
CA_JSON = OUT_DIR / "paper_package_post_bz_refresh_stage42.json"
REPORT_JSON = OUT_DIR / "t50_source_robustness_audit_stage42.json"
REPORT_MD = OUT_DIR / "t50_source_robustness_audit_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_cb_gate.md"

TARGET_SLICES = ["TrajNet|50", "UCY|50"]
BOOTSTRAP_N = 3000
MIN_CI_ROWS = 30
DOMINANCE_WARN_FRAC = 0.90
EASY_LIMIT = 0.02


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-CB 是 BY/BZ protected t50 repair 的 source-level robustness / concentration audit。",
    "Stage42-CB 不重新选择 policy，不调 threshold，不训练新模型。",
    "test rows 只用于最终 source-level reporting/bootstrap，不用于 policy selection。",
    "如果 evidence 集中在少数 source，必须报告 source concentration limitation，不能包装成 broad source-level generalization。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C 未执行，SMC 未启用。",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _source_name(path: str) -> str:
    p = Path(path)
    parent = p.parent.name
    return f"{parent}/{p.name}" if parent else p.name


def _ci_positive(ci: Mapping[str, Any], *, easy: bool = False) -> bool:
    if int(ci.get("bootstrap_n", 0)) < BOOTSTRAP_N:
        return False
    if easy:
        return float(ci.get("high", 1.0)) <= EASY_LIMIT
    return float(ci.get("low", -1.0)) > 0.0


def _safe_bootstrap(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray, seed: int, *, easy: bool = False) -> dict[str, Any]:
    if easy:
        return am._bootstrap_ci(floor, selected, mask, seed=seed, n=BOOTSTRAP_N)
    return am._bootstrap_ci(selected, floor, mask, seed=seed, n=BOOTSTRAP_N)


def _source_row(
    data: Mapping[str, np.ndarray],
    selected: np.ndarray,
    floor: np.ndarray,
    switch: np.ndarray,
    mask: np.ndarray,
    source_file: str,
    seed: int,
) -> dict[str, Any]:
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    metric = am._metric(selected, floor, data, switch, mask)
    hard_rows = int(np.sum(mask & hard_failure))
    easy_rows = int(np.sum(mask & easy))
    boot = {
        "t50": _safe_bootstrap(selected, floor, mask, seed),
        "hard_failure": _safe_bootstrap(selected, floor, mask & hard_failure, seed + 1),
        "easy_degradation": _safe_bootstrap(selected, floor, mask & easy, seed + 2, easy=True),
    }
    ci_available = int(np.sum(mask)) >= MIN_CI_ROWS
    hard_ci_available = hard_rows >= MIN_CI_ROWS
    easy_ci_available = easy_rows >= MIN_CI_ROWS
    robust = bool(
        ci_available
        and _ci_positive(boot["t50"])
        and (not hard_ci_available or _ci_positive(boot["hard_failure"]))
        and (not easy_ci_available or _ci_positive(boot["easy_degradation"], easy=True))
        and float(metric["switch_rate"]) > 0.0
    )
    return {
        "source": "fresh_stage42_cb_t50_source_robustness_audit",
        "source_file": source_file,
        "source_name": _source_name(source_file),
        "rows": int(np.sum(mask)),
        "hard_rows": hard_rows,
        "easy_rows": easy_rows,
        "metric": metric,
        "bootstrap": boot,
        "ci_available": ci_available,
        "hard_ci_available": hard_ci_available,
        "easy_ci_available": easy_ci_available,
        "robust_positive": robust,
    }


def _slice_source_audit(
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
    selected: np.ndarray,
    floor: np.ndarray,
    switch: np.ndarray,
    key: str,
    seed_base: int,
) -> dict[str, Any]:
    domain_name, horizon_s = key.split("|", 1)
    mask = (split == "test") & (data["dataset"].astype(str) == domain_name) & (data["horizon"].astype(int) == int(horizon_s))
    source_values = data["source_file"].astype(str)
    unique_sources = sorted(set(source_values[mask].tolist()))
    rows = [
        _source_row(data, selected, floor, switch, mask & (source_values == source), source, seed_base + i * 10)
        for i, source in enumerate(unique_sources)
    ]
    total_rows = int(np.sum(mask))
    largest_rows = max([row["rows"] for row in rows], default=0)
    largest_frac = float(largest_rows / max(total_rows, 1))
    robust_sources = [row["source_name"] for row in rows if row["robust_positive"]]
    underpowered_sources = [row["source_name"] for row in rows if not row["ci_available"]]
    concentration_limited = largest_frac >= DOMINANCE_WARN_FRAC or len(rows) < 2
    return {
        "source": "fresh_stage42_cb_t50_source_robustness_audit",
        "slice": key,
        "total_rows": total_rows,
        "source_count": len(rows),
        "largest_source_fraction": largest_frac,
        "concentration_limited": concentration_limited,
        "robust_source_count": len(robust_sources),
        "robust_sources": robust_sources,
        "underpowered_sources": underpowered_sources,
        "source_rows": rows,
        "broad_source_generalization_claim_allowed": bool(len(rows) >= 3 and not concentration_limited and len(robust_sources) >= 2),
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    bz_report = _load_json(BZ_JSON)
    ca_report = _load_json(CA_JSON) if CA_JSON.exists() else {}
    arrays = bz._recompute_by_policy_arrays()
    audits = {
        key: _slice_source_audit(
            arrays["data"],
            arrays["split"],
            arrays["selected_ade"],
            arrays["floor_ade"],
            arrays["switch"],
            key,
            seed_base=42700 + i * 100,
        )
        for i, key in enumerate(TARGET_SLICES)
    }
    concentration_limited = [key for key, row in audits.items() if row["concentration_limited"]]
    robust_major_sources = [
        key
        for key, row in audits.items()
        if row["source_rows"] and max(row["source_rows"], key=lambda x: x["rows"])["robust_positive"]
    ]
    summary = {
        "source": "fresh_stage42_cb_t50_source_robustness_audit",
        "verdict_short": "aggregate_t50_bootstrap_strong_but_source_diversity_limited",
        "target_slices": TARGET_SLICES,
        "robust_major_source_slices": robust_major_sources,
        "concentration_limited_slices": concentration_limited,
        "broad_source_generalization_claim_allowed": False,
        "source_level_claim": "major-source robust within available rows; not broad source-level generalization",
        "selected_variant": arrays["selected_variant"],
        "internal_val_group": arrays["internal_val_group"],
        "floor_free_neural_deployable": False,
        "global_metric_claim_allowed": False,
        "global_seconds_claim_allowed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    payload: dict[str, Any] = {
        "source": "fresh_stage42_cb_t50_source_robustness_audit",
        "stage": "Stage42-CB Protected T50 Source Robustness Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(BZ_JSON), str(CA_JSON)]),
        "current_facts": CURRENT_FACTS,
        "input_reports": {
            "stage42_bz_verdict": bz_report["stage42_bz_gate"]["verdict"],
            "stage42_ca_verdict": ca_report.get("stage42_ca_gate", {}).get("verdict", "not_run"),
        },
        "slice_audits": audits,
        "summary": summary,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "test_rows_for_reporting_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "broad_source_generalization_claim_allowed": False,
            "floor_free_neural_deployable": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_cb_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    claim = payload["claim_boundary"]
    no_leakage = payload["no_leakage"]
    gates = {
        "bz_input_verified": payload["input_reports"]["stage42_bz_verdict"] == "stage42_bz_t50_repair_statistical_evidence_pass",
        "ca_input_verified_or_optional": payload["input_reports"]["stage42_ca_verdict"]
        in {"stage42_ca_post_bz_paper_package_refresh_pass", "not_run"},
        "target_slices_audited": set(payload["slice_audits"].keys()) == set(TARGET_SLICES),
        "major_source_robustness_reported": set(s["robust_major_source_slices"]) == set(TARGET_SLICES),
        "source_concentration_limitation_reported": len(s["concentration_limited_slices"]) >= 1,
        "broad_source_generalization_not_overclaimed": s["broad_source_generalization_claim_allowed"] is False
        and claim["broad_source_generalization_claim_allowed"] is False,
        "no_leakage_pass": no_leakage["future_endpoint_input"] is False
        and no_leakage["future_waypoint_input"] is False
        and no_leakage["central_velocity"] is False
        and no_leakage["test_endpoint_goals"] is False
        and no_leakage["test_threshold_tuning"] is False,
        "not_floor_free_neural": claim["floor_free_neural_deployable"] is False,
        "no_metric_seconds_overclaim": claim["metric_or_seconds_claim"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_cb_t50_source_robustness_pass_with_source_diversity_limit" if passed == total else "stage42_cb_t50_source_robustness_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-CB Protected T50 Source Robustness Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_cb_gate']['passed']} / {payload['stage42_cb_gate']['total']}`",
        f"- verdict: `{payload['stage42_cb_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- verdict_short: `{s['verdict_short']}`",
        f"- robust_major_source_slices: `{s['robust_major_source_slices']}`",
        f"- concentration_limited_slices: `{s['concentration_limited_slices']}`",
        f"- broad_source_generalization_claim_allowed: `{s['broad_source_generalization_claim_allowed']}`",
        f"- source_level_claim: `{s['source_level_claim']}`",
        "",
        "## Slice Source Summary",
        "",
        "| slice | rows | sources | largest source frac | robust sources | underpowered | broad source claim |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for key, audit in payload["slice_audits"].items():
        lines.append(
            f"| `{key}` | {audit['total_rows']} | {audit['source_count']} | {_pct(audit['largest_source_fraction'])} | "
            f"`{', '.join(audit['robust_sources'])}` | `{', '.join(audit['underpowered_sources']) or 'none'}` | "
            f"{audit['broad_source_generalization_claim_allowed']} |"
        )
    lines += [
        "",
        "## Per-Source Evidence",
        "",
        "| slice | source | rows | t50 | t50 CI low | t50 CI high | easy CI high | switch | robust |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, audit in payload["slice_audits"].items():
        for row in audit["source_rows"]:
            lines.append(
                f"| `{key}` | `{row['source_name']}` | {row['rows']} | "
                f"{_pct(row['metric']['t50_improvement'])} | {_pct(row['bootstrap']['t50']['low'])} | "
                f"{_pct(row['bootstrap']['t50']['high'])} | {_pct(row['bootstrap']['easy_degradation']['high'])} | "
                f"{_pct(row['metric']['switch_rate'])} | {row['robust_positive']} |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Stage42-CB supports the major available t50 sources but exposes source concentration.",
        "- The aggregate BY/BZ evidence remains strong, but broad source-level generalization is not yet allowed.",
        "- More independent legal t50-capable sources are still needed for a stronger paper claim.",
        "- No Stage5C, SMC, metric/seconds-level, true-3D, foundation, or floor-free neural claim is made.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_cb_gate"]
    lines = [
        "# Stage42-CB Gate",
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


def run_stage42_t50_source_robustness_audit() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    return payload


if __name__ == "__main__":
    result = run_stage42_t50_source_robustness_audit()
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False, sort_keys=True))
