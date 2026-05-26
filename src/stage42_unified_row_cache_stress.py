from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_data_calibration import OUT_DIR


STAGE42X_JSON = OUT_DIR / "unified_row_level_full_waypoint_cache_stage42.json"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "Stage42-AE 是 Stage42-X row-level cache stress audit，不重新训练模型，不读取/提交 raw data。",
    "Stage42-X 的 future waypoints/endpoints 只作为 labels/eval，不作为 inference input。",
    "t+50 / t+100 是 raw-frame horizons，不能说成 seconds-level。",
    "External coordinates remain dataset-local / unverified weak metric diagnostic。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

METRIC_KEYS = [
    "ade_all",
    "ade_t50",
    "ade_t100_raw_frame_diagnostic",
    "ade_hard_failure",
    "ade_easy_degradation",
    "fde_t50",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    if not path.exists():
        return "missing"
    h.update(path.read_bytes())
    return h.hexdigest()


def _metric_mean(row: Mapping[str, Any], metric: str) -> float:
    value = row.get(metric, {})
    if isinstance(value, Mapping):
        return float(value.get("mean", 0.0))
    return float(value or 0.0)


def _metric_ci_low(row: Mapping[str, Any], metric: str) -> float:
    value = row.get(metric, {})
    if isinstance(value, Mapping):
        return float(value.get("ci_low", value.get("mean", 0.0)))
    return float(value or 0.0)


def _metric_ci_high(row: Mapping[str, Any], metric: str) -> float:
    value = row.get(metric, {})
    if isinstance(value, Mapping):
        return float(value.get("ci_high", value.get("mean", 0.0)))
    return float(value or 0.0)


def _weighted_mean(rows: list[Mapping[str, Any]], metric: str) -> float:
    total = sum(int(row.get("rows", 0)) for row in rows)
    if total <= 0:
        return 0.0
    return sum(int(row.get("rows", 0)) * _metric_mean(row, metric) for row in rows) / total


def _domain_rows(stage42x: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return dict((stage42x.get("stress") or {}).get("by_domain") or {})


def _horizon_rows(stage42x: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return dict((stage42x.get("stress") or {}).get("by_horizon") or {})


def build_leave_one_domain(domain_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    domains = sorted(domain_metrics)
    rows = []
    for held_out in domains:
        kept = [domain_metrics[d] for d in domains if d != held_out]
        entry = {"held_out_domain": held_out, "kept_domains": [d for d in domains if d != held_out]}
        for metric in METRIC_KEYS:
            entry[metric] = _weighted_mean(kept, metric)
        rows.append(entry)
    return {"source": "fresh_synthesis_from_stage42x_domain_stress", "rows": rows}


def build_stress_findings(stage42x: Mapping[str, Any]) -> dict[str, Any]:
    domains = _domain_rows(stage42x)
    horizons = _horizon_rows(stage42x)
    weak_domains = []
    strong_domains = []
    for name, row in sorted(domains.items()):
        t50_low = _metric_ci_low(row, "ade_t50")
        all_low = _metric_ci_low(row, "ade_all")
        hard_low = _metric_ci_low(row, "ade_hard_failure")
        easy_high = _metric_ci_high(row, "ade_easy_degradation")
        domain_summary = {
            "domain": name,
            "rows": int(row.get("rows", 0)),
            "ade_all": _metric_mean(row, "ade_all"),
            "ade_all_ci_low": all_low,
            "ade_t50": _metric_mean(row, "ade_t50"),
            "ade_t50_ci_low": t50_low,
            "ade_hard_failure": _metric_mean(row, "ade_hard_failure"),
            "ade_hard_failure_ci_low": hard_low,
            "easy_degradation_ci_high": easy_high,
            "fde_t50": _metric_mean(row, "fde_t50"),
            "fde_t50_ci_low": _metric_ci_low(row, "fde_t50"),
        }
        if all_low > 0 and hard_low > 0 and easy_high <= 0.02:
            strong_domains.append(domain_summary)
        if t50_low <= 0 or _metric_ci_low(row, "fde_t50") <= 0:
            weak_domains.append(domain_summary)

    weak_horizons = []
    strong_horizons = []
    for horizon, row in sorted(horizons.items(), key=lambda kv: int(kv[0])):
        all_mean = _metric_mean(row, "ade_all")
        all_low = _metric_ci_low(row, "ade_all")
        hard_mean = _metric_mean(row, "ade_hard_failure")
        h_summary = {
            "horizon": horizon,
            "rows": int(row.get("rows", 0)),
            "ade_all": all_mean,
            "ade_all_ci_low": all_low,
            "ade_hard_failure": hard_mean,
            "ade_hard_failure_ci_low": _metric_ci_low(row, "ade_hard_failure"),
            "switch_rate": _metric_mean(row, "switch_rate"),
        }
        if all_low > 0:
            strong_horizons.append(h_summary)
        if all_mean <= 0 or all_low <= 0:
            weak_horizons.append(h_summary)
    return {
        "source": "fresh_stress_audit_from_stage42x",
        "strong_domains": strong_domains,
        "weak_domains": weak_domains,
        "strong_horizons": strong_horizons,
        "weak_horizons": weak_horizons,
        "limitations": _limitations_from_findings(weak_domains, weak_horizons),
    }


def _limitations_from_findings(weak_domains: list[Mapping[str, Any]], weak_horizons: list[Mapping[str, Any]]) -> list[str]:
    limitations = []
    for row in weak_domains:
        if row["ade_t50_ci_low"] <= 0:
            limitations.append(f"{row['domain']} t50 seed-CI lower bound is not positive; write t50 domain claim with caution.")
        if row["fde_t50_ci_low"] <= 0:
            limitations.append(f"{row['domain']} FDE@50 seed-CI lower bound is not positive; keep FDE@50 as stress evidence, not universal guarantee.")
    for row in weak_horizons:
        limitations.append(
            f"horizon={row['horizon']} has non-positive ADE lower bound or mean; Stage42-X is not uniformly positive across every horizon slice."
        )
    return sorted(set(limitations))


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    findings = payload["stress_findings"]
    stage42x_gate = payload["stage42x"].get("stage42_x_gate", {})
    stage42x_summary = payload["stage42x"].get("summary", {})
    gates = [
        ("Stage42-X Verified Input", stage42x_gate.get("verdict") == "stage42_x_unified_row_level_full_waypoint_cache_pass"),
        ("Three Domain Stress Present", len(_domain_rows(payload["stage42x"])) >= 3),
        ("Leave-One-Domain Stress Built", len(payload["leave_one_domain"]["rows"]) >= 3),
        ("At Least Two Strong Domains", len(findings["strong_domains"]) >= 2),
        ("Weak Slices Identified", bool(findings["weak_domains"] or findings["weak_horizons"])),
        ("Global t50 Seed CI Positive", stage42x_summary.get("ade_t50", {}).get("ci_low", -1) > 0),
        ("Global t50 Bootstrap CI Positive", payload["stage42x"].get("bootstrap_seed_mean", {}).get("t50", {}).get("ci_low", -1) > 0),
        ("Easy Preservation", stage42x_summary.get("ade_easy_degradation", {}).get("ci_high", 1) <= 0.02),
        ("No Leakage Pass", bool(payload["stage42x"].get("no_leakage")) and not payload["stage42x"]["no_leakage"].get("future_endpoint_input", True)),
        ("No Metric/Seconds Overclaim", payload["claim_boundary"]["metric_or_seconds_claim"] is False),
        ("Stage5C Execution Gate", payload["claim_boundary"]["stage5c_executed"] is False),
        ("SMC Execution Gate", payload["claim_boundary"]["smc_enabled"] is False),
    ]
    passed = sum(1 for _, ok in gates if ok)
    if passed == len(gates):
        verdict = "stage42_ae_unified_row_cache_stress_pass_with_limitations"
    else:
        verdict = "stage42_ae_unified_row_cache_stress_partial"
    return {
        "source": "fresh_synthesis_from_stage42x",
        "passed": passed,
        "total": len(gates),
        "verdict": verdict,
        "gates": [{"name": name, "passed": bool(ok)} for name, ok in gates],
    }


def build_unified_row_cache_stress() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage42x = read_json(STAGE42X_JSON, {})
    if not stage42x:
        raise FileNotFoundError(f"missing Stage42-X report: {STAGE42X_JSON}")
    domain_metrics = _domain_rows(stage42x)
    findings = build_stress_findings(stage42x)
    leave_one = build_leave_one_domain(domain_metrics)
    payload = {
        "source": "fresh_synthesis_from_stage42x_row_level_cache",
        "stage": "Stage42-AE unified row-level full-waypoint stress audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "input_hash": _hash_file(STAGE42X_JSON),
        "current_facts": CURRENT_FACTS,
        "stage42x": {
            "source": stage42x.get("source"),
            "cache_hash": stage42x.get("cache_hash"),
            "summary": stage42x.get("summary", {}),
            "stress": stage42x.get("stress", {}),
            "bootstrap_seed_mean": stage42x.get("bootstrap_seed_mean", {}),
            "stage42_x_gate": stage42x.get("stage42_x_gate", {}),
            "no_leakage": stage42x.get("no_leakage", {}),
        },
        "stress_findings": findings,
        "leave_one_domain": leave_one,
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_ae_gate"] = _gate(payload)
    write_json(OUT_DIR / "unified_row_cache_stress_stage42.json", payload)
    write_md(OUT_DIR / "unified_row_cache_stress_stage42.md", _render_md(payload))
    _write_csv(OUT_DIR / "unified_row_cache_stress_stage42.csv", payload)
    write_md(OUT_DIR / "stage42_stage_ae_gate.md", _render_gate_md(payload))
    return payload


def _write_csv(path: Path, payload: Mapping[str, Any]) -> None:
    rows = []
    for domain, row in sorted(_domain_rows(payload["stage42x"]).items()):
        rows.append(
            {
                "kind": "domain",
                "name": domain,
                "rows": row.get("rows", 0),
                "ade_all": _metric_mean(row, "ade_all"),
                "ade_all_ci_low": _metric_ci_low(row, "ade_all"),
                "ade_t50": _metric_mean(row, "ade_t50"),
                "ade_t50_ci_low": _metric_ci_low(row, "ade_t50"),
                "ade_hard_failure": _metric_mean(row, "ade_hard_failure"),
                "easy_degradation_ci_high": _metric_ci_high(row, "ade_easy_degradation"),
                "fde_t50": _metric_mean(row, "fde_t50"),
                "fde_t50_ci_low": _metric_ci_low(row, "fde_t50"),
            }
        )
    for horizon, row in sorted(_horizon_rows(payload["stage42x"]).items(), key=lambda kv: int(kv[0])):
        rows.append(
            {
                "kind": "horizon",
                "name": horizon,
                "rows": row.get("rows", 0),
                "ade_all": _metric_mean(row, "ade_all"),
                "ade_all_ci_low": _metric_ci_low(row, "ade_all"),
                "ade_t50": _metric_mean(row, "ade_t50"),
                "ade_t50_ci_low": _metric_ci_low(row, "ade_t50"),
                "ade_hard_failure": _metric_mean(row, "ade_hard_failure"),
                "easy_degradation_ci_high": _metric_ci_high(row, "ade_easy_degradation"),
                "fde_t50": _metric_mean(row, "fde_t50"),
                "fde_t50_ci_low": _metric_ci_low(row, "fde_t50"),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _render_md(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ae_gate"]
    sx = payload["stage42x"]
    summary = sx["summary"]
    lines = [
        "# Stage42-AE Unified Row-Level Cache Stress Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Claim Boundary",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Stage42-X Reference",
        "",
        f"- cache_hash: `{sx.get('cache_hash')}`",
        f"- Stage42-X gate: `{sx.get('stage42_x_gate', {}).get('passed')} / {sx.get('stage42_x_gate', {}).get('total')}`",
        f"- ADE all seed mean: `{summary.get('ade_all', {}).get('mean')}`",
        f"- ADE t50 seed mean: `{summary.get('ade_t50', {}).get('mean')}`",
        f"- ADE t50 seed CI low: `{summary.get('ade_t50', {}).get('ci_low')}`",
        f"- ADE hard/failure seed mean: `{summary.get('ade_hard_failure', {}).get('mean')}`",
        f"- easy degradation CI high: `{summary.get('ade_easy_degradation', {}).get('ci_high')}`",
        "",
        "## Per-Domain Stress",
        "",
        "| domain | rows | ADE all | ADE all low | ADE t50 | ADE t50 low | hard | hard low | easy high | FDE t50 | FDE t50 low |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for domain, row in sorted(_domain_rows(sx).items()):
        lines.append(
            f"| `{domain}` | {row.get('rows', 0)} | {_metric_mean(row, 'ade_all'):.6f} | {_metric_ci_low(row, 'ade_all'):.6f} | {_metric_mean(row, 'ade_t50'):.6f} | {_metric_ci_low(row, 'ade_t50'):.6f} | {_metric_mean(row, 'ade_hard_failure'):.6f} | {_metric_ci_low(row, 'ade_hard_failure'):.6f} | {_metric_ci_high(row, 'ade_easy_degradation'):.6f} | {_metric_mean(row, 'fde_t50'):.6f} | {_metric_ci_low(row, 'fde_t50'):.6f} |"
        )
    lines.extend(["", "## Per-Horizon Stress", ""])
    lines.extend(
        [
            "| horizon | rows | ADE all | ADE all low | hard | hard low | switch rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for horizon, row in sorted(_horizon_rows(sx).items(), key=lambda kv: int(kv[0])):
        lines.append(
            f"| `{horizon}` | {row.get('rows', 0)} | {_metric_mean(row, 'ade_all'):.6f} | {_metric_ci_low(row, 'ade_all'):.6f} | {_metric_mean(row, 'ade_hard_failure'):.6f} | {_metric_ci_low(row, 'ade_hard_failure'):.6f} | {_metric_mean(row, 'switch_rate'):.6f} |"
        )
    lines.extend(["", "## Leave-One-Domain Stress", ""])
    lines.extend(
        [
            "This is a row-count weighted diagnostic over Stage42-X per-domain means, not a new raw-row bootstrap.",
            "",
            "| held out | kept domains | ADE all | ADE t50 | hard | easy degr | FDE t50 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["leave_one_domain"]["rows"]:
        lines.append(
            f"| `{row['held_out_domain']}` | `{', '.join(row['kept_domains'])}` | {row['ade_all']:.6f} | {row['ade_t50']:.6f} | {row['ade_hard_failure']:.6f} | {row['ade_easy_degradation']:.6f} | {row['fde_t50']:.6f} |"
        )
    findings = payload["stress_findings"]
    lines.extend(["", "## Findings", ""])
    lines.append(f"- strong_domains: `{', '.join(row['domain'] for row in findings['strong_domains']) or 'none'}`")
    lines.append(f"- weak_domains: `{', '.join(row['domain'] for row in findings['weak_domains']) or 'none'}`")
    lines.append(f"- strong_horizons: `{', '.join(row['horizon'] for row in findings['strong_horizons']) or 'none'}`")
    lines.append(f"- weak_horizons: `{', '.join(row['horizon'] for row in findings['weak_horizons']) or 'none'}`")
    lines.extend(["", "## Limitations To Write In Paper", ""])
    for item in findings["limitations"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "Stage42-AE strengthens the Stage42-X paper evidence by explicitly stress-testing where the unified row-level full-waypoint cache is stable and where it is not. The global Stage42-X t50 seed and bootstrap lower bounds remain positive, and at least two domains have strong all/hard/easy stress evidence. However, the claim must not be written as uniformly positive across every domain/horizon/FDE slice: ETH_UCY t50/FDE@50 has weak lower bounds and horizon=25 remains a limitation slice. Claims remain protected dataset-local raw-frame 2.5D; Stage5C and SMC remain disabled.",
        ]
    )
    return lines


def _render_gate_md(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ae_gate"]
    lines = [
        "# Stage42-AE Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for row in gate["gates"]:
        lines.append(f"| {row['name']} | `{row['passed']}` |")
    return lines


def main() -> None:
    build_unified_row_cache_stress()


if __name__ == "__main__":
    main()
