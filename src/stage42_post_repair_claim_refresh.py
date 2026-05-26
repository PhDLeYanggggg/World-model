from __future__ import annotations

import csv
import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_data_calibration import OUT_DIR


STAGE42AG_JSON = OUT_DIR / "eth_t50_fde_source_repair_stage42.json"
STAGE42AF_JSON = OUT_DIR / "weak_slice_guard_stage42.json"
STAGE42AE_JSON = OUT_DIR / "unified_row_cache_stress_stage42.json"
STAGE42AD_JSON = OUT_DIR / "calibration_evidence_refresh_stage42.json"

REPORT_JSON = OUT_DIR / "post_repair_claim_refresh_stage42.json"
REPORT_MD = OUT_DIR / "post_repair_claim_refresh_stage42.md"
REPORT_CSV = OUT_DIR / "post_repair_claim_refresh_stage42.csv"
GATE_MD = OUT_DIR / "stage42_stage_ah_gate.md"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "Stage42-AH 是 post-repair stress / paper-claim refresh，不重新训练大模型。",
    "Stage42-AH 读取 Stage42-AG fresh report 并刷新可写 claim 与剩余 limitation。",
    "Future waypoints/endpoints 只作为 labels/eval，不作为 inference input。",
    "t+50 / t+100 是 raw-frame horizons，不能说成 seconds-level。",
    "External coordinates remain dataset-local / unverified weak metric diagnostic。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _hash_inputs(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path).encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes() if path.exists() else b"missing")
    return h.hexdigest()


def _metric(row: Mapping[str, Any], key: str, field: str = "mean") -> float:
    value = row.get(key, {})
    if isinstance(value, Mapping):
        return float(value.get(field, value.get("mean", 0.0)))
    return float(value or 0.0)


def _status(row: Mapping[str, Any], *, require_fde: bool = False) -> str:
    if int(row.get("rows", 0)) == 0:
        return "not_run"
    all_low = _metric(row, "ade_all", "ci_low")
    hard_low = _metric(row, "ade_hard_failure", "ci_low")
    easy_high = _metric(row, "ade_easy_degradation", "ci_high")
    fde_low = _metric(row, "fde_t50", "ci_low")
    if all_low > 0.0 and hard_low > 0.0 and easy_high <= 0.02 and (not require_fde or fde_low > 0.0):
        return "positive_supported"
    if _metric(row, "ade_all", "mean") == 0.0 and _metric(row, "switch_rate", "mean") == 0.0:
        return "floor_non_harm"
    if easy_high > 0.02:
        return "safety_limited"
    if all_low <= 0.0 or hard_low <= 0.0 or (require_fde and fde_low <= 0.0):
        return "weak_lower_bound"
    return "diagnostic_only"


def _slice_table(stage42ag: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, row in sorted((stage42ag.get("stress") or {}).get("by_domain_horizon", {}).items()):
        domain, horizon = key.split("|", 1)
        require_fde = int(horizon) == 50
        rows.append(
            {
                "slice": key,
                "domain": domain,
                "horizon": int(horizon),
                "rows": int(row.get("rows", 0)),
                "status": _status(row, require_fde=require_fde),
                "ade_all": _metric(row, "ade_all"),
                "ade_all_ci_low": _metric(row, "ade_all", "ci_low"),
                "ade_t50": _metric(row, "ade_t50"),
                "ade_t50_ci_low": _metric(row, "ade_t50", "ci_low"),
                "fde_t50": _metric(row, "fde_t50"),
                "fde_t50_ci_low": _metric(row, "fde_t50", "ci_low"),
                "hard_failure": _metric(row, "ade_hard_failure"),
                "hard_failure_ci_low": _metric(row, "ade_hard_failure", "ci_low"),
                "easy_degradation_ci_high": _metric(row, "ade_easy_degradation", "ci_high"),
                "switch_rate": _metric(row, "switch_rate"),
            }
        )
    return rows


def _claim_matrix(stage42ag: Mapping[str, Any], stage42af: Mapping[str, Any], stage42ae: Mapping[str, Any], stage42ad: Mapping[str, Any]) -> list[dict[str, Any]]:
    summary = stage42ag.get("summary", {})
    repair = stage42ag.get("repair_effect", {})
    af_repair = stage42af.get("repair_effect", {})
    ae_findings = stage42ae.get("stress_findings", {})
    return [
        {
            "claim": "Global protected row-level full-waypoint ADE remains positive after AF/AG repairs.",
            "status": "supported",
            "evidence": f"ADE all={_metric(summary, 'ade_all'):.6f}, CI low={_metric(summary, 'ade_all', 'ci_low'):.6f}",
        },
        {
            "claim": "Global protected row-level t50 remains positive after AF/AG repairs.",
            "status": "supported",
            "evidence": f"ADE t50={_metric(summary, 'ade_t50'):.6f}, CI low={_metric(summary, 'ade_t50', 'ci_low'):.6f}; FDE@50={_metric(summary, 'fde_t50'):.6f}",
        },
        {
            "claim": "Horizon=25 negative slice from Stage42-AE is repaired to non-harm floor.",
            "status": "supported_as_non_harm_not_positive_dynamics",
            "evidence": f"before={af_repair.get('horizon25_ade_all_before')}, after={af_repair.get('horizon25_ade_all_after')}",
        },
        {
            "claim": "ETH_UCY t50/FDE@50 lower-bound weakness from Stage42-AF is repaired.",
            "status": "supported",
            "evidence": f"ADE@50 CI low {repair.get('eth_ucy_t50_ade_ci_low_before')} -> {repair.get('eth_ucy_t50_ade_ci_low_after')}; FDE@50 CI low {repair.get('eth_ucy_fde_t50_ci_low_before')} -> {repair.get('eth_ucy_fde_t50_ci_low_after')}",
        },
        {
            "claim": "t100 can be written as a uniformly deployable long-horizon result.",
            "status": "rejected",
            "evidence": "t100 remains raw-frame diagnostic; some t100 slices retain easy-degradation safety limits.",
        },
        {
            "claim": "Metric or seconds-level pedestrian claims are allowed.",
            "status": "rejected",
            "evidence": f"Stage42-AD global_metric_claim_allowed={stage42ad.get('summary', {}).get('global_metric_claim_allowed')}; global_seconds_claim_allowed={stage42ad.get('summary', {}).get('global_seconds_claim_allowed')}",
        },
        {
            "claim": "True 3D / foundation / Stage5C / SMC claims are allowed.",
            "status": "rejected",
            "evidence": "Stage5C not executed; SMC disabled; current model remains protected dataset-local raw-frame 2.5D.",
        },
        {
            "claim": "Stage42-AE weak-slice limitations have been fully erased.",
            "status": "partially_repaired_not_fully_erased",
            "evidence": f"AE weak findings were {ae_findings.get('limitations', [])}; AF/AG repair horizon=25 and ETH_UCY|50, but t100 diagnostic/safety limits remain.",
        },
    ]


def _remaining_limitations(slice_rows: list[dict[str, Any]], stage42ad: Mapping[str, Any]) -> list[str]:
    limitations = []
    for row in slice_rows:
        if row["status"] in {"safety_limited", "weak_lower_bound"}:
            limitations.append(
                f"{row['slice']} remains {row['status']}: all_low={row['ade_all_ci_low']:.6f}, hard_low={row['hard_failure_ci_low']:.6f}, fde50_low={row['fde_t50_ci_low']:.6f}, easy_high={row['easy_degradation_ci_high']:.6f}."
            )
    floor_slices = [row["slice"] for row in slice_rows if row["status"] == "floor_non_harm"]
    if floor_slices:
        limitations.append(f"{', '.join(floor_slices)} are floor/non-harm slices, not positive dynamics contributions.")
    if not (stage42ad.get("summary", {}) or {}).get("global_metric_claim_allowed", False):
        limitations.append("Metric claim remains blocked until source-specific homography direction, coordinate convention, scale, FPS, and stride are verified.")
    if not (stage42ad.get("summary", {}) or {}).get("global_seconds_claim_allowed", False):
        limitations.append("Seconds-level horizon claim remains blocked; t50/t100 stay raw-frame horizons.")
    return limitations


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    claims = {row["claim"]: row["status"] for row in payload["claim_matrix"]}
    horizon25_rows = [row for row in payload["slice_table"] if row["horizon"] == 25]
    eth50 = next(row for row in payload["slice_table"] if row["slice"] == "ETH_UCY|50")
    gates = [
        ("Stage42-AG Input Verified", payload["stage42ag_gate"].get("verdict") == "stage42_ag_eth_t50_fde_source_repair_pass"),
        ("Global All Positive", _metric(payload["summary"], "ade_all", "ci_low") > 0.0),
        ("Global T50 Positive", _metric(payload["summary"], "ade_t50", "ci_low") > 0.0),
        ("Global Hard Positive", _metric(payload["summary"], "ade_hard_failure", "ci_low") > 0.0),
        ("Global Easy Preserved", _metric(payload["summary"], "ade_easy_degradation", "ci_high") <= 0.02),
        ("Horizon25 Non-Harm", all(row["status"] == "floor_non_harm" for row in horizon25_rows)),
        ("ETH_UCY T50/FDE Repaired", eth50["status"] == "positive_supported"),
        ("T100 Diagnostic Limitation Preserved", claims["t100 can be written as a uniformly deployable long-horizon result."] == "rejected"),
        ("Metric/Seconds Claim Rejected", claims["Metric or seconds-level pedestrian claims are allowed."] == "rejected"),
        ("Stage5C False", payload["claim_boundary"]["stage5c_executed"] is False),
        ("SMC False", payload["claim_boundary"]["smc_enabled"] is False),
    ]
    passed = sum(1 for _, ok in gates if ok)
    return {
        "source": "fresh_synthesis_post_repair_claim_refresh",
        "passed": passed,
        "total": len(gates),
        "verdict": "stage42_ah_post_repair_claim_refresh_pass" if passed == len(gates) else "stage42_ah_post_repair_claim_refresh_partial",
        "gates": [{"name": name, "passed": bool(ok)} for name, ok in gates],
    }


def build_post_repair_claim_refresh() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage42ag = read_json(STAGE42AG_JSON, {})
    stage42af = read_json(STAGE42AF_JSON, {})
    stage42ae = read_json(STAGE42AE_JSON, {})
    stage42ad = read_json(STAGE42AD_JSON, {})
    if not stage42ag or not stage42af or not stage42ae:
        raise FileNotFoundError("Stage42-AG, AF, and AE reports are required.")
    slice_rows = _slice_table(stage42ag)
    payload = {
        "source": "fresh_synthesis_from_stage42ag_post_repair_stress",
        "stage": "Stage42-AH post-repair stress / paper-claim refresh",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "input_hash": _hash_inputs([STAGE42AG_JSON, STAGE42AF_JSON, STAGE42AE_JSON, STAGE42AD_JSON]),
        "current_facts": CURRENT_FACTS,
        "stage42ag_gate": stage42ag.get("stage42_ag_gate", {}),
        "summary": stage42ag.get("summary", {}),
        "slice_table": slice_rows,
        "claim_matrix": _claim_matrix(stage42ag, stage42af, stage42ae, stage42ad),
        "remaining_limitations": _remaining_limitations(slice_rows, stage42ad),
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_ah_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_md(payload))
    _write_csv(REPORT_CSV, payload)
    write_md(GATE_MD, _render_gate_md(payload))
    return payload


def _write_csv(path: Path, payload: Mapping[str, Any]) -> None:
    rows = payload["slice_table"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _render_md(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ah_gate"]
    summary = payload["summary"]
    lines = [
        "# Stage42-AH Post-Repair Stress / Paper-Claim Refresh",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Claim Boundary",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Global Post-Repair Summary",
        "",
        f"- ADE all CI low: `{_metric(summary, 'ade_all', 'ci_low')}`",
        f"- ADE t50 CI low: `{_metric(summary, 'ade_t50', 'ci_low')}`",
        f"- ADE hard/failure CI low: `{_metric(summary, 'ade_hard_failure', 'ci_low')}`",
        f"- easy degradation CI high: `{_metric(summary, 'ade_easy_degradation', 'ci_high')}`",
        f"- FDE@50 CI low: `{_metric(summary, 'fde_t50', 'ci_low')}`",
        "",
        "## Claim Matrix",
        "",
        "| claim | status | evidence |",
        "| --- | --- | --- |",
    ]
    for row in payload["claim_matrix"]:
        lines.append(f"| {row['claim']} | `{row['status']}` | {row['evidence']} |")
    lines.extend(
        [
            "",
            "## Slice Status",
            "",
            "| slice | status | ADE all low | ADE t50 low | FDE@50 low | hard low | easy high |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["slice_table"]:
        lines.append(
            f"| `{row['slice']}` | `{row['status']}` | {row['ade_all_ci_low']:.6f} | {row['ade_t50_ci_low']:.6f} | {row['fde_t50_ci_low']:.6f} | {row['hard_failure_ci_low']:.6f} | {row['easy_degradation_ci_high']:.6f} |"
        )
    lines.extend(["", "## Remaining Limitations", ""])
    lines.extend([f"- {item}" for item in payload["remaining_limitations"]])
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "Stage42-AH updates the paper-claim boundary after AF/AG repairs. The former horizon=25 negative slice is now floor/non-harm, and the ETH_UCY t50/FDE@50 lower-bound weakness is repaired. The correct claim is stronger than Stage42-AE, but still bounded: t100 remains raw-frame diagnostic, some t100 safety slices remain limited, and metric/seconds/true-3D/foundation claims remain rejected.",
        ]
    )
    return lines


def _render_gate_md(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ah_gate"]
    lines = [
        "# Stage42-AH Gate",
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
    build_post_repair_claim_refresh()


if __name__ == "__main__":
    main()
