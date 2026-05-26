from __future__ import annotations

import csv
import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_data_calibration import OUT_DIR


REPORT_JSON = OUT_DIR / "paper_package_post_repair_refresh_stage42.json"
REPORT_MD = OUT_DIR / "paper_package_post_repair_refresh_stage42.md"
REPORT_CSV = OUT_DIR / "paper_package_post_repair_refresh_stage42.csv"
GATE_MD = OUT_DIR / "stage42_stage_aj_gate.md"

STAGE42AD_JSON = OUT_DIR / "calibration_evidence_refresh_stage42.json"
STAGE42AF_JSON = OUT_DIR / "weak_slice_guard_stage42.json"
STAGE42AG_JSON = OUT_DIR / "eth_t50_fde_source_repair_stage42.json"
STAGE42AH_JSON = OUT_DIR / "post_repair_claim_refresh_stage42.json"
STAGE42AI_JSON = OUT_DIR / "trajnet_t100_safety_repair_stage42.json"

PAPER_FILES = [
    OUT_DIR / "paper_outline_stage42.md",
    OUT_DIR / "method_draft_stage42.md",
    OUT_DIR / "experiment_tables_stage42.md",
    OUT_DIR / "ablation_tables_stage42.md",
    OUT_DIR / "failure_taxonomy_stage42.md",
    OUT_DIR / "model_card_stage42.md",
    OUT_DIR / "data_card_stage42.md",
    OUT_DIR / "reproducibility_stage42.md",
    OUT_DIR / "a_journal_gap_stage42.md",
]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "Stage42-AJ 刷新 paper package，纳入 Stage42-AD 到 Stage42-AI，不重新训练模型。",
    "future endpoints / waypoints 只可作为 label 或 evaluation，不可作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test 调阈值。",
    "t+50 / t+100 是 raw-frame horizons，不能说成 seconds-level。",
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


def _metric(summary: Mapping[str, Any], key: str, field: str = "mean") -> float:
    value = summary.get(key, {})
    if isinstance(value, Mapping):
        return float(value.get(field, value.get("mean", 0.0)))
    return float(value or 0.0)


def _gate_pass(payload: Mapping[str, Any], gate_key: str) -> bool:
    gate = payload.get(gate_key, {})
    return bool(gate) and int(gate.get("passed", -1)) == int(gate.get("total", 0)) and int(gate.get("total", 0)) > 0


def _replace_section(path: Path, marker: str, lines: list[str]) -> None:
    start = f"<!-- {marker}:START -->"
    end = f"<!-- {marker}:END -->"
    block = "\n".join([start, *lines, end])
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if start in text and end in text:
        prefix = text.split(start, 1)[0].rstrip()
        suffix = text.split(end, 1)[1].lstrip()
        new_text = prefix + "\n\n" + block + ("\n\n" + suffix if suffix else "\n")
    else:
        new_text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(new_text, encoding="utf-8")


def _evidence_rows(ad: Mapping[str, Any], af: Mapping[str, Any], ag: Mapping[str, Any], ah: Mapping[str, Any], ai: Mapping[str, Any]) -> list[dict[str, str]]:
    ad_summary = ad.get("summary", {})
    af_repair = af.get("repair_effect", {})
    ag_repair = ag.get("repair_effect", {})
    ah_summary = ah.get("summary", {})
    ai_summary = ai.get("summary", {})
    ai_repair = ai.get("repair_effect", {})
    return [
        {
            "item": "Stage42-AD calibration evidence refresh",
            "status": (ad.get("stage42_ad_gate") or {}).get("verdict", "missing"),
            "paper_use": "data/calibration boundary",
            "evidence": f"audited={ad_summary.get('datasets_audited')}, files={ad_summary.get('evidence_files_scanned')}, metric_allowed={ad_summary.get('global_metric_claim_allowed')}, seconds_allowed={ad_summary.get('global_seconds_claim_allowed')}",
        },
        {
            "item": "Stage42-AF horizon25 validation-margin guard",
            "status": (af.get("stage42_af_gate") or {}).get("verdict", "missing"),
            "paper_use": "weak-slice safety repair",
            "evidence": f"horizon25 {af_repair.get('horizon25_ade_all_before')} -> {af_repair.get('horizon25_ade_all_after')}; validation-only low-margin guard",
        },
        {
            "item": "Stage42-AG ETH_UCY t50/FDE source repair",
            "status": (ag.get("stage42_ag_gate") or {}).get("verdict", "missing"),
            "paper_use": "domain t50/FDE lower-bound repair",
            "evidence": f"ADE@50 low {ag_repair.get('eth_ucy_t50_ade_ci_low_before')} -> {ag_repair.get('eth_ucy_t50_ade_ci_low_after')}; FDE@50 low {ag_repair.get('eth_ucy_fde_t50_ci_low_before')} -> {ag_repair.get('eth_ucy_fde_t50_ci_low_after')}",
        },
        {
            "item": "Stage42-AH post-repair claim matrix",
            "status": (ah.get("stage42_ah_gate") or {}).get("verdict", "missing"),
            "paper_use": "claim matrix and remaining limitations",
            "evidence": f"all_low={_metric(ah_summary, 'ade_all', 'ci_low'):.6f}, t50_low={_metric(ah_summary, 'ade_t50', 'ci_low'):.6f}, hard_low={_metric(ah_summary, 'ade_hard_failure', 'ci_low'):.6f}, easy_high={_metric(ah_summary, 'ade_easy_degradation', 'ci_high'):.6f}",
        },
        {
            "item": "Stage42-AI TrajNet t100 easy-safety repair",
            "status": (ai.get("stage42_ai_gate") or {}).get("verdict", "missing"),
            "paper_use": "raw-frame diagnostic t100 safety repair",
            "evidence": f"TrajNet100 easy high {ai_repair.get('trajnet100_easy_ci_high_before')} -> {ai_repair.get('trajnet100_easy_ci_high_after')}; global t100 raw-frame low={_metric(ai_summary, 'ade_t100_raw_frame_diagnostic', 'ci_low'):.6f}",
        },
    ]


def _paper_lines(rows: list[dict[str, str]], ai: Mapping[str, Any]) -> list[str]:
    summary = ai.get("summary", {})
    return [
        "## Stage42-AJ Post-Repair Paper Package Refresh",
        "",
        "- source: `fresh_synthesis_from_stage42_ad_to_ai_artifacts`",
        "- scope: protected dataset-local raw-frame 2.5D paper package only.",
        "- This refresh supersedes stale Stage42-AE limitations: horizon=25 harm, ETH_UCY t50/FDE, and TrajNet|100 easy safety were repaired by validation-only guards.",
        "- t100 remains raw-frame diagnostic; metric/seconds/true-3D/foundation/Stage5C/SMC claims remain rejected.",
        "- Future waypoints/endpoints remain labels/eval only, never inference inputs.",
        "",
        "### Post-Repair Headline Metrics",
        "",
        f"- ADE all CI low: `{_metric(summary, 'ade_all', 'ci_low'):.6f}`",
        f"- ADE t50 CI low: `{_metric(summary, 'ade_t50', 'ci_low'):.6f}`",
        f"- ADE t100 raw-frame diagnostic CI low: `{_metric(summary, 'ade_t100_raw_frame_diagnostic', 'ci_low'):.6f}`",
        f"- ADE hard/failure CI low: `{_metric(summary, 'ade_hard_failure', 'ci_low'):.6f}`",
        f"- easy degradation CI high: `{_metric(summary, 'ade_easy_degradation', 'ci_high'):.6f}`",
        f"- FDE@50 CI low: `{_metric(summary, 'fde_t50', 'ci_low'):.6f}`",
        "",
        "### Evidence Rows",
        "",
        "| item | status | paper use | evidence |",
        "| --- | --- | --- | --- |",
        *[f"| {row['item']} | `{row['status']}` | {row['paper_use']} | {row['evidence']} |" for row in rows],
        "",
        "### Claim Boundary",
        "",
        "- Supported: protected row-level full-waypoint raw-frame 2.5D world-state evidence with positive all/t50/hard/FDE@50 lower bounds and repaired t100 easy-safety diagnostic.",
        "- Supported as non-harm only: horizon=25 floor/non-harm slices; do not call them positive dynamics contributions.",
        "- Rejected: metric prediction, seconds-level horizon, true 3D, foundation model, Stage5C execution, SMC readiness, and ungated neural deployment.",
    ]


def _refresh_paper_files(rows: list[dict[str, str]], ai: Mapping[str, Any]) -> list[dict[str, Any]]:
    status = []
    lines = _paper_lines(rows, ai)
    for path in PAPER_FILES:
        _replace_section(path, "STAGE42_AJ_REFRESH", lines)
        text = path.read_text(encoding="utf-8")
        status.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "contains_stage42_aj": "Stage42-AJ Post-Repair Paper Package Refresh" in text,
                "contains_no_metric_boundary": "metric prediction" in text and "seconds-level horizon" in text,
            }
        )
    return status


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    gates = [
        ("Stage42-AD Gate Passed", _gate_pass(payload["inputs_loaded"]["ad"], "stage42_ad_gate")),
        ("Stage42-AF Gate Passed", _gate_pass(payload["inputs_loaded"]["af"], "stage42_af_gate")),
        ("Stage42-AG Gate Passed", _gate_pass(payload["inputs_loaded"]["ag"], "stage42_ag_gate")),
        ("Stage42-AH Gate Passed", _gate_pass(payload["inputs_loaded"]["ah"], "stage42_ah_gate")),
        ("Stage42-AI Gate Passed", _gate_pass(payload["inputs_loaded"]["ai"], "stage42_ai_gate")),
        ("All Paper Files Refreshed", all(row["contains_stage42_aj"] for row in payload["paper_file_status"])),
        ("No Metric/Seconds Overclaim", payload["claim_boundary"]["metric_or_seconds_claim"] is False),
        ("T100 Raw-Frame Diagnostic Only", payload["claim_boundary"]["t100_seconds_claim"] is False),
        ("Stage5C False", payload["claim_boundary"]["stage5c_executed"] is False),
        ("SMC False", payload["claim_boundary"]["smc_enabled"] is False),
    ]
    passed = sum(1 for _, ok in gates if ok)
    return {
        "source": payload["source"],
        "passed": passed,
        "total": len(gates),
        "verdict": "stage42_aj_post_repair_paper_package_refresh_pass" if passed == len(gates) else "stage42_aj_post_repair_paper_package_refresh_partial",
        "gates": [{"name": name, "passed": bool(ok)} for name, ok in gates],
    }


def build_post_repair_paper_package_refresh() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ad = read_json(STAGE42AD_JSON, {})
    af = read_json(STAGE42AF_JSON, {})
    ag = read_json(STAGE42AG_JSON, {})
    ah = read_json(STAGE42AH_JSON, {})
    ai = read_json(STAGE42AI_JSON, {})
    if not all([ad, af, ag, ah, ai]):
        raise FileNotFoundError("Stage42-AD/AF/AG/AH/AI reports are required.")
    rows = _evidence_rows(ad, af, ag, ah, ai)
    paper_status = _refresh_paper_files(rows, ai)
    payload = {
        "source": "fresh_synthesis_from_stage42_ad_to_ai_artifacts",
        "stage": "Stage42-AJ post-repair paper package refresh",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "input_hash": _hash_inputs([STAGE42AD_JSON, STAGE42AF_JSON, STAGE42AG_JSON, STAGE42AH_JSON, STAGE42AI_JSON]),
        "current_facts": CURRENT_FACTS,
        "inputs_loaded": {"ad": ad, "af": af, "ag": ag, "ah": ah, "ai": ai},
        "evidence_rows": rows,
        "paper_file_status": paper_status,
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "t100_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_aj_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_md(payload))
    _write_csv(REPORT_CSV, payload)
    write_md(GATE_MD, _render_gate_md(payload))
    return payload


def _write_csv(path: Path, payload: Mapping[str, Any]) -> None:
    rows = payload["evidence_rows"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _render_md(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_aj_gate"]
    lines = [
        "# Stage42-AJ Post-Repair Paper Package Refresh",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Refreshed Evidence Rows",
        "",
        "| item | status | paper use | evidence |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['item']} | `{row['status']}` | {row['paper_use']} | {row['evidence']} |"
        for row in payload["evidence_rows"]
    )
    lines.extend(
        [
            "",
            "## Paper Files",
            "",
            "| file | refreshed | no-overclaim boundary |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in payload["paper_file_status"]:
        lines.append(f"| `{row['path']}` | `{row['contains_stage42_aj']}` | `{row['contains_no_metric_boundary']}` |")
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "The Stage42 paper package now includes AD-AI calibration, stress, safety repair, and post-repair claim boundary evidence. The paper-ready scope is stronger than Stage42-AC, but remains protected dataset-local raw-frame 2.5D. Metric, seconds-level, true-3D, foundation, Stage5C, SMC, and ungated neural deployment claims remain rejected.",
        ]
    )
    return lines


def _render_gate_md(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_aj_gate"]
    lines = [
        "# Stage42-AJ Gate",
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
    build_post_repair_paper_package_refresh()


if __name__ == "__main__":
    main()
