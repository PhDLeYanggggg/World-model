from __future__ import annotations

import csv
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "paper_package_refresh_stage42.json"
REPORT_MD = OUT_DIR / "paper_package_refresh_stage42.md"
REPORT_CSV = OUT_DIR / "paper_package_refresh_stage42.csv"
GATE_MD = OUT_DIR / "stage42_stage_ac_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

PAPER_PACKAGE_JSON = OUT_DIR / "paper_package_stage42.json"
CLAIM_AUDIT_JSON = OUT_DIR / "paper_claim_evidence_audit_stage42.json"
ROW_CACHE_JSON = OUT_DIR / "unified_row_level_full_waypoint_cache_stage42.json"
UNIFIED_ABLATION_JSON = OUT_DIR / "unified_ablation_evidence_stage42.json"
RETRAINED_MATRIX_JSON = OUT_DIR / "retrained_ablation_matrix_stage42.json"
AUX_ABLATION_JSON = OUT_DIR / "full_waypoint_auxiliary_ablation_stage42.json"

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
    "Stage42-AC 刷新 paper package，不重新训练模型，不读取 raw data/cache。",
    "Stage42-AB auxiliary-head ablation 是 fresh retrained evidence，但结果是 mixed/partial，不是统一正贡献。",
    "future endpoints / waypoints 只可作为 label 或 evaluation，不可作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test 调阈值。",
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


def _mean(summary: Mapping[str, Any], key: str) -> float:
    item = summary.get(key, {})
    if isinstance(item, Mapping):
        return float(item.get("mean", 0.0))
    return float(item or 0.0)


def _ci_low(summary: Mapping[str, Any], key: str) -> float:
    item = summary.get(key, {})
    if isinstance(item, Mapping):
        return float(item.get("ci_low", item.get("mean", 0.0)))
    return float(item or 0.0)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def _gate_pass(payload: Mapping[str, Any], gate_key: str) -> bool:
    gate = payload.get(gate_key, {})
    return bool(gate) and int(gate.get("passed", -1)) == int(gate.get("total", 0)) and int(gate.get("total", 0)) > 0


def _claim_boundary_ok(*payloads: Mapping[str, Any]) -> bool:
    for payload in payloads:
        boundary = payload.get("claim_boundary", {})
        if boundary.get("true_3d") or boundary.get("foundation_world_model") or boundary.get("metric_or_seconds_claim"):
            return False
        if boundary.get("stage5c_executed") or boundary.get("smc_enabled"):
            return False
    return True


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


def _paper_refresh_rows(
    row_cache: Mapping[str, Any],
    unified_ablation: Mapping[str, Any],
    retrained_matrix: Mapping[str, Any],
    aux: Mapping[str, Any],
    claim_audit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    x_summary = row_cache.get("summary", {})
    y_gate = unified_ablation.get("stage42_y_gate", {})
    z_gate = claim_audit.get("stage42_z_gate", {})
    aa_gate = retrained_matrix.get("stage42_aa_gate", {})
    ab_gate = aux.get("stage42_ab_gate", {})
    aux_summary = aux.get("summary", {})
    aux_delta = aux.get("delta_vs_stage42i_full", {})
    return [
        {
            "item": "Stage42-X unified row-level full-waypoint cache",
            "source": row_cache.get("source"),
            "status": (row_cache.get("stage42_x_gate") or {}).get("verdict"),
            "paper_use": "main protected 2.5D full-waypoint evidence",
            "evidence": f"ADE all={_fmt(_mean(x_summary, 'ade_all'))}, t50={_fmt(_mean(x_summary, 'ade_t50'))}, hard={_fmt(_mean(x_summary, 'ade_hard_failure'))}, easy={_fmt(_mean(x_summary, 'ade_easy_degradation'))}",
        },
        {
            "item": "Stage42-Y unified ablation evidence",
            "source": unified_ablation.get("source"),
            "status": y_gate.get("verdict"),
            "paper_use": "paper-table ablation synthesis",
            "evidence": f"gate={y_gate.get('passed')}/{y_gate.get('total')}; history/domain positive, goal/neighbor mixed",
        },
        {
            "item": "Stage42-Z claim evidence audit",
            "source": claim_audit.get("source"),
            "status": z_gate.get("verdict"),
            "paper_use": "claim boundary audit",
            "evidence": "supports protected 2.5D raw-frame paper scope; rejects metric/seconds/foundation/ungated claims",
        },
        {
            "item": "Stage42-AA retrained ablation matrix",
            "source": retrained_matrix.get("source"),
            "status": aa_gate.get("verdict"),
            "paper_use": "required ablation coverage matrix",
            "evidence": f"gate={aa_gate.get('passed')}/{aa_gate.get('total')}; no-JEPA cached negative; no-Transformer proxy boundary",
        },
        {
            "item": "Stage42-AB auxiliary-head retrained ablation",
            "source": aux.get("source"),
            "status": ab_gate.get("verdict"),
            "paper_use": "mixed/partial auxiliary evidence, not main uniform-positive claim",
            "evidence": f"no_aux all={_fmt(_mean(aux_summary, 'ade_all'))}, t50={_fmt(_mean(aux_summary, 'ade_t50'))}; full-minus-no-aux t50={_fmt(aux_delta.get('ade_t50_delta_full_minus_no_aux'))}, all={_fmt(aux_delta.get('ade_all_delta_full_minus_no_aux'))}; uniform_positive={aux.get('interpretation', {}).get('uniform_aux_positive_claim_allowed')}",
        },
    ]


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    aux = result.get("inputs_loaded", {}).get("aux_ablation", {})
    aux_interp = aux.get("interpretation", {})
    gates = {
        "paper_package_present": result.get("inputs", {}).get("paper_package_exists") is True,
        "claim_audit_present": result.get("inputs", {}).get("claim_audit_exists") is True,
        "row_cache_present": result.get("inputs", {}).get("row_cache_exists") is True,
        "unified_ablation_present": result.get("inputs", {}).get("unified_ablation_exists") is True,
        "retrained_matrix_present": result.get("inputs", {}).get("retrained_matrix_exists") is True,
        "aux_ablation_present": result.get("inputs", {}).get("aux_ablation_exists") is True,
        "aux_ablation_gate_passed": _gate_pass(aux, "stage42_ab_gate"),
        "aux_mixed_evidence_not_overclaimed": aux_interp.get("uniform_aux_positive_claim_allowed") is False,
        "paper_files_refreshed": all(row.get("contains_stage42_ac") for row in result.get("paper_file_status", [])),
        "claim_boundary_ok": result.get("claim_boundary_ok") is True,
        "stage5c_false": result.get("claim_boundary", {}).get("stage5c_executed") is False,
        "smc_false": result.get("claim_boundary", {}).get("smc_enabled") is False,
    }
    return {
        "source": result.get("source"),
        "gates": gates,
        "passed": int(sum(bool(v) for v in gates.values())),
        "total": len(gates),
        "verdict": "stage42_ac_paper_package_refresh_pass" if all(gates.values()) else "stage42_ac_paper_package_refresh_partial",
    }


def _refresh_paper_files(rows: list[dict[str, Any]], aux: Mapping[str, Any]) -> None:
    aux_delta = aux.get("delta_vs_stage42i_full", {})
    aux_summary = aux.get("summary", {})
    common_lines = [
        "## Stage42-AC Latest Evidence Refresh",
        "",
        "- source: `fresh_synthesis_from_stage42_wxyz_aa_ab_artifacts`",
        "- scope: protected dataset-local raw-frame 2.5D paper package only.",
        "- Stage42-AB is now included as auxiliary-head evidence.",
        "- Auxiliary-head conclusion: mixed / partial; not a uniform main contribution claim.",
        "- Stage5C and SMC remain disabled.",
    ]
    _replace_section(PAPER_FILES[0], "STAGE42_AC_REFRESH", common_lines + ["", "### Latest Claim Boundary", "", "The paper-ready scope remains a protected 2.5D raw-frame world-state candidate. Stage42-AB adds a useful negative/mixed result: interaction/occupancy/physical auxiliary losses help t50 slightly but do not improve all/hard ADE uniformly."])
    _replace_section(PAPER_FILES[1], "STAGE42_AC_REFRESH", common_lines + ["", "### Method Update", "", "The deployable full-waypoint model should treat interaction/occupancy/physical heads as auxiliary diagnostics. They may regularize t50/FDE@50, but the current evidence does not justify making them a central uniform contribution."])
    exp_lines = common_lines + ["", "| evidence | source | all | t50 | hard | easy | conclusion |", "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    exp_lines.append(f"| Stage42-AB no-aux full-waypoint ablation | `{aux.get('source')}` | {_fmt(_mean(aux_summary, 'ade_all'))} | {_fmt(_mean(aux_summary, 'ade_t50'))} | {_fmt(_mean(aux_summary, 'ade_hard_failure'))} | {_fmt(_mean(aux_summary, 'ade_easy_degradation'))} | no-aux variant is negative on t50; auxiliary heads have small t50 support but mixed all/hard evidence |")
    _replace_section(PAPER_FILES[2], "STAGE42_AC_REFRESH", exp_lines)
    ab_lines = common_lines + ["", "| item | source | status | paper use | evidence |", "| --- | --- | --- | --- | --- |"]
    for row in rows:
        ab_lines.append(f"| {row['item']} | `{row['source']}` | `{row['status']}` | {row['paper_use']} | {row['evidence']} |")
    ab_lines.extend(["", "### Auxiliary-Head Delta", "", "| delta | value |", "| --- | ---: |"])
    for key, value in aux_delta.items():
        ab_lines.append(f"| `{key}` | {_fmt(value)} |")
    _replace_section(PAPER_FILES[3], "STAGE42_AC_REFRESH", ab_lines)
    _replace_section(PAPER_FILES[4], "STAGE42_AC_REFRESH", common_lines + ["", "### Newly Confirmed Failure / Limitation", "", "- Auxiliary interaction/occupancy/physical heads are not uniformly positive: Stage42-AB shows small t50/FDE@50 support but negative all/hard ADE deltas. Treat them as partial/mixed evidence, not as a main contribution.", "- Ungated neural replacement, metric/seconds-level claims, true 3D, foundation claims, Stage5C, and SMC remain rejected/not enabled."])
    _replace_section(PAPER_FILES[5], "STAGE42_AC_REFRESH", common_lines + ["", "### Model-Card Update", "", "Interaction/occupancy/physical heads are present in the full-waypoint model interface, but Stage42-AB shows they should be described as auxiliary diagnostics/regularizers with mixed evidence, not as uniformly beneficial deployable heads."])
    _replace_section(PAPER_FILES[6], "STAGE42_AC_REFRESH", common_lines + ["", "### Data-Card Update", "", "No new metric/time calibration was introduced by Stage42-AB/AC. All updated paper-package claims remain dataset-local raw-frame 2.5D."])
    _replace_section(PAPER_FILES[7], "STAGE42_AC_REFRESH", common_lines + ["", "### Additional Command", "", "```bash", ".venv-pytorch/bin/python run_stage42_full_waypoint_auxiliary_ablation.py", "python3 -m pytest tests/test_stage42_full_waypoint_auxiliary_ablation.py tests/test_stage42_paper_package_refresh.py", "python3 -m pytest tests", "```"])
    _replace_section(PAPER_FILES[8], "STAGE42_AC_REFRESH", common_lines + ["", "### Updated A-Journal Gap", "", "- Need stronger uniformly positive scene/goal/interaction evidence or a cleaner theoretical framing that treats these heads as safety/diagnostic auxiliaries rather than core dynamics modules.", "- Need metric/time calibration or continue strict raw-frame language.", "- Need broader external legal datasets and stronger floor-reduction evidence before claiming foundation-track generalization."])


def _paper_file_status() -> list[dict[str, Any]]:
    rows = []
    for path in PAPER_FILES:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        rows.append(
            {
                "file": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "contains_stage42_ac": "STAGE42_AC_REFRESH:START" in text,
            }
        )
    return rows


def _write_report(result: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-AC Paper Package Refresh",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_ac_gate']['passed']} / {result['stage42_ac_gate']['total']}`",
        f"- verdict: `{result['stage42_ac_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Refreshed Evidence Rows",
        "",
        "| item | source | status | paper use | evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in result["refresh_rows"]:
        lines.append(f"| {row['item']} | `{row['source']}` | `{row['status']}` | {row['paper_use']} | {row['evidence']} |")
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "- Stage42-AB is included in the paper package as mixed/partial auxiliary-head evidence.",
            "- The current paper-ready claim remains protected dataset-local raw-frame 2.5D world-state dynamics.",
            "- No metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim is enabled.",
        ]
    )
    write_md(REPORT_MD, lines)


def _write_csv(rows: list[dict[str, Any]]) -> None:
    with REPORT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item", "source", "status", "paper_use", "evidence"])
        writer.writeheader()
        writer.writerows(rows)


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-AC Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | `{bool(ok)}` |")
    write_md(GATE_MD, lines)


def _append_if_missing(path: Path, marker: str, block: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker not in text:
        path.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def _update_readme_and_state(result: Mapping[str, Any]) -> None:
    gate = result["stage42_ac_gate"]
    block = f"""
## Stage42-AC Paper Package Refresh

```text
source = {result['source']}
verdict = {gate['verdict']}
gates = {gate['passed']} / {gate['total']}
auxiliary_head_evidence = mixed_partial_not_uniform_main_claim
paper_ready_scope = protected_dataset_local_raw_frame_2p5d_world_state_candidate
stage5c_executed = false
smc_enabled = false
```

Stage42-AC refreshes the paper outline, method draft, experiment tables, ablation tables, failure taxonomy, model card, data card, reproducibility notes, and A-journal gap analysis with Stage42-AB. The auxiliary heads are now explicitly recorded as mixed evidence: small t50/FDE@50 support, but not uniform all/hard ADE improvement.
"""
    for path in [Path("README_RESULTS.md"), Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md"), Path("README_M3W_RESEARCH_SUMMARY_ZH.md")]:
        _append_if_missing(path, "## Stage42-AC Paper Package Refresh", block)
    state = read_json(Path("research_state.json"), {})
    state["current_stage"] = "stage42_ac_paper_package_refresh"
    state["current_verdict"] = gate["verdict"]
    state.setdefault("stage42", {})["stage_ac_paper_package_refresh"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "csv": str(REPORT_CSV),
        "gates_passed": gate["passed"],
        "gates_total": gate["total"],
        "verdict": gate["verdict"],
        "paper_files_refreshed": [row["file"] for row in result.get("paper_file_status", []) if row.get("contains_stage42_ac")],
        "auxiliary_head_claim": "mixed_partial_not_uniform_main_claim",
        "claim_boundary": result["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, REPORT_CSV, GATE_MD, *PAPER_FILES]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(Path("research_state.json"), _jsonable(state))


def _append_ledger(result: Mapping[str, Any]) -> None:
    entry = {
        "command": "run_stage42_paper_package_refresh.py",
        "step": "stage42_ac_paper_package_refresh",
        "source": result.get("source"),
        "status": "success",
        "input_hash": result.get("input_hash"),
        "output_hash": _combined_hash([REPORT_JSON, REPORT_MD, REPORT_CSV, GATE_MD, *PAPER_FILES]),
        "git_commit": _git_commit(),
        "generated_at_utc": result.get("generated_at_utc"),
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


def run_stage42_paper_package_refresh() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    paper = read_json(PAPER_PACKAGE_JSON, {})
    claim_audit = read_json(CLAIM_AUDIT_JSON, {})
    row_cache = read_json(ROW_CACHE_JSON, {})
    unified_ablation = read_json(UNIFIED_ABLATION_JSON, {})
    retrained_matrix = read_json(RETRAINED_MATRIX_JSON, {})
    aux = read_json(AUX_ABLATION_JSON, {})
    rows = _paper_refresh_rows(row_cache, unified_ablation, retrained_matrix, aux, claim_audit)
    _refresh_paper_files(rows, aux)
    result = {
        "stage": "Stage42-AC paper package refresh",
        "source": "fresh_synthesis_from_stage42_wxyz_aa_ab_artifacts",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "paper_package": str(PAPER_PACKAGE_JSON),
            "paper_package_exists": PAPER_PACKAGE_JSON.exists(),
            "claim_audit": str(CLAIM_AUDIT_JSON),
            "claim_audit_exists": CLAIM_AUDIT_JSON.exists(),
            "row_cache": str(ROW_CACHE_JSON),
            "row_cache_exists": ROW_CACHE_JSON.exists(),
            "unified_ablation": str(UNIFIED_ABLATION_JSON),
            "unified_ablation_exists": UNIFIED_ABLATION_JSON.exists(),
            "retrained_matrix": str(RETRAINED_MATRIX_JSON),
            "retrained_matrix_exists": RETRAINED_MATRIX_JSON.exists(),
            "aux_ablation": str(AUX_ABLATION_JSON),
            "aux_ablation_exists": AUX_ABLATION_JSON.exists(),
        },
        "input_hash": _combined_hash([PAPER_PACKAGE_JSON, CLAIM_AUDIT_JSON, ROW_CACHE_JSON, UNIFIED_ABLATION_JSON, RETRAINED_MATRIX_JSON, AUX_ABLATION_JSON]),
        "inputs_loaded": {
            "paper_package": paper,
            "claim_audit": claim_audit,
            "row_cache": row_cache,
            "unified_ablation": unified_ablation,
            "retrained_matrix": retrained_matrix,
            "aux_ablation": aux,
        },
        "refresh_rows": rows,
        "paper_file_status": _paper_file_status(),
        "claim_boundary_ok": _claim_boundary_ok(paper, claim_audit, row_cache, unified_ablation, retrained_matrix, aux),
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    result["stage42_ac_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    _write_csv(rows)
    _write_report(result)
    _write_gate(result["stage42_ac_gate"])
    _update_readme_and_state(result)
    _append_ledger(result)
    return result


if __name__ == "__main__":
    run_stage42_paper_package_refresh()
