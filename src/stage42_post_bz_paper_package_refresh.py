from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
BY_JSON = OUT_DIR / "t50_floor_relaxability_repair_stage42.json"
BZ_JSON = OUT_DIR / "t50_repair_statistical_evidence_stage42.json"
REPORT_JSON = OUT_DIR / "paper_package_post_bz_refresh_stage42.json"
REPORT_MD = OUT_DIR / "paper_package_post_bz_refresh_stage42.md"
REPORT_CSV = OUT_DIR / "paper_package_post_bz_refresh_stage42.csv"
GATE_MD = OUT_DIR / "stage42_stage_ca_gate.md"

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
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-CA 只是 paper package refresh，不重新训练，不调 threshold，不执行 Stage5C，不启用 SMC。",
    "Stage42-BZ 的 t50 bootstrap evidence 是 protected policy evidence，不是 floor-free neural deployment。",
    "test rows 只用于最终 reporting/bootstrap，不用于 policy selection。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


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


def _pct(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%"


def _evidence_rows(by: Mapping[str, Any], bz: Mapping[str, Any]) -> list[dict[str, str]]:
    bz_summary = bz["summary"]
    union = bz["target_union_evidence"]
    rows = [
        {
            "item": "Stage42-BY protected t50 floor-relaxability repair",
            "source": by["source"],
            "status": by["stage42_by_gate"]["verdict"],
            "paper_use": "point-estimate protected t50 slice repair",
            "evidence": (
                f"repaired={', '.join(by['summary']['repaired_t50_slices'])}; "
                f"global t50={_pct(by['summary']['global_t50_improvement'])}; "
                f"easy={_pct(by['summary']['global_easy_degradation'])}; "
                "not floor-free neural"
            ),
        },
        {
            "item": "Stage42-BZ protected t50 bootstrap evidence",
            "source": bz["source"],
            "status": bz["stage42_bz_gate"]["verdict"],
            "paper_use": "bootstrap-backed t50 statistical evidence",
            "evidence": (
                f"target union t50 CI=[{_pct(union['bootstrap']['t50']['low'])}, {_pct(union['bootstrap']['t50']['high'])}]; "
                f"hard CI low={_pct(union['bootstrap']['hard_failure']['low'])}; "
                f"easy CI high={_pct(union['bootstrap']['easy_degradation']['high'])}; "
                f"n={bz_summary['bootstrap_n']}"
            ),
        },
    ]
    for key, row in bz["slice_evidence"].items():
        rows.append(
            {
                "item": f"Stage42-BZ slice {key}",
                "source": row["source"],
                "status": "ci_positive_and_easy_safe" if row["ci_positive_and_easy_safe"] else "weak",
                "paper_use": "slice-level t50 evidence",
                "evidence": (
                    f"rows={row['rows']}; "
                    f"t50 CI=[{_pct(row['bootstrap']['t50']['low'])}, {_pct(row['bootstrap']['t50']['high'])}]; "
                    f"easy CI high={_pct(row['bootstrap']['easy_degradation']['high'])}; "
                    f"switch={_pct(row['metric']['switch_rate'])}"
                ),
            }
        )
    return rows


def _paper_lines(rows: list[dict[str, str]], bz: Mapping[str, Any]) -> list[str]:
    s = bz["summary"]
    union = bz["target_union_evidence"]
    return [
        "## Stage42-CA Post-BZ Paper Evidence Refresh",
        "",
        "- source: `fresh_synthesis_from_stage42_by_bz_artifacts`",
        "- scope: protected dataset-local raw-frame 2.5D paper package only.",
        "- Stage42-BY repaired the protected t50 slices; Stage42-BZ adds bootstrap evidence.",
        "- This is protected policy evidence under the Stage37/teacher floor, not floor-free neural world dynamics.",
        "- t+50/t+100 remain raw-frame horizons; no global metric or seconds-level claim is allowed.",
        "- Stage5C remains unexecuted and SMC remains disabled.",
        "",
        "### Post-BZ Headline Evidence",
        "",
        f"- selected variant: `{s['selected_variant']}`",
        f"- internal validation group: `{s['internal_val_group']}`",
        f"- robust t50 slices: `{', '.join(s['robust_t50_slices'])}`",
        f"- target union t50 CI: `[{_pct(union['bootstrap']['t50']['low'])}, {_pct(union['bootstrap']['t50']['high'])}]`",
        f"- target union hard/failure CI low: `{_pct(union['bootstrap']['hard_failure']['low'])}`",
        f"- target union easy degradation CI high: `{_pct(union['bootstrap']['easy_degradation']['high'])}`",
        f"- bootstrap_n: `{s['bootstrap_n']}`",
        "",
        "### Evidence Rows",
        "",
        "| item | status | paper use | evidence |",
        "| --- | --- | --- | --- |",
        *[f"| {row['item']} | `{row['status']}` | {row['paper_use']} | {row['evidence']} |" for row in rows],
        "",
        "### Claim Boundary",
        "",
        "- Supported: protected t50 slice repair with bootstrap evidence for `TrajNet|50` and `UCY|50`.",
        "- Still required: teacher/floor rollout context, protected safe-switch, train/internal-validation policy selection.",
        "- Rejected: true 3D, foundation model, global metric prediction, seconds-level horizon, Stage5C execution, SMC readiness, and ungated/floor-free neural deployment.",
    ]


def _refresh_paper_files(rows: list[dict[str, str]], bz: Mapping[str, Any]) -> list[dict[str, Any]]:
    status = []
    lines = _paper_lines(rows, bz)
    for path in PAPER_FILES:
        _replace_section(path, "STAGE42_CA_REFRESH", lines)
        text = path.read_text(encoding="utf-8")
        status.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "contains_stage42_ca": "Stage42-CA Post-BZ Paper Evidence Refresh" in text,
                "contains_floor_boundary": "not floor-free neural world dynamics" in text
                or "floor-free neural deployment" in text,
                "contains_no_metric_boundary": "no global metric" in text
                or "global metric prediction" in text,
            }
        )
    return status


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    bz = payload["inputs_loaded"]["bz"]
    by = payload["inputs_loaded"]["by"]
    gates = {
        "by_gate_passed": by["stage42_by_gate"]["verdict"] == "stage42_by_t50_floor_relaxability_repair_pass",
        "bz_gate_passed": bz["stage42_bz_gate"]["verdict"] == "stage42_bz_t50_repair_statistical_evidence_pass",
        "bootstrap_n_3000": int(bz["summary"]["bootstrap_n"]) >= 3000,
        "trajnet_t50_robust": bz["slice_evidence"]["TrajNet|50"]["ci_positive_and_easy_safe"] is True,
        "ucy_t50_robust": bz["slice_evidence"]["UCY|50"]["ci_positive_and_easy_safe"] is True,
        "paper_files_refreshed": all(row["contains_stage42_ca"] for row in payload["paper_file_status"]),
        "floor_free_not_overclaimed": payload["claim_boundary"]["floor_free_neural_deployable"] is False,
        "no_metric_seconds_overclaim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_ca_post_bz_paper_package_refresh_pass" if passed == total else "stage42_ca_post_bz_paper_package_refresh_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-CA Post-BZ Paper Package Refresh",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_ca_gate']['passed']} / {payload['stage42_ca_gate']['total']}`",
        f"- verdict: `{payload['stage42_ca_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Evidence Rows",
        "",
        "| item | status | paper use | evidence |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['item']} | `{row['status']}` | {row['paper_use']} | {row['evidence']} |"
        for row in payload["evidence_rows"]
    )
    lines += [
        "",
        "## Paper File Status",
        "",
        "| file | refreshed | floor boundary | metric boundary |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in payload["paper_file_status"]:
        lines.append(
            f"| `{row['path']}` | {row['contains_stage42_ca']} | {row['contains_floor_boundary']} | {row['contains_no_metric_boundary']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Stage42-CA makes BY/BZ paper-package-visible evidence rather than a standalone loose report.",
        "- It does not train, tune, or execute any latent generative / SMC component.",
        "- It explicitly preserves the protected-policy-only claim boundary.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ca_gate"]
    lines = [
        "# Stage42-CA Gate",
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


def run_stage42_post_bz_paper_package_refresh() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    by = read_json(BY_JSON, {})
    bz = read_json(BZ_JSON, {})
    if not by or not bz:
        raise FileNotFoundError("Stage42-BY and Stage42-BZ reports are required.")
    rows = _evidence_rows(by, bz)
    paper_status = _refresh_paper_files(rows, bz)
    payload: dict[str, Any] = {
        "source": "fresh_synthesis_from_stage42_by_bz_artifacts",
        "stage": "Stage42-CA Post-BZ Paper Package Refresh",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(BY_JSON), str(BZ_JSON)]),
        "current_facts": CURRENT_FACTS,
        "inputs_loaded": {"by": by, "bz": bz},
        "evidence_rows": rows,
        "paper_file_status": paper_status,
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "floor_free_neural_deployable": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_ca_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["item", "source", "status", "paper_use", "evidence"])
        writer.writeheader()
        writer.writerows(rows)
    return payload


if __name__ == "__main__":
    print(json.dumps(run_stage42_post_bz_paper_package_refresh()["stage42_ca_gate"], indent=2, ensure_ascii=False))
