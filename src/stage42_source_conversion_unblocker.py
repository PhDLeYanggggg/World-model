from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
CG_JSON = OUT_DIR / "source_terms_validation_stage42.json"
DW_JSON = OUT_DIR / "source_specific_conversion_dry_run_stage42.json"
DO_JSON = OUT_DIR / "source_legal_time_action_package_stage42.json"
DS_JSON = OUT_DIR / "source_conversion_readiness_recheck_stage42.json"
TERMS_TEMPLATE_JSON = OUT_DIR / "source_terms_confirmation_template_stage42.json"

REPORT_JSON = OUT_DIR / "source_conversion_unblocker_stage42.json"
REPORT_MD = OUT_DIR / "source_conversion_unblocker_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_source_conversion_unblocker_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ed_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-ED 是 source-conversion unblocker package，不下载、不转换、不训练、不评估。",
    "本阶段把 CG/DW/DO/DS 的 legal/source/time blockers 汇总成可执行用户动作。",
    "local path、parseability、technical dry-run 都不等于 legal conversion readiness。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon；metric/seconds-level claim 仍被阻塞。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "converted_datasets_now": 0,
    "evaluated_datasets_now": 0,
    "stage5c_executed": False,
    "smc_enabled": False,
}

DATASET_TO_DW_DATASETS = {
    "ucy_crowd_original": {"UCY"},
    "eth_biwi_original": {"ETH"},
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _gate_passed(payload: Mapping[str, Any], key: str) -> bool:
    gate = payload.get(key, {})
    return bool(gate.get("passed") == gate.get("total") and gate.get("total", 0) > 0)


def _action_rows(cg: Mapping[str, Any], dw: Mapping[str, Any], do: Mapping[str, Any], ds: Mapping[str, Any]) -> list[dict[str, Any]]:
    validations = {row["dataset_id"]: row for row in cg.get("validations", [])}
    do_rows = {row["dataset_id"]: row for row in do.get("user_action_rows", [])}
    ds_raw = set(ds.get("summary", {}).get("raw_path_found_ids", []))
    ds_derived = set(ds.get("summary", {}).get("derived_cache_found_ids", []))
    dw_by_dataset: dict[str, list[Mapping[str, Any]]] = {}
    for row in dw.get("source_rows", []):
        dw_by_dataset.setdefault(str(row.get("dataset")), []).append(row)
    rows: list[dict[str, Any]] = []
    for dataset_id, validation in validations.items():
        action = do_rows.get(dataset_id, {})
        dw_keys = set(DATASET_TO_DW_DATASETS.get(dataset_id, set()))
        dw_keys.add(dataset_id)
        source_rows = [source_row for key in dw_keys for source_row in dw_by_dataset.get(key, [])]
        technical_ready_sources = [row for row in source_rows if row.get("technical_conversion_ready_after_terms")]
        t50 = sum(int(row.get("horizon_counts", {}).get("50", 0)) for row in source_rows)
        t100 = sum(int(row.get("horizon_counts", {}).get("100", 0)) for row in source_rows)
        terms_blockers = list(validation.get("confirmation_blockers", []))
        cf_blockers = list(validation.get("cf_blockers", []))
        blocker_class = "user_terms_required" if terms_blockers else "source_cv_or_identity_required"
        if "local_path_missing" in cf_blockers or "local_path_confirmation_missing" in terms_blockers:
            blocker_class = "local_path_and_terms_required"
        if dataset_id == "opentraj_toolkit":
            blocker_class = "toolkit_not_independent_source"
        if dataset_id == "aerialmpt_or_other_topdown":
            blocker_class = "new_official_source_required"
        purpose = "source-specific metric/time and source-CV repair" if technical_ready_sources else "source-diversity acquisition or identity repair"
        rows.append(
            {
                "dataset_id": dataset_id,
                "name": validation.get("name", dataset_id),
                "domain": action.get("domain", "unknown"),
                "official_url": validation.get("official_url", action.get("official_url", "")),
                "raw_path_found": dataset_id in ds_raw,
                "derived_cache_found": dataset_id in ds_derived,
                "terms_accepted_by_user": validation.get("terms_accepted_by_user", False),
                "conversion_ready": validation.get("conversion_ready", False),
                "conversion_allowed_now": False,
                "converted_now": False,
                "evaluated_now": False,
                "terms_blockers": terms_blockers,
                "cf_blockers": cf_blockers,
                "domain_blockers": action.get("domain_blockers", []),
                "blocker_class": blocker_class,
                "source_specific_metric_time_sources": action.get("source_specific_metric_time_sources", []),
                "source_specific_time_only_sources": action.get("source_specific_time_only_sources", []),
                "technical_ready_source_ids_after_terms": [row["source_id"] for row in technical_ready_sources],
                "technical_ready_source_count_after_terms": len(technical_ready_sources),
                "estimated_t50_windows_after_terms": int(t50),
                "estimated_t100_windows_after_terms": int(t100),
                "purpose": purpose,
                "required_user_action": action.get("required_user_action", validation.get("next_action", "")),
                "next_command_after_user_confirmation": ".venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py",
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            0 if row["dataset_id"] == "ucy_crowd_original" else 1 if row["dataset_id"] == "eth_biwi_original" else 2,
            -row["technical_ready_source_count_after_terms"],
            -row["estimated_t50_windows_after_terms"],
            row["dataset_id"],
        ),
    )


def _summary(rows: list[Mapping[str, Any]], cg: Mapping[str, Any], dw: Mapping[str, Any], do: Mapping[str, Any], ds: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source": "fresh_synthesis_from_stage42_cg_dw_do_ds",
        "targets": len(rows),
        "conversion_ready_now": sum(1 for row in rows if row["conversion_ready"]),
        "conversion_allowed_now": 0,
        "converted_datasets_now": 0,
        "evaluated_datasets_now": 0,
        "raw_path_found_targets": ds.get("summary", {}).get("raw_path_found_targets", 0),
        "technical_ready_after_terms_targets": sum(1 for row in rows if row["technical_ready_source_count_after_terms"] > 0),
        "estimated_t50_windows_after_terms": sum(int(row["estimated_t50_windows_after_terms"]) for row in rows),
        "estimated_t100_windows_after_terms": sum(int(row["estimated_t100_windows_after_terms"]) for row in rows),
        "domains_with_source_cv_after_terms": dw.get("summary", {}).get("domains_with_source_cv_after_terms", []),
        "source_specific_metric_time_sources": do.get("summary", {}).get("source_specific_metric_time_sources", []),
        "terms_accepted_targets": cg.get("summary", {}).get("terms_accepted_targets", 0),
        "user_action_required_targets": sum(1 for row in rows if not row["conversion_ready"]),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    claim = payload["claim_boundary"]
    gates = {
        "cg_input_passed": payload["input_gates"]["cg"],
        "dw_input_passed": payload["input_gates"]["dw"],
        "do_input_passed": payload["input_gates"]["do"],
        "ds_input_passed": payload["input_gates"]["ds"],
        "action_rows_written": len(payload["action_rows"]) >= 5,
        "ucy_priority_present": any(row["dataset_id"] == "ucy_crowd_original" for row in payload["action_rows"]),
        "eth_priority_present": any(row["dataset_id"] == "eth_biwi_original" for row in payload["action_rows"]),
        "technical_ready_after_terms_recorded": s["technical_ready_after_terms_targets"] >= 2,
        "t50_t100_after_terms_recorded": s["estimated_t50_windows_after_terms"] > 0 and s["estimated_t100_windows_after_terms"] > 0,
        "legal_blocker_preserved": s["conversion_allowed_now"] == 0 and s["conversion_ready_now"] == 0,
        "no_conversion_or_eval_claim": s["converted_datasets_now"] == 0 and s["evaluated_datasets_now"] == 0,
        "user_action_written": payload["user_action_required_written"] is True,
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_ed_source_conversion_unblocker_pass" if passed == total else "stage42_ed_source_conversion_unblocker_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-ED Source Conversion Unblocker Package",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_ed_gate']['passed']} / {payload['stage42_ed_gate']['total']}`",
        f"- verdict: `{payload['stage42_ed_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Summary",
        "",
        *[f"- {key}: `{value}`" for key, value in payload["summary"].items()],
        "",
        "## Unblocker Table",
        "",
        "| dataset | domain | raw path | technical sources after terms | t50 after terms | t100 after terms | blocker class | official URL |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload["action_rows"]:
        lines.append(
            f"| `{row['dataset_id']}` | `{row['domain']}` | {row['raw_path_found']} | {row['technical_ready_source_count_after_terms']} | {row['estimated_t50_windows_after_terms']} | {row['estimated_t100_windows_after_terms']} | `{row['blocker_class']}` | {row['official_url']} |"
        )
    lines.extend(
        [
            "",
            "## Required Next Commands",
            "",
            "1. Fill `outputs/stage42_long_research/source_terms_confirmation_template_stage42.json` with explicit user-confirmed official terms, allowed use, local path, and source identity.",
            "2. Run `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`.",
            "3. Only if conversion-ready targets become nonzero, run a future guarded conversion/no-leakage/source-CV stage. Stage42-ED does not convert.",
            "",
            "## Interpretation",
            "",
            "- UCY and ETH/BIWI are the first legal unblock targets because the dry-run found source-specific metric/time candidates after terms.",
            "- OpenTraj remains useful as toolkit/reference evidence, but toolkit presence is not an independent source-rights claim.",
            "- No metric/seconds-level or converted-data claim is allowed until terms/path/source identity are confirmed and a future conversion/audit runs.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_ed_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-ED Source Conversion Unblocker",
        "",
        "Stage42-ED did not download, convert, train, or evaluate data. The next step requires explicit user confirmation of official terms and source identity.",
        "",
        "| priority | dataset | official URL | purpose | missing action |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for idx, row in enumerate(payload["action_rows"], start=1):
        missing = sorted(set(row["terms_blockers"] + row["cf_blockers"] + row["domain_blockers"]))
        lines.append(
            f"| {idx} | `{row['dataset_id']}` | {row['official_url']} | {row['purpose']} | {', '.join(missing) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "Required confirmation fields per dataset:",
            "",
            "- `dataset_id`",
            "- `official_url`",
            "- `terms_accepted_by_user: true`",
            "- `terms_acceptance_date`",
            "- `allowed_use`",
            "- `local_path`",
            "- `source_identity`",
            "- `notes`",
            "",
            "After filling the confirmation template, run:",
            "",
            "```bash",
            ".venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py",
            "```",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ed_gate"]
    return [
        "# Stage42-ED Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    return [
        "## Stage42-ED Source Conversion Unblocker Package",
        "",
        "- source: `fresh_synthesis_from_stage42_cg_dw_do_ds`",
        "- role: convert local parseability/source-specific calibration hints into exact user actions; no download/conversion/evaluation.",
        f"- gate: `{payload['stage42_ed_gate']['passed']} / {payload['stage42_ed_gate']['total']}`; verdict `{payload['stage42_ed_gate']['verdict']}`.",
        f"- conversion_ready_now: `{s['conversion_ready_now']}`; conversion_allowed_now: `{s['conversion_allowed_now']}`; converted/evaluated now `{s['converted_datasets_now']}` / `{s['evaluated_datasets_now']}`.",
        f"- technical_ready_after_terms_targets: `{s['technical_ready_after_terms_targets']}`; estimated t50/t100 windows after terms `{s['estimated_t50_windows_after_terms']}` / `{s['estimated_t100_windows_after_terms']}`.",
        f"- domains_with_source_cv_after_terms: `{s['domains_with_source_cv_after_terms']}`; first unblock targets remain UCY and ETH/BIWI terms/path/source identity.",
        "- boundary: local path and parseability are not legal conversion; metric/seconds, Stage5C, and SMC remain blocked.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, GOAL_SUMMARY]:
        _replace_section(path, "STAGE42_ED_SOURCE_CONVERSION_UNBLOCKER", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-ED source conversion unblocker"
    state["current_verdict"] = payload["stage42_ed_gate"]["verdict"]
    state["stage42_ed_source_conversion_unblocker"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_ed_gate"]["verdict"],
        "gates": f"{payload['stage42_ed_gate']['passed']}/{payload['stage42_ed_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_source_conversion_unblocker(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    cg = read_json(CG_JSON, {})
    dw = read_json(DW_JSON, {})
    do = read_json(DO_JSON, {})
    ds = read_json(DS_JSON, {})
    rows = _action_rows(cg, dw, do, ds)
    payload: dict[str, Any] = {
        "source": "fresh_synthesis_from_stage42_cg_dw_do_ds",
        "stage": "Stage42-ED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([CG_JSON, DW_JSON, DO_JSON, DS_JSON, TERMS_TEMPLATE_JSON]),
        "current_facts": CURRENT_FACTS,
        "input_gates": {
            "cg": _gate_passed(cg, "stage42_cg_gate"),
            "dw": _gate_passed(dw, "stage42_dw_gate"),
            "do": _gate_passed(do, "stage42_do_gate"),
            "ds": _gate_passed(ds, "stage42_ds_gate"),
        },
        "action_rows": rows,
        "summary": _summary(rows, cg, dw, do, ds),
        "claim_boundary": CLAIM_BOUNDARY,
        "user_action_required_written": True,
    }
    payload["stage42_ed_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_source_conversion_unblocker()
