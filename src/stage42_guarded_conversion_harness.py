from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
CONTRACT_JSON = OUT_DIR / "source_conversion_contract_stage42.json"

REPORT_JSON = OUT_DIR / "guarded_conversion_harness_stage42.json"
REPORT_MD = OUT_DIR / "guarded_conversion_harness_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_guarded_conversion_harness_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_gm_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
ONE_FILE_SUMMARY = Path("README_M3W_ONE_FILE_DETAILED_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_gm_guarded_conversion_harness"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-GM 是 guarded converter execution harness；默认 dry-run，并且当前 contract_ready_now=0 时必须拒绝转换。",
    "本阶段不下载、不转换、不训练、不评估；没有生成新 feature store。",
    "prefill、terms hints、parseability、technical dry-run、contract opportunity 都不等于 legal converted data。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon；不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "converted_dataset_claim_allowed": False,
    "restricted_subset_claim_allowed_now": False,
    "download_executed": False,
    "conversion_executed": False,
    "feature_store_built": False,
    "no_leakage_audit_executed": False,
    "source_cv_executed": False,
    "training_executed": False,
    "evaluation_executed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}


def _ready_contract_rows(contract: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [row for row in contract.get("contract_rows", []) if row.get("contract_conversion_ready_now") is True]


def _blocked_contract_rows(contract: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [row for row in contract.get("contract_rows", []) if row.get("contract_conversion_ready_now") is not True]


def _adapter_registry() -> dict[str, dict[str, Any]]:
    return {
        "ucy_crowd_original": {
            "parser_status": "planned_not_executed",
            "supports": ["obsmat_txt", "source_level_split", "causal_velocity_rebuild"],
            "requires": ["terms_confirmation", "local_path_confirmation", "source_identity_confirmation"],
        },
        "eth_biwi_original": {
            "parser_status": "planned_not_executed",
            "supports": ["txt_xml_candidate", "source_level_split", "causal_velocity_rebuild"],
            "requires": ["terms_confirmation", "local_path_confirmation", "source_identity_confirmation"],
        },
        "trajnetplusplus_official": {
            "parser_status": "planned_not_executed",
            "supports": ["snippet_parseability_diagnostic"],
            "requires": ["longer_h100_capable_source", "terms_confirmation", "source_identity_confirmation"],
        },
    }


def _execution_plan(contract: Mapping[str, Any], *, execute: bool) -> list[dict[str, Any]]:
    registry = _adapter_registry()
    plans: list[dict[str, Any]] = []
    for row in _ready_contract_rows(contract):
        dataset_id = str(row.get("dataset_id", ""))
        adapter = registry.get(dataset_id, {"parser_status": "missing_adapter", "supports": [], "requires": []})
        adapter_ready = adapter["parser_status"] == "planned_not_executed"
        plans.append(
            {
                "dataset_id": dataset_id,
                "domain": row.get("domain", ""),
                "contract_status": row.get("contract_status", ""),
                "adapter": adapter,
                "adapter_ready_for_future_guarded_run": adapter_ready,
                "execute_requested": execute,
                "execution_status": "dry_run_ready_target" if not execute else "blocked_execute_not_implemented_in_stage42_gm",
                "conversion_executed": False,
                "feature_store_built": False,
                "no_leakage_audit_executed": False,
                "source_cv_executed": False,
                "required_pipeline_steps": [
                    "source-specific parser",
                    "row geometry reconstruction",
                    "causal velocity only",
                    "train/val/test or source-CV split rebuild",
                    "train-only goals/prototypes if legal",
                    "no future endpoint input",
                    "no central velocity",
                    "no test endpoint goals",
                    "no-leakage audit",
                    "source-CV evaluation",
                    "metric/time claim guard",
                ],
            }
        )
    return plans


def _summary(contract: Mapping[str, Any], plans: list[Mapping[str, Any]], *, execute: bool) -> dict[str, Any]:
    ready_rows = _ready_contract_rows(contract)
    blocked_rows = _blocked_contract_rows(contract)
    return {
        "source": SOURCE,
        "execute_requested": execute,
        "contract_source": contract.get("source", ""),
        "contract_verdict": contract.get("stage42_gl_gate", {}).get("verdict", ""),
        "contract_ready_now": len(ready_rows),
        "blocked_contract_rows": len(blocked_rows),
        "execution_plan_count": len(plans),
        "conversion_refused_reason": "contract_ready_now_is_zero" if not ready_rows else "",
        "download_executed": False,
        "conversion_executed": False,
        "feature_store_built": False,
        "no_leakage_audit_executed": False,
        "source_cv_executed": False,
        "training_executed": False,
        "evaluation_executed": False,
        "next_required_action": "fill terms/path/source identity and rerun validator, GL contract, then GM harness",
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    c = payload["claim_boundary"]
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "contract_loaded": payload.get("input_status", {}).get("contract_exists") is True,
        "contract_gate_passed": payload.get("contract_gate", {}).get("passed") == payload.get("contract_gate", {}).get("total"),
        "dry_run_default": s["execute_requested"] is False,
        "no_ready_refuses_conversion": s["contract_ready_now"] == 0 and s["execution_plan_count"] == 0,
        "blocked_rows_preserved": s["blocked_contract_rows"] >= 1,
        "no_download_conversion_feature_store": not (
            s["download_executed"] or s["conversion_executed"] or s["feature_store_built"]
        ),
        "no_no_leakage_or_source_cv_claim": not (s["no_leakage_audit_executed"] or s["source_cv_executed"]),
        "no_training_eval_claim": not (s["training_executed"] or s["evaluation_executed"]),
        "user_action_written": payload.get("user_action_required_written") is True,
        "no_metric_seconds_overclaim": c["global_metric_claim_allowed"] is False
        and c["global_seconds_claim_allowed"] is False,
        "no_converted_data_overclaim": c["converted_dataset_claim_allowed"] is False
        and c["restricted_subset_claim_allowed_now"] is False,
        "stage5c_false": c["stage5c_executed"] is False,
        "smc_false": c["smc_enabled"] is False,
    }
    passed = sum(bool(value) for value in gates.values())
    total = len(gates)
    verdict = "stage42_gm_guarded_conversion_harness_pass" if passed == total else "stage42_gm_guarded_conversion_harness_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-GM Guarded Conversion Harness",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_gm_gate']['passed']} / {payload['stage42_gm_gate']['total']}`",
        f"- verdict: `{payload['stage42_gm_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Summary",
        "",
        *[f"- {key}: `{value}`" for key, value in s.items()],
        "",
        "## Execution Plan",
        "",
    ]
    if payload["execution_plan"]:
        lines.extend(
            [
                "| dataset | status | adapter | conversion executed | no-leakage executed | source-CV executed |",
                "| --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in payload["execution_plan"]:
            lines.append(
                f"| `{row['dataset_id']}` | `{row['execution_status']}` | `{row['adapter']['parser_status']}` | "
                f"{row['conversion_executed']} | {row['no_leakage_audit_executed']} | {row['source_cv_executed']} |"
            )
    else:
        lines.append("- No execution plan because `contract_ready_now = 0`. Conversion is correctly refused.")
    lines.extend(
        [
            "",
            "## Blocked Contract Rows",
            "",
            "| dataset | domain | status | missing fields |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in payload["blocked_contract_rows"]:
        missing = ", ".join(row.get("confirmation", {}).get("missing_fields", [])) or "none"
        lines.append(f"| `{row.get('dataset_id', '')}` | `{row.get('domain', '')}` | `{row.get('contract_status', '')}` | {missing} |")
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_gm_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    return [
        "# User Action Required: Stage42-GM Guarded Conversion Harness",
        "",
        "The guarded converter did not run because `contract_ready_now = 0`.",
        "",
        "Required sequence before any conversion:",
        "",
        "1. Fill `outputs/stage42_long_research/source_terms_confirmation_intake_template_stage42.json` after official terms/path/source identity review.",
        "2. Run `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`.",
        "3. Run `.venv-pytorch/bin/python run_stage42_source_conversion_contract.py`.",
        "4. Run `.venv-pytorch/bin/python run_stage42_guarded_conversion_harness.py`.",
        "5. Only a future source-specific converter may build a feature store, and it must redo no-leakage/source-CV/metric-time guards.",
        "",
        "Do not count post-confirmation candidates as permission, converted data, or evaluated results.",
    ]


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_gm_gate"]
    return [
        "# Stage42-GM Gate",
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
        "## Stage42-GM Guarded Conversion Harness",
        "",
        "- source: `fresh_stage42_gm_guarded_conversion_harness`",
        "- role: executable barrier for future source-specific conversion; current run is dry-run and refuses conversion because no contract row is ready.",
        f"- gate: `{payload['stage42_gm_gate']['passed']} / {payload['stage42_gm_gate']['total']}`; verdict `{payload['stage42_gm_gate']['verdict']}`.",
        f"- contract_ready_now: `{s['contract_ready_now']}`; execution_plan_count: `{s['execution_plan_count']}`; blocked_contract_rows: `{s['blocked_contract_rows']}`.",
        "- No download, conversion, feature-store build, no-leakage audit, source-CV, training, or evaluation was executed.",
        "- Boundary: this is not converted data, not metric/seconds evidence, not Stage5C, and not SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, ONE_FILE_SUMMARY]:
        _replace_section(path, "STAGE42_GM_GUARDED_CONVERSION_HARNESS", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-GM guarded conversion harness"
    state["current_verdict"] = payload["stage42_gm_gate"]["verdict"]
    state["stage42_gm_guarded_conversion_harness"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_gm_gate"]["verdict"],
        "gates": f"{payload['stage42_gm_gate']['passed']}/{payload['stage42_gm_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_guarded_conversion_harness(*, execute: bool = False, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    contract = read_json(CONTRACT_JSON, {})
    plans = _execution_plan(contract, execute=execute)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-GM",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([CONTRACT_JSON]) if CONTRACT_JSON.exists() else "",
        "input_status": {"contract_exists": CONTRACT_JSON.exists()},
        "contract_gate": contract.get("stage42_gl_gate", {}),
        "current_facts": CURRENT_FACTS,
        "summary": _summary(contract, plans, execute=execute),
        "execution_plan": plans,
        "blocked_contract_rows": _blocked_contract_rows(contract),
        "claim_boundary": CLAIM_BOUNDARY,
        "user_action_required_written": True,
    }
    payload["stage42_gm_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_guarded_conversion_harness()
