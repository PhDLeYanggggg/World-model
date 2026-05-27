from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_full_trajectory_world_state as ft
from src import stage42_explicit_gain_harm_selector as s42o
from src import stage42_row_gain_static_gate as s42n
from src import stage42_sequence_full_waypoint as s42i
from src import stage42_t50_gain_harm_selector as s42p
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section
from src.stage42_t50_gain_harm_stability_audit import _domain_instability, _validation_score


OUT_DIR = Path("outputs/stage42_long_research")
STAGE42P_JSON = OUT_DIR / "t50_gain_harm_selector_stage42.json"
REPORT_JSON = OUT_DIR / "t50_gain_harm_seed_expansion_stage42.json"
REPORT_MD = OUT_DIR / "t50_gain_harm_seed_expansion_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ih_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
MASTER_README = Path("README_M3W_MASTER_SUMMARY_2026_05_27_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE42_IH_T50_GAIN_HARM_SEED_EXPANSION"
SOURCE = "fresh_stage42_ih_t50_gain_harm_seed_expansion"

EXTRA_SEEDS = [163, 167, 173]
EXTRA_BASE_SEEDS = [109, 113, 127]


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


def _seed_table(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for row in rows:
        ade = (((row.get("test_metrics", {}) or {}).get("ade", {}) or {}))
        fde = (((row.get("test_metrics", {}) or {}).get("fde", {}) or {}))
        source = row.get("source", "unknown")
        selector_source = (((row.get("selector_info", {}) or {}).get("source", source)))
        table.append(
            {
                "seed": int(row.get("seed", -1)),
                "base_seed": int(row.get("base_seed", -1)),
                "source": source,
                "selector_source": selector_source,
                "validation_score": _validation_score(row),
                "ade_all": float(ade.get("all_improvement", 0.0)),
                "ade_t50": float(ade.get("t50_improvement", 0.0)),
                "ade_t100_raw_frame_diagnostic": float(ade.get("t100_improvement", 0.0)),
                "ade_hard_failure": float(ade.get("hard_failure_improvement", 0.0)),
                "ade_easy_degradation": float(ade.get("easy_degradation", 0.0)),
                "fde_t50": float(fde.get("t50_improvement", 0.0)),
                "switch_rate": float(ade.get("switch_rate", row.get("test_metrics", {}).get("switch_rate", 0.0))),
            }
        )
    return sorted(table, key=lambda item: item["seed"])


def _selected_by_val(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"selected": False}
    row = max(rows, key=_validation_score)
    ade = (((row.get("test_metrics", {}) or {}).get("ade", {}) or {}))
    fde = (((row.get("test_metrics", {}) or {}).get("fde", {}) or {}))
    return {
        "selected": True,
        "selection_rule": "validation_only_t50_weighted_score_no_test_threshold_tuning",
        "seed": int(row.get("seed", -1)),
        "base_seed": int(row.get("base_seed", -1)),
        "validation_score": _validation_score(row),
        "test_ade_all": float(ade.get("all_improvement", 0.0)),
        "test_ade_t50": float(ade.get("t50_improvement", 0.0)),
        "test_ade_hard_failure": float(ade.get("hard_failure_improvement", 0.0)),
        "test_ade_easy_degradation": float(ade.get("easy_degradation", 0.0)),
        "test_fde_t50": float(fde.get("t50_improvement", 0.0)),
    }


def _prepare_common_data() -> dict[str, Any]:
    ft.build_full_trajectory_labels()
    data = {split: s42i._split_arrays(split) for split in ["train", "val", "test"]}
    vocab = s42o._domain_vocab(data["train"], data["val"], data["test"])
    train_teacher = s42n._row_teacher(data["train"], "train")
    val_teacher = s42n._row_teacher(data["val"], "val")
    return {"data": data, "vocab": vocab, "train_teacher": train_teacher, "val_teacher": val_teacher}


def _run_extra_rows() -> list[dict[str, Any]]:
    common = _prepare_common_data()
    data = common["data"]
    rows: list[dict[str, Any]] = []
    for seed, base_seed in zip(EXTRA_SEEDS, EXTRA_BASE_SEEDS):
        rows.append(
            s42p._eval_seed(
                seed,
                base_seed,
                data["train"],
                data["val"],
                data["test"],
                common["vocab"],
                common["train_teacher"],
                common["val_teacher"],
            )
        )
    return rows


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    original = read_json(STAGE42P_JSON, {})
    original_rows = list(original.get("rows", []))
    extra_rows = _run_extra_rows()
    combined_rows = [*original_rows, *extra_rows]
    original_summary = s42p._summary(original_rows, "test_metrics") if original_rows else {}
    expanded_summary = s42p._summary(combined_rows, "test_metrics")
    original_domain_instability = _domain_instability(original_rows)
    expanded_domain_instability = _domain_instability(combined_rows)
    selected = _selected_by_val(combined_rows)
    payload = {
        "stage": "Stage42-IH",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(
            [
                STAGE42P_JSON,
                ft.DATA_DIR / "all_agent_train.npz",
                ft.DATA_DIR / "all_agent_val.npz",
                ft.DATA_DIR / "all_agent_test.npz",
                ft.DATA_DIR / "full_trajectory_train.npz",
                ft.DATA_DIR / "full_trajectory_val.npz",
                ft.DATA_DIR / "full_trajectory_test.npz",
                OUT_DIR / "row_gain_static_gate_stage42.json",
            ]
        ),
        "original_seed_count": len(original_rows),
        "extra_seed_count": len(extra_rows),
        "combined_seed_count": len(combined_rows),
        "extra_seeds": EXTRA_SEEDS,
        "extra_base_seeds": EXTRA_BASE_SEEDS,
        "original_summary": original_summary,
        "expanded_summary": expanded_summary,
        "seed_table": _seed_table(combined_rows),
        "validation_selected": selected,
        "original_domain_instability": original_domain_instability,
        "expanded_domain_instability": expanded_domain_instability,
        "source_labels": {
            "original_stage42p_rows": "cached_verified",
            "extra_seed_training": "fresh_run_or_cached_checkpoint_resume",
            "base_model_checkpoints": "cached_verified_stage42n",
            "base_model_seed_reuse": "explicitly_disclosed",
            "feature_normalization": "train_split_stats_only",
            "test_evaluation": "fresh_once_per_extra_seed_plus_cached_original",
            "row_bootstrap": "not_run_in_this_stage_use_stage42_ig_for_validation_selected_seed",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "future_waypoints_used_as_train_val_label_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_statistics_normalization": False,
            "thresholds_selected_on_val": True,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_ih_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload.get("expanded_summary", {})
    claim = payload.get("claim_boundary", {})
    no_leakage = payload.get("no_leakage", {})
    selected = payload.get("validation_selected", {})
    gates = {
        "original_rows_loaded": payload.get("original_seed_count", 0) >= 3,
        "extra_seeds_attempted": payload.get("extra_seed_count", 0) >= len(EXTRA_SEEDS),
        "combined_seed_count_at_least_six": payload.get("combined_seed_count", 0) >= 6,
        "combined_ade_all_positive": s.get("ade_all", {}).get("mean", 0.0) > 0.0,
        "combined_ade_t50_positive": s.get("ade_t50", {}).get("mean", 0.0) > 0.0,
        "combined_ade_t50_seed_ci_positive": s.get("ade_t50", {}).get("ci_low", -1.0) > 0.0,
        "combined_fde_t50_seed_ci_positive": s.get("fde_t50", {}).get("ci_low", -1.0) > 0.0,
        "combined_hard_positive": s.get("ade_hard_failure", {}).get("mean", 0.0) > 0.0,
        "combined_easy_preserved": s.get("ade_easy_degradation", {}).get("mean", 1.0) <= 0.02,
        "validation_selected_t50_positive": selected.get("test_ade_t50", -1.0) > 0.0,
        "domain_instability_audited": "expanded_domain_instability" in payload,
        "no_future_endpoint_or_waypoint_input": no_leakage.get("future_endpoint_input") is False
        and no_leakage.get("future_waypoints_input") is False,
        "no_central_velocity_or_test_goal": no_leakage.get("central_velocity") is False
        and no_leakage.get("test_endpoint_goals") is False,
        "no_metric_seconds_overclaim": claim.get("metric_or_seconds_claim") is False,
        "stage5c_false": claim.get("stage5c_executed") is False,
        "smc_false": claim.get("smc_enabled") is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    if passed == total:
        verdict = "stage42_ih_t50_seed_expansion_stabilizes_ade_t50"
    elif gates["combined_ade_t50_positive"] and not gates["combined_ade_t50_seed_ci_positive"]:
        verdict = "stage42_ih_t50_seed_expansion_mean_positive_ci_blocker_remains"
    else:
        verdict = "stage42_ih_t50_seed_expansion_partial"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _fmt_metric(summary: Mapping[str, Any], key: str) -> str:
    row = summary.get(key, {}) or {}
    return f"{row.get('mean', 0.0):.6f} [{row.get('ci_low', 0.0):.6f}, {row.get('ci_high', 0.0):.6f}]"


def _write_report(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_ih_gate"]
    orig = payload["original_summary"]
    exp = payload["expanded_summary"]
    selected = payload["validation_selected"]
    lines = [
        "# Stage42-IH T50 Gain/Harm Seed Expansion",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Purpose",
        "",
        "Stage42-IF found seed-level ADE t+50 instability, while Stage42-IG validated the validation-selected seed with row bootstrap. Stage42-IH adds more t+50 gain/harm selector seeds to test whether the mean t+50 gain becomes seed-stable.",
        "",
        "## Claim Boundary",
        "",
        "- dataset-local/raw-frame 2.5D only",
        "- no true 3D, metric, seconds-level, or foundation claim",
        "- no Stage5C execution",
        "- no SMC",
        "- future waypoints are train/val labels and final evaluation labels only, not inference inputs",
        "",
        "## Seed Expansion Summary",
        "",
        "| metric | original 3 seeds | expanded seeds |",
        "| --- | ---: | ---: |",
        f"| seed count | {payload['original_seed_count']} | {payload['combined_seed_count']} |",
        f"| ADE all mean [CI] | {_fmt_metric(orig, 'ade_all')} | {_fmt_metric(exp, 'ade_all')} |",
        f"| ADE t50 mean [CI] | {_fmt_metric(orig, 'ade_t50')} | {_fmt_metric(exp, 'ade_t50')} |",
        f"| ADE t100 raw diag mean [CI] | {_fmt_metric(orig, 'ade_t100_raw_frame_diagnostic')} | {_fmt_metric(exp, 'ade_t100_raw_frame_diagnostic')} |",
        f"| ADE hard/failure mean [CI] | {_fmt_metric(orig, 'ade_hard_failure')} | {_fmt_metric(exp, 'ade_hard_failure')} |",
        f"| ADE easy degradation mean [CI] | {_fmt_metric(orig, 'ade_easy_degradation')} | {_fmt_metric(exp, 'ade_easy_degradation')} |",
        f"| FDE t50 mean [CI] | {_fmt_metric(orig, 'fde_t50')} | {_fmt_metric(exp, 'fde_t50')} |",
        "",
        "## Validation-Selected Combined Seed",
        "",
        f"- selected: `{selected.get('selected')}`",
        f"- seed: `{selected.get('seed')}`",
        f"- base_seed: `{selected.get('base_seed')}`",
        f"- validation_score: `{selected.get('validation_score', 0.0):.6f}`",
        f"- test ADE all/t50/hard/easy: `{selected.get('test_ade_all', 0.0):.6f}` / `{selected.get('test_ade_t50', 0.0):.6f}` / `{selected.get('test_ade_hard_failure', 0.0):.6f}` / `{selected.get('test_ade_easy_degradation', 0.0):.6f}`",
        f"- test FDE t50: `{selected.get('test_fde_t50', 0.0):.6f}`",
        "",
        "## Per-Seed Test Metrics",
        "",
        "| seed | source | base_seed | val_score | ADE all | ADE t50 | ADE t100 raw | ADE hard | ADE easy degr | FDE t50 | switch |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["seed_table"]:
        lines.append(
            f"| {row['seed']} | `{row['selector_source']}` | {row['base_seed']} | {row['validation_score']:.6f} | {row['ade_all']:.6f} | {row['ade_t50']:.6f} | {row['ade_t100_raw_frame_diagnostic']:.6f} | {row['ade_hard_failure']:.6f} | {row['ade_easy_degradation']:.6f} | {row['fde_t50']:.6f} | {row['switch_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Domain Instability",
            "",
            f"- original negative t50 slices: `{payload['original_domain_instability']['negative_t50_slice_count']}`",
            f"- expanded negative t50 slices: `{payload['expanded_domain_instability']['negative_t50_slice_count']}`",
            "",
            "Worst expanded t+50 slices:",
            "",
            "| seed | domain | rows | all | t50 | hard/failure | easy degradation | switch |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["expanded_domain_instability"]["worst_t50_slices"]:
        lines.append(
            f"| {row['seed']} | `{row['domain']}` | {row['rows']} | {row['all_improvement']:.6f} | {row['t50_improvement']:.6f} | {row['hard_failure_improvement']:.6f} | {row['easy_degradation']:.6f} | {row['switch_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This stage expands selector random seeds while reusing the three cached Stage42-N base checkpoints; that reuse is deliberate and explicitly disclosed.",
            "- If the combined ADE t+50 CI lower bound is positive, Stage42-P's t+50 selector has stronger seed-level evidence.",
            "- If it remains negative, the open blocker is not row bootstrap but train-seed/domain instability, and the next step should alter the policy family or training objective.",
        ]
    )
    write_md(REPORT_MD, lines)


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-IH Gate",
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


def _refresh_readmes_and_state(payload: Mapping[str, Any]) -> None:
    exp = payload["expanded_summary"]
    gate = payload["stage42_ih_gate"]
    selected = payload["validation_selected"]
    lines = [
        "## Stage42-IH T50 Gain/Harm Seed Expansion",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        f"- combined seeds: `{payload['combined_seed_count']}`",
        f"- expanded ADE t50 mean / CI low: `{exp['ade_t50']['mean']:.6f}` / `{exp['ade_t50']['ci_low']:.6f}`",
        f"- expanded FDE t50 mean / CI low: `{exp['fde_t50']['mean']:.6f}` / `{exp['fde_t50']['ci_low']:.6f}`",
        f"- expanded ADE hard/failure mean: `{exp['ade_hard_failure']['mean']:.6f}`",
        f"- expanded ADE easy degradation mean: `{exp['ade_easy_degradation']['mean']:.6f}`",
        f"- validation-selected seed: `{selected.get('seed')}` with ADE t50 `{selected.get('test_ade_t50', 0.0):.6f}`",
        "- boundary: dataset-local/raw-frame 2.5D only; no metric/seconds claim, no Stage5C, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, MASTER_README]:
        _replace_section(path, SECTION, lines)
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage42_ih_t50_gain_harm_seed_expansion"
    state["current_verdict"] = gate["verdict"]
    stage42 = state.setdefault("stage42", {})
    stage42["stage_ih_t50_gain_harm_seed_expansion"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": gate["verdict"],
        "gates": f"{gate['passed']}/{gate['total']}",
        "combined_seed_count": payload["combined_seed_count"],
        "expanded_ade_t50_mean": exp["ade_t50"]["mean"],
        "expanded_ade_t50_ci_low": exp["ade_t50"]["ci_low"],
        "expanded_fde_t50_mean": exp["fde_t50"]["mean"],
        "expanded_fde_t50_ci_low": exp["fde_t50"]["ci_low"],
        "expanded_ade_hard_failure_mean": exp["ade_hard_failure"]["mean"],
        "expanded_ade_easy_degradation_mean": exp["ade_easy_degradation"]["mean"],
        "validation_selected": selected,
        "claim_boundary": payload["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(RESEARCH_STATE, _jsonable(state))


def run_stage42_t50_gain_harm_seed_expansion() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, _jsonable(payload))
    _write_report(payload)
    _write_gate(payload["stage42_ih_gate"])
    _refresh_readmes_and_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_t50_gain_harm_seed_expansion()
    print(json.dumps(_jsonable(result["stage42_ih_gate"]), ensure_ascii=False, indent=2))
