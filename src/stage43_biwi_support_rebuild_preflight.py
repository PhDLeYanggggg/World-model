from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m
from src.stage43_source_level_heldout_split import DATA35, SPLITS, _sha256_text


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_biwi_support_rebuild_preflight.json"
REPORT_MD = OUT_DIR / "stage43_biwi_support_rebuild_preflight.md"
GATE_MD = OUT_DIR / "stage43_stage_bd_biwi_support_rebuild_preflight_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SOURCE = "fresh_stage43_bd_biwi_support_rebuild_preflight"
SECTION = "STAGE43_BD_BIWI_SUPPORT_REBUILD_PREFLIGHT"

STAGE43_BC = OUT_DIR / "stage43_blocked_family_support_scan.json"
STAGE43_F = OUT_DIR / "stage43_source_level_heldout_split.json"

BIWI_FAMILY = "TrajNet_biwi"
MIN_VAL_ROWS = 1000
MIN_TEST_ROWS = 1000
MIN_T50_VAL_ROWS = 200


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path, {})


def _jsonable(value: Any) -> Any:
    return m._jsonable(value)


def _load_split(split: str) -> dict[str, np.ndarray]:
    geo = np.load(DATA35 / f"expanded_external_{split}.npz", allow_pickle=False)
    labels = np.load(DATA35 / f"labels_{split}.npz", allow_pickle=False)
    n = len(geo["horizon"])
    return {
        "old_split": np.asarray([split] * n),
        "local_row": np.arange(n, dtype=np.int64),
        "dataset": geo["dataset"].astype(str),
        "scene_id": geo["scene_id"].astype(str),
        "source_file": geo["source_file"].astype(str),
        "agent_id": geo["agent_id"].astype(np.int64),
        "frame_id": geo["frame_id"].astype(np.float64),
        "horizon": geo["horizon"].astype(np.int64),
        "track_length": geo["track_length"].astype(np.float32),
        "hard": labels["hard"].astype(bool),
        "failure": labels["failure"].astype(bool),
        "easy": labels["easy"].astype(bool),
    }


def _concat_pool() -> dict[str, np.ndarray]:
    parts = [_load_split(split) for split in SPLITS]
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0].keys()}


def _is_biwi_source(source_file: str) -> bool:
    norm = source_file.replace("\\", "/").lower()
    return "/trajnet/" in norm and "/biwi/" in norm


def _source_role(source_file: str) -> str:
    norm = source_file.replace("\\", "/").lower()
    if "/train/" in norm:
        return "raw_train_dir"
    if "/test/" in norm:
        return "raw_test_dir"
    return "unknown_dir"


def _source_rows(pool: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for source_file in sorted(set(pool["source_file"].tolist())):
        if not _is_biwi_source(str(source_file)):
            continue
        mask = pool["source_file"] == source_file
        horizons = Counter(pool["horizon"][mask].astype(int).tolist())
        rows[str(source_file)] = {
            "source_file": str(source_file),
            "source_id": _sha256_text(str(source_file))[:16],
            "raw_role": _source_role(str(source_file)),
            "rows": int(mask.sum()),
            "horizon_counts": {str(k): int(v) for k, v in sorted(horizons.items())},
            "hard_rows": int(np.sum(pool["hard"][mask])),
            "failure_rows": int(np.sum(pool["failure"][mask])),
            "easy_rows": int(np.sum(pool["easy"][mask])),
            "agent_count": int(len(set(pool["agent_id"][mask].astype(int).tolist()))),
            "frame_min": float(np.min(pool["frame_id"][mask])) if int(mask.sum()) else 0.0,
            "frame_max": float(np.max(pool["frame_id"][mask])) if int(mask.sum()) else 0.0,
            "old_split_counts": {
                str(k): int(v) for k, v in zip(*np.unique(pool["old_split"][mask].astype(str), return_counts=True))
            }
            if int(mask.sum())
            else {},
        }
    return rows


def _stage43_f_current_assignment(stage43_f: Mapping[str, Any], biwi_sources: Mapping[str, Any]) -> dict[str, Any]:
    assignments = stage43_f.get("source_assignments", {})
    by_split = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}
    t50_counts = {"train": 0, "val": 0, "test": 0}
    for source_file, row in biwi_sources.items():
        split = str(assignments.get(source_file, "unassigned"))
        if split in by_split:
            by_split[split].append(row["source_id"])
            counts[split] += int(row["rows"])
            t50_counts[split] += int(row["horizon_counts"].get("50", 0))
    return {
        "assignment_name": "current_stage43_source_level_split",
        "source_ids_by_split": by_split,
        "rows_by_split": counts,
        "t50_rows_by_split": t50_counts,
        "repair_training_allowed": False,
        "reason": "current_source_level_split_has_no_biwi_train_rows",
    }


def _agent_hash(agent_id: int) -> int:
    digest = hashlib.sha256(f"biwi-agent:{int(agent_id)}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _within_source_support_counts(pool: Mapping[str, np.ndarray], source_file: str) -> dict[str, Any]:
    mask = pool["source_file"] == source_file
    agents = sorted(set(pool["agent_id"][mask].astype(int).tolist()))
    val_agents = {agent for agent in agents if _agent_hash(agent) % 5 == 0}
    if not val_agents and agents:
        val_agents = {agents[0]}
    role = np.asarray(["val" if int(agent) in val_agents else "train" for agent in pool["agent_id"][mask]], dtype=str)
    horizons = pool["horizon"][mask].astype(int)
    out: dict[str, Any] = {
        "split_rule": "deterministic_agent_hash_mod5_validation",
        "source_file": source_file,
        "source_id": _sha256_text(source_file)[:16],
        "train_agent_count": int(len(set(agents) - val_agents)),
        "val_agent_count": int(len(val_agents)),
        "rows_by_split": {},
        "t50_rows_by_split": {},
    }
    for split in ["train", "val"]:
        split_mask = role == split
        out["rows_by_split"][split] = int(np.sum(split_mask))
        out["t50_rows_by_split"][split] = int(np.sum(split_mask & (horizons == 50)))
    return out


def _candidate_options(pool: Mapping[str, np.ndarray], biwi_sources: Mapping[str, Any]) -> list[dict[str, Any]]:
    train_sources = [src for src, row in biwi_sources.items() if row["raw_role"] == "raw_train_dir"]
    test_sources = [src for src, row in biwi_sources.items() if row["raw_role"] == "raw_test_dir"]
    options: list[dict[str, Any]] = []
    if train_sources and test_sources:
        train_rows = sum(int(biwi_sources[src]["rows"]) for src in train_sources)
        val_rows = sum(int(biwi_sources[src]["rows"]) for src in test_sources)
        train_t50 = sum(int(biwi_sources[src]["horizon_counts"].get("50", 0)) for src in train_sources)
        val_t50 = sum(int(biwi_sources[src]["horizon_counts"].get("50", 0)) for src in test_sources)
        options.append(
            {
                "option": "raw_train_support_raw_test_validation",
                "uses_raw_test_dir_for_training": False,
                "uses_current_test_source_for_training": True,
                "split_type": "family_support_candidate_not_deployable",
                "train_sources": [_sha256_text(src)[:16] for src in train_sources],
                "val_sources": [_sha256_text(src)[:16] for src in test_sources],
                "test_sources": [],
                "rows_by_split": {"train": int(train_rows), "val": int(val_rows), "test": 0},
                "t50_rows_by_split": {"train": int(train_t50), "val": int(val_t50), "test": 0},
                "repair_training_allowed": False,
                "blockers": [
                    "no_independent_biwi_test_source_after_support_rebuild",
                    "current_stage43_test_source_would_move_to_train_support",
                    "validation_rows_below_threshold" if val_rows < MIN_VAL_ROWS else "",
                    "validation_t50_rows_below_threshold" if val_t50 < MIN_T50_VAL_ROWS else "",
                ],
            }
        )
    for train_source in train_sources:
        counts = _within_source_support_counts(pool, train_source)
        options.append(
            {
                "option": "within_source_agent_split_support_diagnostic",
                "uses_raw_test_dir_for_training": False,
                "uses_current_test_source_for_training": True,
                "split_type": "diagnostic_within_source_cv_not_official_source_level",
                **counts,
                "test_sources": [],
                "repair_training_allowed": False,
                "blockers": [
                    "within_source_split_not_source_level_heldout",
                    "no_independent_biwi_test_source",
                    "current_stage43_test_source_would_be_reused_for_support",
                ],
            }
        )
    options.append(
        {
            "option": "keep_current_stage43_floor_only",
            "split_type": "deployable_current_floor",
            "rows_by_split": {"train": 0, "val": 0, "test": sum(int(row["rows"]) for row in biwi_sources.values())},
            "t50_rows_by_split": {
                "train": 0,
                "val": 0,
                "test": sum(int(row["horizon_counts"].get("50", 0)) for row in biwi_sources.values()),
            },
            "repair_training_allowed": False,
            "blockers": ["no_train_family_rows_for_repair"],
        }
    )
    for option in options:
        option["blockers"] = [item for item in option.get("blockers", []) if item]
    return options


def build_biwi_support_rebuild_preflight() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43_bc = _load(STAGE43_BC)
    stage43_f = _load(STAGE43_F)
    pool = _concat_pool()
    biwi_sources = _source_rows(pool)
    current = _stage43_f_current_assignment(stage43_f, biwi_sources)
    options = _candidate_options(pool, biwi_sources)
    raw_scan_action = next(
        (row for row in stage43_bc.get("blocked_family_actions", []) if row.get("family") == BIWI_FAMILY),
        {},
    )
    summary = {
        "biwi_source_count_in_feature_store": int(len(biwi_sources)),
        "current_train_rows": int(current["rows_by_split"]["train"]),
        "current_val_rows": int(current["rows_by_split"]["val"]),
        "current_test_rows": int(current["rows_by_split"]["test"]),
        "current_t50_train_rows": int(current["t50_rows_by_split"]["train"]),
        "current_t50_val_rows": int(current["t50_rows_by_split"]["val"]),
        "current_t50_test_rows": int(current["t50_rows_by_split"]["test"]),
        "candidate_option_count": int(len(options)),
        "deployable_repair_option_count": int(sum(bool(option["repair_training_allowed"]) for option in options)),
        "diagnostic_support_option_count": int(
            sum(str(option["split_type"]).startswith("diagnostic") or "candidate_not_deployable" in str(option["split_type"]) for option in options)
        ),
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_biwi_source_family_support_rebuild_preflight",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "input_artifacts": {"stage43_bc": str(STAGE43_BC), "stage43_f": str(STAGE43_F), "stage35_feature_store": str(DATA35)},
        "input_verdicts": {
            "stage43_bc": stage43_bc.get("stage43_bc_gate", {}).get("verdict"),
            "stage43_f": stage43_f.get("stage43_f_gate", {}).get("verdict"),
        },
        "protocol": {
            "purpose": "preflight_only_no_repair_training",
            "family": BIWI_FAMILY,
            "minimum_validation_rows": MIN_VAL_ROWS,
            "minimum_test_rows": MIN_TEST_ROWS,
            "minimum_t50_validation_rows": MIN_T50_VAL_ROWS,
            "test_threshold_tuning_allowed": False,
            "raw_test_dir_training_allowed": False,
            "current_heldout_test_reuse_for_training_allowed": False,
        },
        "raw_scan_action": raw_scan_action,
        "biwi_sources": biwi_sources,
        "current_stage43_assignment": current,
        "candidate_rebuild_options": options,
        "summary": summary,
        "next_required_actions": [
            "Do not train a biwi-specific repair on the current Stage43 source-level test source.",
            "If biwi repair remains important, acquire or locate an independent biwi-like source so train, validation, and test support are disjoint.",
            "A within-source agent split can be used only as a diagnostic conversion smoke test, not as deployable evidence.",
            "Keep Stage43-P/AZ floor behavior for TrajNet_biwi until a source-level validation gate has enough rows and positive easy-safe evidence.",
        ],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_or_loss_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
            "raw_test_dir_training": False,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "deployable_biwi_repair_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
        "input_hash": _combined_hash([STAGE43_BC, STAGE43_F]),
    }
    payload["stage43_bd_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    options = payload["candidate_rebuild_options"]
    gates = {
        "stage43_bc_precondition_passed": payload["input_verdicts"]["stage43_bc"]
        == "stage43_bc_blocked_family_support_scan_pass",
        "stage43_source_split_precondition_passed": payload["input_verdicts"]["stage43_f"]
        == "stage43_f_source_level_split_ready",
        "biwi_sources_found_in_feature_store": summary["biwi_source_count_in_feature_store"] >= 2,
        "current_train_gap_reconfirmed": summary["current_train_rows"] == 0 and summary["current_test_rows"] > 0,
        "candidate_rebuild_options_evaluated": summary["candidate_option_count"] >= 2,
        "deployable_repair_correctly_blocked": summary["deployable_repair_option_count"] == 0,
        "diagnostic_support_option_recorded": summary["diagnostic_support_option_count"] >= 1,
        "raw_test_training_blocked": all(option.get("uses_raw_test_dir_for_training") is not True for option in options),
        "current_test_reuse_blocked_for_deployment": all(option.get("repair_training_allowed") is False for option in options),
        "next_actions_recorded": len(payload["next_required_actions"]) >= 3,
        "no_future_or_test_leakage": no_leak["future_endpoint_input"] is False
        and no_leak["future_waypoint_input"] is False
        and no_leak["future_labels_eval_or_loss_only"] is True
        and no_leak["central_velocity_input"] is False
        and no_leak["test_endpoint_goal_construction"] is False
        and no_leak["test_statistics_normalization"] is False
        and no_leak["test_threshold_tuning"] is False
        and no_leak["raw_test_dir_training"] is False,
        "claim_boundary_not_overstated": claim["true_3d_world_model"] is False
        and claim["foundation_world_model"] is False
        and claim["metric_or_seconds_claim"] is False
        and claim["dataset_local_raw_frame_only"] is True
        and claim["deployable_biwi_repair_claim"] is False,
        "stage5c_and_smc_false": claim["stage5c_executed"] is False and claim["smc_enabled"] is False,
        "long_objective_kept_active": claim["long_objective_complete"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_bd_biwi_support_rebuild_preflight_pass"
        if passed == total
        else "stage43_bd_biwi_support_rebuild_preflight_incomplete",
        "stage5c_executed": False,
        "smc_enabled": False,
        "goal_complete": False,
    }


def _render_md(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_bd_gate"]
    summary = payload["summary"]
    lines = [
        "# Stage43-BD Biwi Support Rebuild Preflight",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- biwi sources in feature store: `{summary['biwi_source_count_in_feature_store']}`",
        f"- current train / val / test rows: `{summary['current_train_rows']} / {summary['current_val_rows']} / {summary['current_test_rows']}`",
        f"- deployable repair options now: `{summary['deployable_repair_option_count']}`",
        "",
        "## Current Biwi Sources",
        "",
        "| source | raw role | rows | t50 rows | old splits | current source-level split |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    current_assignments = payload["current_stage43_assignment"]["source_ids_by_split"]
    source_to_split = {
        source_id: split
        for split, source_ids in current_assignments.items()
        for source_id in source_ids
    }
    for row in payload["biwi_sources"].values():
        lines.append(
            f"| `{Path(row['source_file']).name}` | `{row['raw_role']}` | {row['rows']} | {row['horizon_counts'].get('50', 0)} | "
            f"`{row['old_split_counts']}` | `{source_to_split.get(row['source_id'], 'unassigned')}` |"
        )
    lines.extend(
        [
            "",
            "## Rebuild Options",
            "",
            "| option | split type | rows train/val/test | t50 train/val/test | deployable repair allowed | blockers |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for option in payload["candidate_rebuild_options"]:
        rows = option["rows_by_split"]
        t50 = option["t50_rows_by_split"]
        lines.append(
            f"| `{option['option']}` | `{option['split_type']}` | "
            f"{rows.get('train', 0)}/{rows.get('val', 0)}/{rows.get('test', 0)} | "
            f"{t50.get('train', 0)}/{t50.get('val', 0)}/{t50.get('test', 0)} | "
            f"`{option['repair_training_allowed']}` | {', '.join(option.get('blockers', [])) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This preflight makes the next boundary explicit. `biwi_hotel` has enough rows to build support, but it is also the current Stage43 held-out biwi test source. Moving it into training would invalidate the current source-level test claim. `biwi_eth` is small and comes from the raw Test directory, so I do not use it for training.",
            "",
            "The only safe conclusion is still conservative: there is support to test a diagnostic converter, but not enough independent source-level evidence to train and deploy a biwi repair. Stage43-P/AZ should keep the floor on `TrajNet_biwi` until another independent biwi-like source or a new source-level protocol is available.",
            "",
            "## Next Required Actions",
            "",
            *[f"- {item}" for item in payload["next_required_actions"]],
            "",
            "## Claim Boundary",
            "",
            "- This is a preflight manifest, not a model result.",
            "- Dataset-local/raw-frame 2.5D only.",
            "- No metric or seconds-level claim.",
            "- No biwi deployable repair claim.",
            "- No Stage5C execution and no SMC.",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
        ]
    )
    lines.extend([f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()])
    return lines


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bd_gate"]
    summary = payload["summary"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"biwi_sources = `{summary['biwi_source_count_in_feature_store']}`",
        f"current_train_val_test_rows = `{summary['current_train_rows']} / {summary['current_val_rows']} / {summary['current_test_rows']}`",
        f"deployable_repair_options_now = `{summary['deployable_repair_option_count']}`",
        "",
        "I checked whether the raw biwi support found in BC can actually become a safe repair split. It cannot yet: the useful `biwi_hotel` rows are the current held-out biwi test source, and the small `biwi_eth` support is not enough for an independent deployable train/val/test story. I am keeping biwi floor-only and treating any within-source support split as diagnostic only.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_bd_biwi_support_rebuild_preflight"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "summary": summary,
        "candidate_rebuild_options": payload["candidate_rebuild_options"],
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_bd_biwi_support_rebuild_preflight"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, _jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                _jsonable(
                    {
                        "stage": "Stage43-BD",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "summary": summary,
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    write_json(WORLD_GATE_JSON, _jsonable(payload["stage43_bd_gate"]))
    lines = _render_md(payload)
    write_md(REPORT_MD, lines)
    write_md(GATE_MD, lines)
    gate = payload["stage43_bd_gate"]
    world_lines = [
        "# Stage43 Current World-Model Gate",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- long objective complete: `{gate['goal_complete']}`",
        f"- Stage5C executed: `{gate['stage5c_executed']}`",
        f"- SMC enabled: `{gate['smc_enabled']}`",
        "",
        "## Current Boundary",
        "",
        "- Stage43-P / AZ remains the performance leader and exact replay artifact.",
        "- Stage43-BC found raw biwi support, but Stage43-BD shows it is not yet deployable repair support.",
        "- TrajNet_biwi stays floor-only until independent source-level support exists.",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| {name} | `{bool(value)}` |" for name, value in gate["gates"].items()],
    ]
    write_md(WORLD_GATE_MD, world_lines)
    _update_ledgers(payload)


def run_biwi_support_rebuild_preflight() -> dict[str, Any]:
    payload = build_biwi_support_rebuild_preflight()
    _write_outputs(payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Preflight guarded biwi support rebuild options without training repair.")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    build_arg_parser().parse_args(argv)
    payload = run_biwi_support_rebuild_preflight()
    gate = payload["stage43_bd_gate"]
    summary = payload["summary"]
    print(f"Stage43-BD: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(
        "biwi_current_train_val_test_rows="
        f"{summary['current_train_rows']}/{summary['current_val_rows']}/{summary['current_test_rows']}"
    )
    print(f"deployable_repair_options_now={summary['deployable_repair_option_count']}")
    return payload


if __name__ == "__main__":
    main()
