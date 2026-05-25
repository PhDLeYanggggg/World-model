from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_bounded_neural_blend_dynamics as blend
from src import stage41_source_level_validation_repair as slv
from src import stage41_teacher_guided_proposal as tgp


OUT_DIR = Path("outputs/stage41_external_split")
REPORT_JSON = OUT_DIR / "stage41_pure_ucy_source_validation.json"
REPORT_MD = OUT_DIR / "stage41_pure_ucy_source_validation.md"


DUPLICATE_LIKE_BLOCKERS = [
    "TrajNet/Train/crowds/crowds_zara03.txt and UCY/zara03/crowds_zara03.txt are duplicate-like zara03 sources; they are not counted as independent UCY validation sources."
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _append_ledger(step: str, status: str, started: float, inputs: Sequence[str | Path], outputs: Sequence[str | Path]) -> None:
    ensure_dir(OUT_DIR)
    entry = {
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "step": step,
        "source": "fresh_run",
        "status": status,
        "wall_time_s": time.perf_counter() - started,
        "input_hash": _combined_hash(inputs),
        "output_hash": _combined_hash(outputs),
        "git_commit": _git_commit(),
    }
    with (OUT_DIR / "run_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


def _source_key_array(data: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([slv._source_key(src) for src in data["labels"]["source_file"].astype(str)], dtype="U256")


def _subset_value(value: Any, mask: np.ndarray, n: int) -> Any:
    if isinstance(value, dict):
        return {k: _subset_value(v, mask, n) for k, v in value.items()}
    if isinstance(value, np.ndarray) and len(value.shape) > 0 and value.shape[0] == n:
        return value[mask]
    return value


def _subset_bundle(data: Mapping[str, Any], mask: np.ndarray) -> dict[str, Any]:
    n = len(data["labels"]["horizon"])
    return {k: _subset_value(v, mask, n) for k, v in data.items()}


def _strict_positive(metrics: Mapping[str, Any]) -> bool:
    return bool(
        metrics.get("all_improvement", 0.0) > 0
        and metrics.get("t50_improvement", 0.0) > 0
        and metrics.get("t100_improvement", 0.0) > 0
        and metrics.get("hard_failure_improvement", 0.0) > 0
        and metrics.get("easy_degradation", 1.0) <= 0.02
        and metrics.get("collision_delta_vs_floor_005", 1.0) <= blend.TEST_COLLISION_CEILING
    )


def _is_pure_ucy_source(source_key: str) -> bool:
    return source_key.startswith("UCY/")


def _source_inventory(*splits: tuple[str, Mapping[str, Any]]) -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    for split, data in splits:
        keys = _source_key_array(data)
        horizon = data["labels"]["horizon"].astype(int)
        rows = []
        for source in sorted(set(keys.tolist())):
            mask = keys == source
            rows.append(
                {
                    "source": source,
                    "rows": int(np.sum(mask)),
                    "t10": int(np.sum(mask & (horizon == 10))),
                    "t25": int(np.sum(mask & (horizon == 25))),
                    "t50": int(np.sum(mask & (horizon == 50))),
                    "t100": int(np.sum(mask & (horizon == 100))),
                    "is_pure_ucy_source": _is_pure_ucy_source(source),
                }
            )
        inventory[split] = rows
    return inventory


def _target_source_locations(val: Mapping[str, Any], test: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    locations: dict[str, dict[str, Any]] = {}
    for split, data in [("val", val), ("test", test)]:
        keys = _source_key_array(data)
        for source in sorted(set(keys.tolist())):
            if not _is_pure_ucy_source(source):
                continue
            mask = keys == source
            if np.any(mask):
                locations[source.replace("/", "__").replace(".", "_")] = {
                    "split": split,
                    "source": source,
                    "rows": int(np.sum(mask)),
                }
    return locations


def _select_non_ucy_policy(val: Mapping[str, Any]) -> dict[str, Any]:
    domain = val["labels"]["domain"].astype(str)
    non_ucy = domain != "UCY"
    selected_on = _subset_bundle(val, non_ucy)
    selection = blend._select_safe_switch_policy(selected_on, selected_on)
    return {
        "selection": selection,
        "selected_policy": (selection.get("selected") or {}).get("policy") or {},
        "selected_rows": int(np.sum(non_ucy)),
        "selected_domains": sorted(set(domain[non_ucy].tolist())),
    }


def _evaluate_target(data: Mapping[str, Any], source_key: str, policy: Mapping[str, Any]) -> dict[str, Any]:
    keys = _source_key_array(data)
    mask = keys == source_key
    target = _subset_bundle(data, mask)
    ev = blend._evaluate_blend(target, policy)
    metrics = dict(ev["metrics"])
    return {
        "source": source_key,
        "rows": int(np.sum(mask)),
        "metrics": metrics,
        "alpha_mean": float(metrics.get("alpha_mean", 0.0)),
        "switch_rate": float(metrics.get("switch_rate", 0.0)),
        "strict_positive": _strict_positive(metrics),
        "floor_stats": ev["floor_stats"],
        "blend_stats": ev["blend_stats"],
    }


def run_pure_ucy_source_validation() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    checkpoint, teacher_policy, min_sep = blend._load_frozen_model()
    val = blend._bundle("val", checkpoint, teacher_policy, min_sep)
    test = blend._bundle("test", checkpoint, teacher_policy, min_sep)
    train = blend._bundle("train", checkpoint, teacher_policy, min_sep)
    inventory = _source_inventory(("train", train), ("val", val), ("test", test))
    locations = _target_source_locations(val, test)
    selection = _select_non_ucy_policy(val)
    policy = selection["selected_policy"]
    targets: dict[str, Any] = {}
    for name, loc in locations.items():
        source_key = loc["source"]
        split_data = val if loc["split"] == "val" else test
        targets[name] = {
            **loc,
            **_evaluate_target(split_data, source_key, policy),
        }
    pass_count = sum(1 for row in targets.values() if row.get("strict_positive"))
    result = {
        "source": "fresh_run",
        "protocol": "pure_ucy_source_heldout_validation_without_target_threshold_tuning",
        "checkpoint": checkpoint,
        "policy_selected_on": "non_ucy_validation_rows_only",
        "non_ucy_selection": selection,
        "source_inventory": inventory,
        "target_source_locations": locations,
        "target_results": targets,
        "pure_ucy_source_heldout_gate": bool(pass_count == len(targets) and len(targets) >= 2),
        "pure_ucy_three_way_train_val_test_gate": False,
        "remaining_blocker": (
            "This validates frozen-policy UCY source-heldout behavior, but it is not a pure UCY-only retrain/select/test protocol because the frozen model and safety floor were trained on mixed external train data."
        ),
        "duplicate_blockers": DUPLICATE_LIKE_BLOCKERS,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_labels_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "target_source_excluded_from_policy_selection": True,
            "policy_selected_on_non_ucy_validation_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "caveat": "This is a pure-UCY source-heldout frozen-policy check, not a strict pure-UCY-only train/val/test retraining protocol. Coordinates remain dataset-local raw-frame 2.5D.",
    }
    write_json(REPORT_JSON, _jsonable(result))
    lines = [
        "# Stage41 Pure UCY Source-Heldout Validation",
        "",
        "- source: `fresh_run`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "- metric/seconds claim: `False`",
        f"- policy selected on: `{result['policy_selected_on']}`",
        f"- pure UCY source-heldout gate: `{result['pure_ucy_source_heldout_gate']}`",
        f"- pure UCY three-way train/val/test gate: `{result['pure_ucy_three_way_train_val_test_gate']}`",
        f"- remaining blocker: `{result['remaining_blocker']}`",
        "",
        "| source | split | rows | all | t50 | t100 | hard/failure | easy | switch | pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, row in sorted(targets.items()):
        m = row["metrics"]
        lines.append(
            f"| `{row['source']}` | `{row['split']}` | {row['rows']} | {float(m.get('all_improvement', 0.0)):.4f} | "
            f"{float(m.get('t50_improvement', 0.0)):.4f} | {float(m.get('t100_improvement', 0.0)):.4f} | "
            f"{float(m.get('hard_failure_improvement', 0.0)):.4f} | {float(m.get('easy_degradation', 0.0)):.4f} | "
            f"{float(m.get('switch_rate', 0.0)):.4f} | `{row['strict_positive']}` |"
        )
    lines.extend(
        [
            "",
            f"- non-UCY validation rows used for policy selection: `{selection['selected_rows']}`",
            f"- non-UCY validation domains: `{selection['selected_domains']}`",
            f"- duplicate blockers: `{DUPLICATE_LIKE_BLOCKERS}`",
            f"- no leakage: `{result['no_leakage']}`",
            "",
            "## UCY Source Inventory",
            "",
            "| split | source | rows | t50 | t100 |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for split, rows in inventory.items():
        for row in rows:
            if row["is_pure_ucy_source"]:
                lines.append(f"| `{split}` | `{row['source']}` | {row['rows']} | {row['t50']} | {row['t100']} |")
    lines.extend(
        [
            "",
            result["caveat"],
        ]
    )
    write_md(REPORT_MD, lines)
    return result


def main_pure_ucy_source_validation() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_pure_ucy_source_validation()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_pure_ucy_source_validation",
            status,
            started,
            [Path("outputs/stage41_fresh_confirmation/stage41_bounded_neural_blend_dynamics.json")],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_pure_ucy_source_validation()
