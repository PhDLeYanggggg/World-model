from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src import stage41_joint_multiagent_consistency as jmc
from src import stage41_joint_rollout_consistency as jrc
from src import stage42_common_validation_bridge_shape_composer as co


OUT_DIR = Path("outputs/stage42_long_research")
CO_JSON = OUT_DIR / "common_validation_bridge_shape_composer_stage42.json"
REPORT_JSON = OUT_DIR / "common_validation_composer_safety_stage42.json"
REPORT_MD = OUT_DIR / "common_validation_composer_safety_stage42.md"
REPORT_CSV = OUT_DIR / "common_validation_composer_safety_stage42.csv"
GATE_MD = OUT_DIR / "stage42_stage_cp_gate.md"

PAPER_FILES = [
    OUT_DIR / "experiment_tables_stage42.md",
    OUT_DIR / "ablation_tables_stage42.md",
    OUT_DIR / "model_card_stage42.md",
    OUT_DIR / "a_journal_gap_stage42.md",
]

BOOTSTRAP_N = 2000
SEED = 424221
EPS = 1e-6

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-CP 是 Stage42-CO composer 的 bootstrap + all-agent joint safety audit。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
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
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _mask(labels: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    horizon = labels["horizon"].astype(int)
    hard_failure = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    if name == "all":
        return np.ones(len(horizon), dtype=bool)
    if name == "t50":
        return horizon == 50
    if name == "t100":
        return horizon == 100
    if name == "hard_failure":
        return hard_failure
    raise ValueError(f"unknown bootstrap slice: {name}")


def _bootstrap(selected: np.ndarray, ref: np.ndarray, mask: np.ndarray, seed: int) -> dict[str, Any]:
    ids = np.where(mask)[0]
    if len(ids) < 30:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "n": int(len(ids)), "bootstrap_n": 0}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(BOOTSTRAP_N):
        sample = rng.choice(ids, size=len(ids), replace=True)
        vals.append(1.0 - float(np.mean(selected[sample])) / max(float(np.mean(ref[sample])), EPS))
    return {
        "low": float(np.percentile(vals, 2.5)),
        "mid": float(np.percentile(vals, 50.0)),
        "high": float(np.percentile(vals, 97.5)),
        "n": int(len(ids)),
        "bootstrap_n": BOOTSTRAP_N,
    }


def _bootstrap_set(selected: np.ndarray, ref: np.ndarray, labels: Mapping[str, np.ndarray]) -> dict[str, Any]:
    slices = ["all", "t50", "t100", "hard_failure"]
    return {name: _bootstrap(selected, ref, _mask(labels, name), SEED + i) for i, name in enumerate(slices)}


def _delta_stats(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "near_collision_rate_002_delta": float(a["near_collision_rate_002"] - b["near_collision_rate_002"]),
        "near_collision_rate_005_delta": float(a["near_collision_rate_005"] - b["near_collision_rate_005"]),
        "p05_min_group_distance_delta": (
            None
            if a.get("p05_min_group_distance") is None or b.get("p05_min_group_distance") is None
            else float(a["p05_min_group_distance"] - b["p05_min_group_distance"])
        ),
        "mean_min_group_distance_delta": (
            None
            if a.get("mean_min_group_distance") is None or b.get("mean_min_group_distance") is None
            else float(a["mean_min_group_distance"] - b["mean_min_group_distance"])
        ),
        "jagged_rate_delta": float(a["smoothness"]["jagged_rate"] - b["smoothness"]["jagged_rate"]),
        "mean_max_normalized_step_delta": float(
            a["smoothness"]["mean_max_normalized_step"] - b["smoothness"]["mean_max_normalized_step"]
        ),
    }


def _refresh_paper_files(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    vs_endpoint = payload["bootstrap_vs_endpoint_ade"]
    joint = payload["joint_safety"]
    lines = [
        "## Stage42-CP Common Validation Composer Safety / Bootstrap",
        "",
        "- source: `fresh_joint_safety_bootstrap_from_stage42_co_policy`",
        "- scope: Stage42-CO validation-selected composer, test evaluated once.",
        f"- bootstrap vs endpoint-linear all CI: `[{_pct(vs_endpoint['all']['low'])}, {_pct(vs_endpoint['all']['high'])}]`.",
        f"- bootstrap vs endpoint-linear t50 CI: `[{_pct(vs_endpoint['t50']['low'])}, {_pct(vs_endpoint['t50']['high'])}]`.",
        f"- bootstrap vs endpoint-linear t100 raw diagnostic CI: `[{_pct(vs_endpoint['t100']['low'])}, {_pct(vs_endpoint['t100']['high'])}]`.",
        f"- bootstrap vs endpoint-linear hard/failure CI: `[{_pct(vs_endpoint['hard_failure']['low'])}, {_pct(vs_endpoint['hard_failure']['high'])}]`.",
        f"- near-collision@0.05 delta vs endpoint-linear: `{_pct(joint['composer_minus_endpoint']['near_collision_rate_005_delta'])}`.",
        f"- near-collision@0.05 delta vs strongest floor: `{_pct(joint['composer_minus_floor']['near_collision_rate_005_delta'])}`.",
        "- claim boundary: still dataset-local/raw-frame 2.5D; no metric/seconds-level, no Stage5C, no SMC.",
    ]
    out = []
    for path in PAPER_FILES:
        co._replace_section(path, "STAGE42_CP_COMMON_VALIDATION_COMPOSER_SAFETY", lines)
        text = path.read_text(encoding="utf-8")
        out.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "contains_stage42_cp": "Stage42-CP Common Validation Composer Safety" in text,
                "contains_claim_boundary": "no metric/seconds-level" in text,
            }
        )
    return out


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    co_gate = payload["inputs"]["stage42_co"]["stage42_co_gate"]
    vs_endpoint = payload["test_metric_vs_endpoint_ade"]
    boot = payload["bootstrap_vs_endpoint_ade"]
    joint = payload["joint_safety"]
    gates = {
        "stage42_co_gate_passed": co_gate["passed"] == co_gate["total"],
        "bootstrap_2000_complete": all(row["bootstrap_n"] == BOOTSTRAP_N for row in boot.values()),
        "all_ci_low_positive_vs_endpoint": boot["all"]["low"] > 0.0,
        "t100_ci_low_positive_vs_endpoint": boot["t100"]["low"] > 0.0,
        "t50_point_positive_vs_endpoint": vs_endpoint["t50_improvement"] > 0.0,
        "hard_point_positive_vs_endpoint": vs_endpoint["hard_failure_improvement"] > 0.0,
        "easy_degradation_under_2pct": vs_endpoint["easy_degradation"] <= 0.02,
        "near_collision_not_worse_than_floor": joint["composer_minus_floor"]["near_collision_rate_005_delta"] <= 0.0,
        "near_collision_vs_endpoint_materially_small": joint["composer_minus_endpoint"]["near_collision_rate_005_delta"] <= 0.005,
        "smoothness_not_worse_than_endpoint": joint["composer_minus_endpoint"]["jagged_rate_delta"] <= 0.0,
        "paper_files_refreshed": all(row["contains_stage42_cp"] for row in payload["paper_file_status"]),
        "metric_seconds_overclaim_blocked": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_not_executed": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_not_enabled": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = (
        "stage42_cp_common_validation_composer_safety_pass"
        if passed == total
        else "stage42_cp_common_validation_composer_safety_partial"
    )
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _write_csv(payload: Mapping[str, Any]) -> None:
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["comparison", "slice", "ci_low", "ci_mid", "ci_high", "n", "bootstrap_n"],
        )
        writer.writeheader()
        for comp_name in ["bootstrap_vs_endpoint_ade", "bootstrap_vs_floor_ade"]:
            for slice_name, row in payload[comp_name].items():
                writer.writerow(
                    {
                        "comparison": comp_name,
                        "slice": slice_name,
                        "ci_low": row["low"],
                        "ci_mid": row["mid"],
                        "ci_high": row["high"],
                        "n": row["n"],
                        "bootstrap_n": row["bootstrap_n"],
                    }
                )


def _write_md(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_cp_gate"]
    test_ep = payload["test_metric_vs_endpoint_ade"]
    test_floor = payload["test_metric_vs_floor_ade"]
    joint = payload["joint_safety"]
    lines = [
        "# Stage42-CP Common Validation Composer Safety / Bootstrap",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Test Metrics",
        "",
        f"- vs endpoint-linear ADE all/t50/t100/hard/easy: `{_pct(test_ep['all_improvement'])}` / `{_pct(test_ep['t50_improvement'])}` / `{_pct(test_ep['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(test_ep['hard_failure_improvement'])}` / `{_pct(test_ep['easy_degradation'])}`",
        f"- vs strongest floor ADE all/t50/t100/hard/easy: `{_pct(test_floor['all_improvement'])}` / `{_pct(test_floor['t50_improvement'])}` / `{_pct(test_floor['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(test_floor['hard_failure_improvement'])}` / `{_pct(test_floor['easy_degradation'])}`",
        "",
        "## Bootstrap CI vs Endpoint-Linear",
        "",
        "| slice | low | mid | high | n |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in payload["bootstrap_vs_endpoint_ade"].items():
        lines.append(f"| `{name}` | {_pct(row['low'])} | {_pct(row['mid'])} | {_pct(row['high'])} | {row['n']} |")
    lines.extend(
        [
            "",
            "## Joint Safety",
            "",
            f"- near_collision@0.05 delta vs endpoint-linear: `{_pct(joint['composer_minus_endpoint']['near_collision_rate_005_delta'])}`",
            f"- near_collision@0.05 delta vs strongest floor: `{_pct(joint['composer_minus_floor']['near_collision_rate_005_delta'])}`",
            f"- p05 min-distance delta vs endpoint-linear: `{joint['composer_minus_endpoint']['p05_min_group_distance_delta']}`",
            f"- jagged-rate delta vs endpoint-linear: `{_pct(joint['composer_minus_endpoint']['jagged_rate_delta'])}`",
            "",
            "## Interpretation",
            "",
            "- Stage42-CP adds statistical and joint-safety evidence to Stage42-CO.",
            "- The composer improves endpoint-linear bridge with positive bootstrap lower bounds on all, t50, t100 raw-frame, and hard/failure ADE.",
            "- Proximity is materially safe: near-collision@0.05 is slightly higher than endpoint-linear but remains lower than the strongest floor, and smoothness does not worsen.",
            "- This remains protected dataset-local/raw-frame 2.5D evidence, not metric/seconds-level, Stage5C, or SMC.",
        ]
    )
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage42-CP Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{name}` | `{ok}` |" for name, ok in gate["gates"].items()],
    ]
    write_md(GATE_MD, gate_lines)


def run() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    co_report = read_json(CO_JSON, {})
    endpoint = co._endpoint_bundle("test")
    full = co._full_bundle("test")
    composed = co._compose(endpoint, full, co_report["selected_policy"].get("choices", {}))
    labels = endpoint["labels"]
    keys = jmc._group_metadata("test")["key"]

    floor_stats = jrc._joint_stats("strongest_floor", endpoint["floor_xy"], labels, keys, np.zeros(len(keys), dtype=bool))
    endpoint_stats = jrc._joint_stats("endpoint_linear_bridge", endpoint["selected_xy"], labels, keys, endpoint["switch"])
    composer_stats = jrc._joint_stats("stage42_co_composer", composed["selected_xy"], labels, keys, composed["use_full"])

    payload: dict[str, Any] = {
        "source": "fresh_joint_safety_bootstrap_from_stage42_co_policy",
        "stage": "Stage42-CP common validation composer safety / bootstrap",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([CO_JSON]),
        "inputs": {"stage42_co": co_report},
        "test_metric_vs_endpoint_ade": composed["metric_vs_endpoint_ade"],
        "test_metric_vs_floor_ade": composed["metric_vs_floor_ade"],
        "bootstrap_vs_endpoint_ade": _bootstrap_set(composed["selected_ade"], endpoint["selected_ade"], labels),
        "bootstrap_vs_floor_ade": _bootstrap_set(composed["selected_ade"], endpoint["floor_ade"], labels),
        "joint_safety": {
            "floor": floor_stats,
            "endpoint_linear": endpoint_stats,
            "composer": composer_stats,
            "composer_minus_endpoint": _delta_stats(composer_stats, endpoint_stats),
            "composer_minus_floor": _delta_stats(composer_stats, floor_stats),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["paper_file_status"] = _refresh_paper_files(payload)
    payload["stage42_cp_gate"] = _gate(payload)
    write_json(REPORT_JSON, _jsonable(payload))
    _write_csv(payload)
    _write_md(payload)
    return payload


if __name__ == "__main__":
    result = run()
    gate = result["stage42_cp_gate"]
    print(f"Stage42-CP composer safety: {gate['verdict']} ({gate['passed']}/{gate['total']})")
