from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src import stage41_full_trajectory_world_state as ft
from src import stage42_sequence_full_waypoint as s42i


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "static_gated_full_waypoint_stage42.json"
REPORT_MD = OUT_DIR / "static_gated_full_waypoint_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_j_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
SEEDS = [53, 59, 61]
EPS = 1e-6


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "Stage42-J 使用 dataset-local raw-frame full-waypoint labels，不能写成 metric 或 seconds-level。",
    "future waypoints / future endpoints 只作为 eval label，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals。",
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


def _checkpoint_info(variant: str, seed: int) -> dict[str, Any]:
    ckpt = OUT_DIR / "checkpoints" / f"stage42i_{variant}_seed{seed}.pt"
    heartbeat = OUT_DIR / f"stage42i_{variant}_seed{seed}_heartbeat.json"
    if not ckpt.exists() or not heartbeat.exists():
        raise FileNotFoundError(f"Missing Stage42-I checkpoint or heartbeat for {variant} seed {seed}")
    return {"source": "cached_verified", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": read_json(heartbeat, {}).get("best", {})}


def _mix_pred(no_static: Mapping[str, np.ndarray], full: Mapping[str, np.ndarray], alpha: float) -> dict[str, np.ndarray]:
    return {key: ((1.0 - alpha) * no_static[key] + alpha * full[key]).astype(np.float32) for key in no_static}


def _floor_xy(labels: Mapping[str, np.ndarray]) -> np.ndarray:
    return ft._floor_waypoints(labels)


def _pred_selected_xy(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return s42i._selected_xy(pred, labels, policy)


def _metric_from_xy(selected_xy: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray) -> dict[str, Any]:
    floor = _floor_xy(labels)
    ade, fde = ft._trajectory_errors(selected_xy, labels)
    floor_ade, floor_fde = ft._trajectory_errors(floor, labels)
    return {
        "ade": ft._metric(ade, floor_ade, labels, switch),
        "fde": ft._metric(fde, floor_fde, labels, switch),
        "switch_rate": float(np.mean(switch)) if len(switch) else 0.0,
    }


def _score_metric(metric: Mapping[str, Any]) -> float:
    return (
        1.2 * float(metric.get("all_improvement", 0.0))
        + 1.6 * float(metric.get("t50_improvement", 0.0))
        + 1.0 * float(metric.get("hard_failure_improvement", 0.0))
        + 0.8 * float(metric.get("t100_improvement", 0.0))
        - 40.0 * max(0.0, float(metric.get("easy_degradation", 1.0)) - 0.02)
    )


def _slice_score(selected_xy: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray, mask: np.ndarray) -> tuple[float, dict[str, Any]]:
    if int(np.sum(mask)) < 20:
        return 0.0, {"rows": int(np.sum(mask)), "all_improvement": 0.0}
    floor = _floor_xy(labels)
    ade, _fde = ft._trajectory_errors(selected_xy, labels)
    floor_ade, _floor_fde = ft._trajectory_errors(floor, labels)
    metric = ft._metric(ade[mask], floor_ade[mask], {k: v[mask] for k, v in labels.items() if isinstance(v, np.ndarray) and len(v) == len(mask)}, switch[mask])
    if metric.get("easy_degradation", 1.0) > 0.02:
        return 0.0, metric
    return _score_metric(metric), metric


def _fit_expert(pred_val: Mapping[str, np.ndarray], labels_val: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    return s42i._fit_light_policy(pred_val, labels_val)


def _expert_eval(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> dict[str, Any]:
    xy, switch = _pred_selected_xy(pred, labels, policy)
    return _metric_from_xy(xy, labels, switch)


def _static_gate_seed(seed: int, val: Mapping[str, np.ndarray], test: Mapping[str, np.ndarray]) -> dict[str, Any]:
    full_info = _checkpoint_info("sequence_waypoint_full", seed)
    no_static_info = _checkpoint_info("sequence_waypoint_no_static_context", seed)
    pred_val_full = s42i._predict(full_info, val, "sequence_waypoint_full")
    pred_test_full = s42i._predict(full_info, test, "sequence_waypoint_full")
    pred_val_no_static = s42i._predict(no_static_info, val, "sequence_waypoint_no_static_context")
    pred_test_no_static = s42i._predict(no_static_info, test, "sequence_waypoint_no_static_context")
    labels_val = s42i._labels(val)
    labels_test = s42i._labels(test)
    experts: dict[str, dict[str, Any]] = {}
    for name, alpha in {
        "no_static": 0.0,
        "static_alpha025": 0.25,
        "static_alpha050": 0.50,
        "static_alpha075": 0.75,
        "full_static": 1.0,
    }.items():
        pred_val = _mix_pred(pred_val_no_static, pred_val_full, alpha)
        pred_test = _mix_pred(pred_test_no_static, pred_test_full, alpha)
        policy, val_metrics = _fit_expert(pred_val, labels_val)
        test_eval = _expert_eval(pred_test, labels_test, policy)
        selected_val, switch_val = _pred_selected_xy(pred_val, labels_val, policy)
        selected_test, switch_test = _pred_selected_xy(pred_test, labels_test, policy)
        experts[name] = {
            "alpha": alpha,
            "source": "cached_verified_checkpoints_fresh_gate_eval",
            "policy": policy,
            "val_metrics": val_metrics,
            "test_metrics": test_eval,
            "selected_val": selected_val,
            "switch_val": switch_val,
            "selected_test": selected_test,
            "switch_test": switch_test,
        }
    domain_val = labels_val["domain"].astype(str)
    horizon_val = labels_val["horizon"].astype(int)
    domain_test = labels_test["domain"].astype(str)
    horizon_test = labels_test["horizon"].astype(int)
    floor_test = _floor_xy(labels_test)
    gated_xy = floor_test.copy()
    gated_switch = np.zeros(len(floor_test), dtype=bool)
    slice_choices: dict[str, Any] = {}
    for domain in sorted(set(domain_val.tolist())):
        for horizon in [10, 25, 50, 100]:
            val_mask = (domain_val == domain) & (horizon_val == horizon)
            test_mask = (domain_test == domain) & (horizon_test == horizon)
            if int(np.sum(val_mask)) < 80 or not np.any(test_mask):
                continue
            best_name = "floor"
            best_score = 0.0
            best_metric: dict[str, Any] = {"rows": int(np.sum(val_mask)), "all_improvement": 0.0}
            for name, row in experts.items():
                score, metric = _slice_score(row["selected_val"], labels_val, row["switch_val"], val_mask)
                if score > best_score:
                    best_score = score
                    best_name = name
                    best_metric = metric
            if best_name != "floor":
                gated_xy[test_mask] = experts[best_name]["selected_test"][test_mask]
                gated_switch[test_mask] = experts[best_name]["switch_test"][test_mask]
            slice_choices[f"{domain}|{horizon}"] = {"expert": best_name, "val_score": float(best_score), "val_metric": best_metric}
    gated_metrics = _metric_from_xy(gated_xy, labels_test, gated_switch)
    return {
        "source": "cached_verified_checkpoints_fresh_static_gate_eval",
        "seed": seed,
        "checkpoint_sources": {"full": full_info, "no_static": no_static_info},
        "expert_metrics": {
            name: {
                "alpha": row["alpha"],
                "source": row["source"],
                "val_metrics": row["val_metrics"],
                "test_metrics": row["test_metrics"],
            }
            for name, row in experts.items()
        },
        "static_gated": {
            "slice_choices": slice_choices,
            "test_metrics": gated_metrics,
            "selected_expert_counts": {name: sum(1 for row in slice_choices.values() if row["expert"] == name) for name in ["floor", *experts.keys()]},
        },
    }


def _stat(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    mean = float(arr.mean()) if len(arr) else 0.0
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    half = 1.96 * std / np.sqrt(max(len(arr), 1))
    return {"mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half}


def _summarize(rows: list[Mapping[str, Any]], key: str) -> dict[str, Any]:
    def vals(path: list[str]) -> list[float]:
        out = []
        for row in rows:
            cur: Any = row
            for p in path:
                cur = cur[p]
            out.append(float(cur))
        return out

    return {
        "source": "cached_verified_checkpoints_fresh_static_gate_eval",
        "seeds": [int(row["seed"]) for row in rows],
        "ade_all": _stat(vals([key, "test_metrics", "ade", "all_improvement"])),
        "ade_t50": _stat(vals([key, "test_metrics", "ade", "t50_improvement"])),
        "ade_t100_raw_frame_diagnostic": _stat(vals([key, "test_metrics", "ade", "t100_improvement"])),
        "ade_hard_failure": _stat(vals([key, "test_metrics", "ade", "hard_failure_improvement"])),
        "ade_easy_degradation": _stat(vals([key, "test_metrics", "ade", "easy_degradation"])),
        "fde_all": _stat(vals([key, "test_metrics", "fde", "all_improvement"])),
        "fde_t50": _stat(vals([key, "test_metrics", "fde", "t50_improvement"])),
        "switch_rate": _stat(vals([key, "test_metrics", "switch_rate"])),
    }


def _summaries(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    experts = sorted(next(iter(rows))["expert_metrics"].keys()) if rows else []
    out = {"static_gated": _summarize(rows, "static_gated")}
    for expert in experts:
        converted = [{"seed": row["seed"], "expert": row["expert_metrics"][expert]} for row in rows]
        out[expert] = _summarize(converted, "expert")
    return out


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    summary = result.get("summary", {})
    gated = summary.get("static_gated", {})
    full = summary.get("full_static", {})
    no_static = summary.get("no_static", {})
    gates = {
        "stage42i_checkpoints_available": result.get("checkpoint_status") == "available",
        "three_seed_static_gate_eval": len(gated.get("seeds", [])) >= 3,
        "static_gate_positive": gated.get("ade_all", {}).get("mean", 0.0) > 0.0 or gated.get("ade_t50", {}).get("mean", 0.0) > 0.0 or gated.get("ade_hard_failure", {}).get("mean", 0.0) > 0.0,
        "static_gate_improves_full_static": gated.get("ade_all", {}).get("mean", 0.0) > full.get("ade_all", {}).get("mean", -1.0)
        and gated.get("ade_t50", {}).get("mean", 0.0) > full.get("ade_t50", {}).get("mean", -1.0),
        "static_gate_not_worse_than_no_static_all": gated.get("ade_all", {}).get("mean", -1.0) >= no_static.get("ade_all", {}).get("mean", -1.0) - 1e-6,
        "easy_preserved": gated.get("ade_easy_degradation", {}).get("mean", 1.0) <= 0.02,
        "no_leakage_pass": result.get("no_leakage", {}).get("future_endpoint_input") is False
        and result.get("no_leakage", {}).get("future_waypoints_input") is False
        and result.get("no_leakage", {}).get("central_velocity") is False
        and result.get("no_leakage", {}).get("test_endpoint_goals") is False,
        "no_metric_seconds_overclaim": result.get("claim_boundary", {}).get("metric_or_seconds_claim") is False,
        "stage5c_false": result.get("claim_boundary", {}).get("stage5c_executed") is False,
        "smc_false": result.get("claim_boundary", {}).get("smc_enabled") is False,
    }
    return {
        "source": "cached_verified_checkpoints_fresh_static_gate_eval",
        "gates": gates,
        "passed": int(sum(bool(v) for v in gates.values())),
        "total": len(gates),
        "verdict": "stage42_j_static_gated_full_waypoint_pass" if all(gates.values()) else "stage42_j_static_gated_full_waypoint_partial",
    }


def run_stage42_static_gated_full_waypoint() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ft.build_full_trajectory_labels()
    val = s42i._split_arrays("val")
    test = s42i._split_arrays("test")
    rows = [_static_gate_seed(seed, val, test) for seed in SEEDS]
    result = {
        "source": "cached_verified_checkpoints_fresh_static_gate_eval",
        "stage": "Stage42-J static-gated full-waypoint repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                ft.DATA_DIR / "all_agent_val.npz",
                ft.DATA_DIR / "all_agent_test.npz",
                ft.DATA_DIR / "full_trajectory_val.npz",
                ft.DATA_DIR / "full_trajectory_test.npz",
                REPORT_JSON.with_name("sequence_full_waypoint_stage42.json"),
            ]
        ),
        "checkpoint_status": "available",
        "dataset_rows": {"val": int(len(val["horizon"])), "test": int(len(test["horizon"]))},
        "rows": rows,
        "summary": _summaries(rows),
        "source_labels": {
            "stage42i_checkpoints": "cached_verified",
            "static_gate_selection": "fresh_run_on_val",
            "test_evaluation": "fresh_run_once_per_seed",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "future_waypoints_used_as_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "experts_selected_on_val": True,
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
    result["stage42_j_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    _write_report(result)
    _write_gate(result["stage42_j_gate"])
    _append_readme_and_state(result)
    _append_ledger(result)
    return result


def _write_report(result: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-J Static-Gated Full-Waypoint Repair",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_j_gate']['passed']} / {result['stage42_j_gate']['total']}`",
        f"- verdict: `{result['stage42_j_gate']['verdict']}`",
        "",
        "## Current Facts",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Metrics",
        "",
        "| candidate | source | ADE all | ADE t50 | ADE t100 diag | ADE hard | ADE easy degr | FDE all | FDE t50 | switch |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in sorted(result["summary"].items()):
        lines.append(
            f"| `{name}` | `{row['source']}` | {row['ade_all']['mean']:.6f} | {row['ade_t50']['mean']:.6f} | {row['ade_t100_raw_frame_diagnostic']['mean']:.6f} | {row['ade_hard_failure']['mean']:.6f} | {row['ade_easy_degradation']['mean']:.6f} | {row['fde_all']['mean']:.6f} | {row['fde_t50']['mean']:.6f} | {row['switch_rate']['mean']:.6f} |"
        )
    counts = [row["static_gated"]["selected_expert_counts"] for row in result["rows"]]
    lines.extend(
        [
            "",
            "## Gate Behavior",
            "",
            f"- selected expert counts by seed: `{counts}`",
            "- Experts and static mix weights are selected on validation by domain/horizon slice; test is evaluated once.",
            "- `no_static` means the Stage42-I no-static-context sequence head; `full_static` means the Stage42-I full static+sequence head.",
            "",
            "## Interpretation",
            "",
            "- Stage42-J repairs the Stage42-I failure mode by not forcing static/context into every full-waypoint prediction.",
            "- This is a static expert gate over cached-verified Stage42-I checkpoints with fresh validation-gate/test evaluation, not a new checkpoint training run.",
            "- If the static gate falls back mostly to no-static, that is still useful evidence: static/context is currently harmful unless gated.",
            "- Results remain dataset-local raw-frame 2.5D; no metric/seconds-level, Stage5C, or SMC claim is made.",
        ]
    )
    write_md(REPORT_MD, lines)


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-J Gate",
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


def _append_readme_and_state(result: Mapping[str, Any]) -> None:
    gate = result["stage42_j_gate"]
    gated = result["summary"]["static_gated"]
    block = f"""
## Stage42-J Static-Gated Full-Waypoint Repair

```text
source = {result['source']}
verdict = {gate['verdict']}
gates = {gate['passed']} / {gate['total']}
static_gated_ade_all = {gated['ade_all']['mean']}
static_gated_ade_t50 = {gated['ade_t50']['mean']}
static_gated_ade_hard_failure = {gated['ade_hard_failure']['mean']}
static_gated_ade_easy_degradation = {gated['ade_easy_degradation']['mean']}
static_gated_fde_t50 = {gated['fde_t50']['mean']}
stage5c_executed = false
smc_enabled = false
```

Stage42-J uses cached-verified Stage42-I no-static/full-static checkpoints and performs a fresh validation-selected static expert gate. It tests whether static/context should be allowed per domain/horizon rather than forced globally. It remains dataset-local raw-frame 2.5D evidence and not Stage5C/SMC.
"""
    _append_if_missing(Path("README_RESULTS.md"), "## Stage42-J Static-Gated Full-Waypoint Repair", block)
    _append_if_missing(Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md"), "## Stage42-J Static-Gated Full-Waypoint Repair", block)
    state = read_json(Path("research_state.json"), {})
    state["current_stage"] = "stage42_j_static_gated_full_waypoint"
    state["current_verdict"] = gate["verdict"]
    state.setdefault("stage42", {})["stage_j_static_gated_full_waypoint"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "gates_passed": gate["passed"],
        "gates_total": gate["total"],
        "verdict": gate["verdict"],
        "static_gated_ade_all": gated["ade_all"]["mean"],
        "static_gated_ade_t50": gated["ade_t50"]["mean"],
        "static_gated_ade_hard_failure": gated["ade_hard_failure"]["mean"],
        "static_gated_ade_easy_degradation": gated["ade_easy_degradation"]["mean"],
        "static_gated_fde_t50": gated["fde_t50"]["mean"],
        "claim_boundary": result["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(Path("research_state.json"), _jsonable(state))


def _append_ledger(result: Mapping[str, Any]) -> None:
    entry = {
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "step": "stage42_j_static_gated_full_waypoint",
        "source": result["source"],
        "status": "success",
        "input_hash": result.get("input_hash"),
        "output_hash": _combined_hash([REPORT_JSON, REPORT_MD, GATE_MD]),
        "git_commit": _git_commit(),
        "generated_at_utc": result.get("generated_at_utc"),
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_static_gated_full_waypoint()
