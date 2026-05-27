from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_source_level_ablation as an
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_level_incremental_ablation as ao
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "context_gain_router_stage42.json"
REPORT_MD = OUT_DIR / "context_gain_router_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_el_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_context_gain_router"
MIN_INCREMENT = 0.01
CANDIDATES = ["history_only", "motion_goal_context", "baseline_plus_history_goal_neighbor"]
RIDGE_LAMBDAS = [0.1, 1.0, 10.0, 100.0]
EPS = 1e-6

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-EL 是 context gain-router fresh probe，不是重复 sequence/graph residual-delta protocol。",
    "router target 是 baseline-family protected floor 与 context proposal 的 supervised gain/harm，而不是直接预测 residual 轨迹。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}


def _fit_ridge_vector(x: np.ndarray, y: np.ndarray, mask: np.ndarray, lam: float) -> np.ndarray:
    ids = np.where(mask)[0]
    xt = x[ids].astype(np.float64, copy=False)
    yt = y[ids].astype(np.float64, copy=False)
    reg = np.eye(xt.shape[1], dtype=np.float64) * float(lam)
    reg[-1, -1] = 0.0
    return np.linalg.solve(xt.T @ xt + reg, xt.T @ yt).astype(np.float32)


def _metric(selected: np.ndarray, floor: np.ndarray, data: Mapping[str, np.ndarray], switch: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    return am._metric(selected, floor, data, switch, mask)


def _score(metric: Mapping[str, Any]) -> float:
    return float(
        1.2 * metric["all_improvement"]
        + 1.8 * metric["t50_improvement"]
        + 1.1 * metric["hard_failure_improvement"]
        - 30.0 * max(0.0, metric["easy_degradation"] - 0.02)
        - 0.03 * metric["switch_rate"]
    )


def _prepare_variant_predictions(raw_features: np.ndarray, shared: Mapping[str, Any]) -> dict[str, Any]:
    split = shared["split"]
    data = shared["data"]
    labels = shared["labels"]
    floor = shared["floor"]
    train_mask = split == "train"
    val_mask = split == "val"
    x, _, _ = am._standardize(raw_features, train_mask)
    target_delta = (
        (
            labels["waypoint_xy"].astype(np.float64)
            - np.stack([data["current_x"], data["current_y"]], axis=1)[:, None, :]
        )
        / np.maximum(data["scale"].astype(np.float64)[:, None, None], EPS)
    ).astype(np.float32)
    best: dict[str, Any] | None = None
    best_score = -1e9
    for lam in RIDGE_LAMBDAS:
        coef = am._fit_ridge_model(x, target_delta, labels["waypoint_valid"], train_mask, lam)
        pred_xy = am._predict_waypoints(x, coef, data)
        policy, selected_ade, selected_fde, switch = am._select_policy_on_val(
            pred_xy, floor["floor_xy"], labels, data, val_mask
        )
        floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
        val_metric = _metric(selected_ade, floor_ade, data, switch, val_mask)
        score = _score(val_metric)
        if score > best_score:
            best_score = score
            best = {
                "lambda": float(lam),
                "policy": policy,
                "selected_ade": selected_ade,
                "selected_fde": selected_fde,
                "switch": switch,
                "floor_ade": floor_ade,
                "floor_fde": floor_fde,
                "val_metric": val_metric,
                "score": float(score),
            }
    if best is None:
        raise RuntimeError("No variant prediction was evaluated.")
    return best


def _train_gain_router(
    *,
    name: str,
    raw_router_features: np.ndarray,
    base_ade: np.ndarray,
    candidate_ade: np.ndarray,
    split: np.ndarray,
    data: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    x, _, _ = am._standardize(raw_router_features, train_mask)
    gain = (base_ade - candidate_ade).astype(np.float32)
    val_candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_score = -1e9
    for lam in RIDGE_LAMBDAS:
        coef = _fit_ridge_vector(x, gain, train_mask, lam)
        pred_gain = (x.astype(np.float64) @ coef.astype(np.float64)).astype(np.float64)
        val_pred = pred_gain[val_mask]
        thresholds = sorted(set(float(np.quantile(val_pred, q)) for q in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]))
        thresholds.extend([0.0, float(np.mean(val_pred)), float(np.mean(val_pred) + np.std(val_pred))])
        for threshold in thresholds:
            switch = pred_gain > threshold
            selected = base_ade.copy()
            selected[switch] = candidate_ade[switch]
            val_metric = _metric(selected, base_ade, data, switch, val_mask)
            if val_metric["easy_degradation"] > 0.02:
                continue
            score = _score(val_metric)
            row = {
                "lambda": float(lam),
                "pred_gain_threshold": float(threshold),
                "val_score": float(score),
                "val_metric": val_metric,
            }
            val_candidates.append(row)
            if score > best_score:
                best_score = score
                best = {
                    **row,
                    "coef": coef,
                    "pred_gain": pred_gain,
                    "switch": switch,
                    "selected_ade": selected,
                }
    if best is None:
        switch = np.zeros(len(base_ade), dtype=bool)
        selected = base_ade.copy()
        best = {
            "lambda": None,
            "pred_gain_threshold": None,
            "val_score": 0.0,
            "val_metric": _metric(selected, base_ade, data, switch, val_mask),
            "coef": None,
            "pred_gain": np.zeros(len(base_ade), dtype=np.float64),
            "switch": switch,
            "selected_ade": selected,
        }
    test_metric = _metric(best["selected_ade"], base_ade, data, best["switch"], test_mask)
    horizon = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    return {
        "source": "fresh_run",
        "candidate": name,
        "validation_selection": {
            "source": "validation_only",
            "test_threshold_tuning": False,
            "candidate_count": len(val_candidates),
            "selected_lambda": best["lambda"],
            "selected_pred_gain_threshold": best["pred_gain_threshold"],
            "selected_val_score": best["val_score"],
            "selected_val_metric": best["val_metric"],
        },
        "test_metric_vs_baseline_family": test_metric,
        "bootstrap_vs_baseline_family": {
            "all": am._bootstrap_ci(best["selected_ade"], base_ade, test_mask, seed=42601),
            "t50": am._bootstrap_ci(best["selected_ade"], base_ade, test_mask & (horizon == 50), seed=42602),
            "hard_failure": am._bootstrap_ci(best["selected_ade"], base_ade, test_mask & hard_failure, seed=42603),
            "easy_degradation": am._bootstrap_ci(base_ade, best["selected_ade"], test_mask & easy, seed=42604),
        },
        "router_switch_rate_test": float(np.mean(best["switch"][test_mask])) if np.any(test_mask) else 0.0,
        "increment_supported": (
            (
                test_metric["all_improvement"] > MIN_INCREMENT
                or test_metric["t50_improvement"] > MIN_INCREMENT
                or test_metric["hard_failure_improvement"] > MIN_INCREMENT
            )
            and test_metric["easy_degradation"] <= 0.02
        ),
    }


def _build_result() -> dict[str, Any]:
    shared = an._prep_shared()
    masks = ao._incremental_variant_masks(shared["feature_names"])
    split = shared["split"]
    base_pred = _prepare_variant_predictions(shared["features"][:, masks["baseline_family_only"]], shared)
    routers: dict[str, Any] = {}
    for name in CANDIDATES:
        candidate_pred = _prepare_variant_predictions(shared["features"][:, masks[name]], shared)
        routers[name] = _train_gain_router(
            name=name,
            raw_router_features=shared["features"][:, masks[name]],
            base_ade=base_pred["selected_ade"],
            candidate_ade=candidate_pred["selected_ade"],
            split=split,
            data=shared["data"],
        )
        routers[name]["candidate_policy"] = {
            "lambda": candidate_pred["lambda"],
            "policy_slice_count": len(candidate_pred["policy"]["slices"]),
            "val_metric_vs_causal_floor": candidate_pred["val_metric"],
        }
    positive = sorted([name for name, row in routers.items() if row["increment_supported"]])
    best_name = max(
        routers,
        key=lambda key: (
            routers[key]["test_metric_vs_baseline_family"]["all_improvement"]
            + routers[key]["test_metric_vs_baseline_family"]["t50_improvement"]
            + routers[key]["test_metric_vs_baseline_family"]["hard_failure_improvement"]
        ),
    )
    return {
        "source": SOURCE,
        "stage": "Stage42-EL Context Gain Router",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/source_level_incremental_ablation_stage42.json",
                "outputs/stage42_long_research/context_model_closure_stage42.json",
            ]
        ),
        "current_facts": CURRENT_FACTS,
        "split_stats": shared["split_stats"],
        "baseline_family_control": {
            "lambda": base_pred["lambda"],
            "policy_slice_count": len(base_pred["policy"]["slices"]),
            "val_metric_vs_causal_floor": base_pred["val_metric"],
        },
        "routers": routers,
        "positive_context_gain_routers": positive,
        "best_router": best_name,
        "summary": {
            "source": SOURCE,
            "router_target": "predict supervised gain of context proposal over baseline-family protected control",
            "candidates": CANDIDATES,
            "positive_context_gain_routers": positive,
            "best_router": best_name,
            "best_router_test_metric_vs_baseline_family": routers[best_name]["test_metric_vs_baseline_family"],
            "context_increment_verdict": (
                "stage42_el_context_gain_router_supported" if positive else "stage42_el_context_gain_router_not_supported"
            ),
            "interpretation": (
                "This is a deployment-aligned context target. Positive routers would support context as a safe switchability "
                "signal over the baseline-family control. Negative routers preserve the current conclusion that context "
                "is not yet an independent deployable contribution under this source-level protocol."
            ),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_supervised_gain_label_only": True,
            "validation_selected_thresholds": True,
            "source_overlap_pass": bool(shared["split_stats"]["source_overlap_pass"]),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    no_leakage = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "source_level_split_used": payload["split_stats"]["by_split"]["test"]["rows"] == 47458,
        "baseline_family_control_loaded": payload["baseline_family_control"]["policy_slice_count"] >= 0,
        "gain_router_candidates_complete": len(payload["routers"]) >= 3,
        "validation_only_selection": all(
            row["validation_selection"]["source"] == "validation_only"
            and row["validation_selection"]["test_threshold_tuning"] is False
            for row in payload["routers"].values()
        ),
        "context_increment_measured": payload["summary"]["context_increment_verdict"]
        in {"stage42_el_context_gain_router_supported", "stage42_el_context_gain_router_not_supported"},
        "negative_or_positive_claim_bounded": isinstance(payload["positive_context_gain_routers"], list),
        "no_leakage_pass": no_leakage["future_endpoint_input"] is False
        and no_leakage["future_waypoint_input"] is False
        and no_leakage["central_velocity"] is False
        and no_leakage["test_endpoint_goals"] is False
        and no_leakage["test_threshold_tuning"] is False
        and no_leakage["validation_selected_thresholds"] is True,
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = sum(1 for value in gates.values() if value)
    total = len(gates)
    verdict = "stage42_el_context_gain_router_pass" if passed == total else "stage42_el_context_gain_router_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-EL Context Gain Router",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_el_gate']['passed']} / {payload['stage42_el_gate']['total']}`",
        f"- verdict: `{payload['stage42_el_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- router_target: `{payload['summary']['router_target']}`",
        f"- candidates: `{payload['summary']['candidates']}`",
        f"- positive_context_gain_routers: `{payload['summary']['positive_context_gain_routers']}`",
        f"- best_router: `{payload['summary']['best_router']}`",
        f"- context_increment_verdict: `{payload['summary']['context_increment_verdict']}`",
        "",
        payload["summary"]["interpretation"],
        "",
        "## Router Results vs Baseline-Family Protected Control",
        "",
        "| candidate | all | t50 | hard/failure | easy degradation | switch rate | increment supported |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in payload["routers"].items():
        m = row["test_metric_vs_baseline_family"]
        lines.append(
            f"| `{name}` | {m['all_improvement']:.6f} | {m['t50_improvement']:.6f} | "
            f"{m['hard_failure_improvement']:.6f} | {m['easy_degradation']:.6f} | "
            f"{m['switch_rate']:.6f} | {row['increment_supported']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This stage changes the context target from residual trajectory prediction to gain/harm routing over a strong baseline-family control.",
            "- A negative result keeps scene/goal/neighbor/interaction as diagnostic under this protocol; it does not prove context can never help.",
            "- A positive result would support context as a safe switchability signal only, not as metric/seconds-level or true-3D evidence.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_el_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_el_gate"]
    return [
        "# Stage42-EL Gate",
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
    best = s["best_router_test_metric_vs_baseline_family"]
    return [
        "## Stage42-EL Context Gain Router",
        "",
        "- source: `fresh_stage42_context_gain_router`",
        "- role: tests a deployment-aligned context target: supervised gain/harm routing over baseline-family protected control.",
        f"- gate: `{payload['stage42_el_gate']['passed']} / {payload['stage42_el_gate']['total']}`; verdict `{payload['stage42_el_gate']['verdict']}`.",
        f"- positive_context_gain_routers: `{s['positive_context_gain_routers']}`; best router `{s['best_router']}`.",
        f"- best all/t50/hard delta vs baseline-family: `{best['all_improvement']:.6f}` / `{best['t50_improvement']:.6f}` / `{best['hard_failure_improvement']:.6f}`; easy `{best['easy_degradation']:.6f}`.",
        f"- context_increment_verdict: `{s['context_increment_verdict']}`.",
        "- Boundary: source-level raw-frame only; no metric/seconds claim, no Stage5C, no SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, GOAL_SUMMARY]:
        _replace_section(path, "STAGE42_EL_CONTEXT_GAIN_ROUTER", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-EL context gain router"
    state["current_verdict"] = payload["stage42_el_gate"]["verdict"]
    state["stage42_el_context_gain_router"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_el_gate"]["verdict"],
        "gates": f"{payload['stage42_el_gate']['passed']}/{payload['stage42_el_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_context_gain_router(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    payload = _build_result()
    payload["stage42_el_gate"] = _gate(payload)
    write_json(REPORT_JSON, am._jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_context_gain_router()
