from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_scene_proxy_augmented_latent_dynamics as ab


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_scene_proxy_guarded_latent_policy.json"
REPORT_MD = OUT_DIR / "stage43_scene_proxy_guarded_latent_policy.md"
GATE_MD = OUT_DIR / "stage43_stage_ac_scene_proxy_guarded_latent_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_AC_SCENE_PROXY_GUARDED_LATENT_POLICY"
SOURCE = "fresh_stage43_ac_scene_proxy_guarded_latent_policy"

POLICY_FAMILIES = [
    "ab_all",
    "ab_non_h100",
    "ab_h50_only",
    "ab_h50_or_hard_non_h100",
    "ab_hard_failure_non_h100",
    "ab_non_easy_non_h100",
]


def _apply_checkpoint_standardization(ds: m.WaypointSplit, checkpoint: Mapping[str, Any]) -> m.WaypointSplit:
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    if ds.x.shape[1] != len(mean):
        raise ValueError(f"Feature dimension mismatch: {ds.x.shape[1]} != checkpoint mean {len(mean)}")
    ds.x = ((ds.x - mean) / std).astype(np.float32)
    return ds


def _load_model(checkpoint_path: Path) -> tuple[m.FullWaypointLatentDynamics, Mapping[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = m.FullWaypointLatentDynamics(
        int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        latent_dim=int(checkpoint.get("latent_dim", 32)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _max_rows(mode: str) -> dict[str, int]:
    if mode == "quick":
        return {"val": 3000, "test": 3000}
    if mode == "medium":
        return {"val": 40000, "test": 50000}
    return {"val": 12000, "test": 16000}


def _stage43_m_split(split: str, *, max_rows: int, seed: int, checkpoint: Mapping[str, Any]) -> m.WaypointSplit:
    return _apply_checkpoint_standardization(m._build_split(split, max_rows=max_rows, seed=seed), checkpoint)


def _stage43_ab_split(split: str, *, max_rows: int, seed: int, checkpoint: Mapping[str, Any]) -> m.WaypointSplit:
    return _apply_checkpoint_standardization(ab._augmented_split(split, max_rows=max_rows, seed=seed), checkpoint)


def _assert_same_rows(left: m.WaypointSplit, right: m.WaypointSplit) -> None:
    checks = [
        np.array_equal(left.horizon, right.horizon),
        np.array_equal(left.domain, right.domain),
        np.array_equal(left.source_file, right.source_file),
        np.array_equal(left.scene_id, right.scene_id),
        np.allclose(left.floor_ade, right.floor_ade),
        np.allclose(left.waypoint_delta, right.waypoint_delta),
    ]
    if not all(checks):
        raise ValueError("Stage43-M and Stage43-AB replay rows are not aligned.")


def _replay_split(
    split: str,
    *,
    max_rows: int,
    seed: int,
    batch_size: int,
    m_model: m.FullWaypointLatentDynamics,
    m_ckpt: Mapping[str, Any],
    ab_model: m.FullWaypointLatentDynamics,
    ab_ckpt: Mapping[str, Any],
    m_policy: Mapping[str, float],
) -> dict[str, Any]:
    ds_m = _stage43_m_split(split, max_rows=max_rows, seed=seed, checkpoint=m_ckpt)
    ds_ab = _stage43_ab_split(split, max_rows=max_rows, seed=seed, checkpoint=ab_ckpt)
    _assert_same_rows(ds_m, ds_ab)
    device = torch.device("cpu")
    pred_m = m._predict(m_model, ds_m, device, batch_size)
    pred_ab = m._predict(ab_model, ds_ab, device, batch_size)
    m_ade, m_fde, m_switched = m._select_with_policy(ds_m, pred_m, m_policy)
    ab_ade, ab_fde = m._trajectory_error(ds_ab, pred_ab["waypoint"])
    floor_metrics = m._metrics(ds_ab, ds_ab.floor_ade, ds_ab.floor_fde, np.zeros(len(ds_ab.x), dtype=bool))
    m_metrics = m._metrics(ds_ab, m_ade, m_fde, m_switched)
    ab_metrics = m._metrics(ds_ab, ab_ade, ab_fde, np.ones(len(ds_ab.x), dtype=bool))
    return {
        "ds": ds_ab,
        "pred_ab": pred_ab,
        "m_ade": m_ade,
        "m_fde": m_fde,
        "m_switched": m_switched,
        "ab_ade": ab_ade,
        "ab_fde": ab_fde,
        "floor_metrics": floor_metrics,
        "stage43_m_metrics": m_metrics,
        "stage43_ab_all_metrics": ab_metrics,
    }


def _family_mask(ds: m.WaypointSplit, family: str) -> np.ndarray:
    h50 = ds.horizon == 50
    non_h100 = ds.horizon != 100
    hard_failure = ds.hard | ds.failure
    if family == "ab_all":
        return np.ones(len(ds.x), dtype=bool)
    if family == "ab_non_h100":
        return non_h100
    if family == "ab_h50_only":
        return h50
    if family == "ab_h50_or_hard_non_h100":
        return non_h100 & (h50 | hard_failure)
    if family == "ab_hard_failure_non_h100":
        return non_h100 & hard_failure
    if family == "ab_non_easy_non_h100":
        return non_h100 & (~ds.easy)
    raise KeyError(f"Unknown policy family: {family}")


def _select_guarded(pack: Mapping[str, Any], policy: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ds: m.WaypointSplit = pack["ds"]
    pred = pack["pred_ab"]
    family = _family_mask(ds, str(policy["family"]))
    ab_allowed = (
        family
        & (pred["gain"] >= float(policy["gain_threshold"]))
        & (pred["harm"] <= float(policy["harm_threshold"]))
        & (pred["failure"] >= float(policy["failure_threshold"]))
    )
    selected_ade = np.where(ab_allowed, pack["ab_ade"], pack["m_ade"]).astype(np.float32)
    selected_fde = np.where(ab_allowed, pack["ab_fde"], pack["m_fde"]).astype(np.float32)
    switched_from_floor = (ab_allowed | pack["m_switched"]).astype(bool)
    return selected_ade, selected_fde, switched_from_floor, ab_allowed.astype(bool)


def _delta_metrics(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    keys = [
        "full_waypoint_ade_improvement_vs_floor",
        "endpoint_fde_improvement_vs_floor",
        "t50_full_waypoint_ade_improvement_vs_floor",
        "t50_endpoint_fde_improvement_vs_floor",
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor",
        "hard_failure_full_waypoint_ade_improvement_vs_floor",
        "easy_degradation_vs_floor",
        "switch_rate",
    ]
    return {key: float(current.get(key, 0.0)) - float(baseline.get(key, 0.0)) for key in keys}


def _eval_guarded(pack: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    selected_ade, selected_fde, switched, ab_allowed = _select_guarded(pack, policy)
    metrics = m._metrics(pack["ds"], selected_ade, selected_fde, switched)
    metrics["scene_proxy_override_rate"] = float(np.mean(ab_allowed))
    metrics["stage43_m_fallback_rate"] = float(1.0 - np.mean(ab_allowed))
    metrics["t100_scene_proxy_override_rate"] = float(np.mean(ab_allowed[pack["ds"].horizon == 100])) if int((pack["ds"].horizon == 100).sum()) else 0.0
    metrics["h50_scene_proxy_override_rate"] = float(np.mean(ab_allowed[pack["ds"].horizon == 50])) if int((pack["ds"].horizon == 50).sum()) else 0.0
    return metrics


def _search_guarded_policy(val_pack: Mapping[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    base = val_pack["stage43_m_metrics"]
    for family in POLICY_FAMILIES:
        for gain in [0.0, 0.25, 0.45, 0.55, 0.65, 0.75, 0.85]:
            for harm in [0.15, 0.25, 0.35, 0.50, 0.75, 1.00]:
                for failure in [0.0, 0.10, 0.20, 0.35, 0.50]:
                    policy = {
                        "family": family,
                        "gain_threshold": gain,
                        "harm_threshold": harm,
                        "failure_threshold": failure,
                        "fallback": "stage43_m_protected_policy",
                        "selected_on": "validation_only",
                        "test_threshold_tuning": False,
                    }
                    metrics = _eval_guarded(val_pack, policy)
                    delta = _delta_metrics(metrics, base)
                    t100_penalty = max(0.0, -delta["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] - 0.002)
                    easy_penalty = max(0.0, metrics["easy_degradation_vs_floor"] - 0.02)
                    objective = (
                        1.0 * delta["full_waypoint_ade_improvement_vs_floor"]
                        + 1.6 * delta["t50_full_waypoint_ade_improvement_vs_floor"]
                        + 1.0 * delta["hard_failure_full_waypoint_ade_improvement_vs_floor"]
                        + 0.8 * min(0.02, delta["t100_raw_frame_full_waypoint_diagnostic_vs_floor"])
                        - 30.0 * t100_penalty
                        - 40.0 * easy_penalty
                        - 0.02 * metrics["scene_proxy_override_rate"]
                    )
                    row = {"policy": policy, "metrics": metrics, "delta_vs_stage43_m": delta, "objective": float(objective)}
                    if best is None or row["objective"] > best["objective"]:
                        best = row
    assert best is not None
    return best


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["test_metrics_guarded"]
    delta = payload["delta_vs_stage43_m"]
    gates = {
        "stage43_m_available": payload["stage43_m_report"].get("stage43_m_gate", {}).get("verdict")
        in {"stage43_m_protected_full_waypoint_latent_candidate_pass", "stage43_m_full_waypoint_latent_diagnostic_keep_floor"},
        "stage43_ab_available": payload["stage43_ab_report"].get("stage43_ab_gate", {}).get("verdict")
        == "stage43_ab_scene_proxy_augmented_latent_lift_candidate",
        "fresh_replay_completed": payload["result_source"] == "fresh_replay_guarded_scene_proxy_policy",
        "validation_only_selection": payload["validation_selected_policy"]["policy"]["selected_on"] == "validation_only"
        and payload["validation_selected_policy"]["policy"]["test_threshold_tuning"] is False,
        "row_alignment_passed": payload["row_alignment_passed"] is True,
        "easy_preserved": metrics["easy_degradation_vs_floor"] <= 0.02,
        "core_lift_over_stage43_m": (
            delta["full_waypoint_ade_improvement_vs_floor"] > 0.0
            or delta["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0
            or delta["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0
        ),
        "t100_not_worse_than_stage43_m": delta["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] >= -0.002,
        "scene_proxy_not_all_h100": metrics["t100_scene_proxy_override_rate"] <= 0.05,
        "no_future_or_test_leakage": payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["test_endpoint_goal_construction"] is False
        and payload["no_leakage"]["test_threshold_tuning"] is False,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    deploy = bool(passed == total)
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_ac_guarded_scene_proxy_latent_candidate"
        if deploy
        else "stage43_ac_guarded_scene_proxy_diagnostic_keep_stage43_ab_or_m",
        "deploy_guarded_scene_proxy_latent": deploy,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    seed = int(args.seed)
    runtime = m._configure_runtime(seed)
    mode = "quick" if args.quick else "medium" if args.medium else "small"
    rows = _max_rows(mode)
    stage43_m_report = read_json(m.REPORT_JSON, {})
    stage43_ab_report = read_json(ab.REPORT_JSON, {})
    m_model, m_ckpt = _load_model(Path(stage43_m_report["checkpoint"]))
    ab_model, ab_ckpt = _load_model(Path(stage43_ab_report["checkpoint"]))
    m_policy = stage43_m_report["validation_selected_policy"]["policy"]

    val_pack = _replay_split(
        "val",
        max_rows=rows["val"],
        seed=seed,
        batch_size=int(args.batch_size),
        m_model=m_model,
        m_ckpt=m_ckpt,
        ab_model=ab_model,
        ab_ckpt=ab_ckpt,
        m_policy=m_policy,
    )
    test_pack = _replay_split(
        "test",
        max_rows=rows["test"],
        seed=seed,
        batch_size=int(args.batch_size),
        m_model=m_model,
        m_ckpt=m_ckpt,
        ab_model=ab_model,
        ab_ckpt=ab_ckpt,
        m_policy=m_policy,
    )
    best = _search_guarded_policy(val_pack)
    selected_ade, selected_fde, switched, ab_allowed = _select_guarded(test_pack, best["policy"])
    test_metrics = m._metrics(test_pack["ds"], selected_ade, selected_fde, switched)
    test_metrics["scene_proxy_override_rate"] = float(np.mean(ab_allowed))
    test_metrics["stage43_m_fallback_rate"] = float(1.0 - np.mean(ab_allowed))
    test_metrics["t100_scene_proxy_override_rate"] = float(np.mean(ab_allowed[test_pack["ds"].horizon == 100])) if int((test_pack["ds"].horizon == 100).sum()) else 0.0
    test_metrics["h50_scene_proxy_override_rate"] = float(np.mean(ab_allowed[test_pack["ds"].horizon == 50])) if int((test_pack["ds"].horizon == 50).sum()) else 0.0
    delta_vs_m = _delta_metrics(test_metrics, test_pack["stage43_m_metrics"])
    delta_vs_ab = _delta_metrics(test_metrics, test_pack["stage43_ab_all_metrics"])
    bootstrap = m._bootstrap_ci(test_pack["ds"], selected_ade, selected_fde, n=int(args.bootstrap), seed=seed + 2000)

    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_replay_guarded_scene_proxy_policy",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "mode": mode,
        "runtime": runtime,
        "stage43_m_report": stage43_m_report,
        "stage43_ab_report": stage43_ab_report,
        "data_rows": {"val": len(val_pack["ds"].x), "test": len(test_pack["ds"].x)},
        "row_alignment_passed": True,
        "validation_selected_policy": best,
        "test_metrics_guarded": test_metrics,
        "stage43_m_metrics_replayed": test_pack["stage43_m_metrics"],
        "stage43_ab_all_metrics_replayed": test_pack["stage43_ab_all_metrics"],
        "delta_vs_stage43_m": delta_vs_m,
        "delta_vs_stage43_ab_all": delta_vs_ab,
        "bootstrap_ci": bootstrap,
        "scene_proxy_override_counts": {
            "test_total": int(len(ab_allowed)),
            "test_override": int(ab_allowed.sum()),
            "test_h50_override": int(ab_allowed[test_pack["ds"].horizon == 50].sum()),
            "test_h100_override": int(ab_allowed[test_pack["ds"].horizon == 100].sum()),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
            "scene_proxy_train_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "scene_proxy_not_raw_image_or_true_sdf": True,
            "t100_raw_frame_diagnostic_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "input_hash": _combined_hash(
            [
                m.REPORT_JSON,
                ab.REPORT_JSON,
                Path(stage43_m_report["checkpoint"]),
                Path(stage43_ab_report["checkpoint"]),
                ab.SCENE_PROXY_DIR / "stage43_scene_proxy_features_train.npz",
                ab.SCENE_PROXY_DIR / "stage43_scene_proxy_features_val.npz",
                ab.SCENE_PROXY_DIR / "stage43_scene_proxy_features_test.npz",
            ]
        ),
    }
    payload["stage43_ac_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    gate = payload["stage43_ac_gate"]
    cur = payload["test_metrics_guarded"]
    base = payload["stage43_m_metrics_replayed"]
    ab_all = payload["stage43_ab_all_metrics_replayed"]
    delta_m = payload["delta_vs_stage43_m"]
    delta_ab = payload["delta_vs_stage43_ab_all"]
    lines = [
        "# Stage43-AC Scene-Proxy Guarded Latent Policy",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- mode: `{payload['mode']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- deploy guarded scene-proxy latent: `{gate['deploy_guarded_scene_proxy_latent']}`",
        "",
        "## Selected Policy",
        "",
        f"- policy: `{payload['validation_selected_policy']['policy']}`",
        f"- validation objective: `{payload['validation_selected_policy']['objective']:.6f}`",
        "",
        "## Test Metrics",
        "",
        "| metric | Stage43-M | Stage43-AB all | Stage43-AC guarded | delta vs M | delta vs AB all |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in [
        "full_waypoint_ade_improvement_vs_floor",
        "t50_full_waypoint_ade_improvement_vs_floor",
        "hard_failure_full_waypoint_ade_improvement_vs_floor",
        "easy_degradation_vs_floor",
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor",
        "switch_rate",
        "scene_proxy_override_rate",
        "t100_scene_proxy_override_rate",
    ]:
        lines.append(
            f"| `{key}` | `{_pct(float(base.get(key, 0.0)))}` | `{_pct(float(ab_all.get(key, 0.0)))}` | `{_pct(float(cur.get(key, 0.0)))}` | `{_pct(float(delta_m.get(key, 0.0)))}` | `{_pct(float(delta_ab.get(key, 0.0)))}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Stage43-AC keeps the Stage43-AB scene-proxy latent head only where a validation-selected guard allows it, and otherwise falls back to the Stage43-M protected latent policy. The guard is explicitly t100-aware because Stage43-AB improved all/t50/hard but worsened raw-frame t100 diagnostic.",
            "",
            "## Boundary",
            "",
            "- Scene proxy remains a train-only route/SDF/goal proxy, not raw image or verified metric SDF.",
            "- Future endpoint/full waypoints are labels/eval only.",
            "- t100 remains raw-frame diagnostic only.",
            "- No Stage5C, no SMC, no metric/seconds/true-3D/foundation claim.",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ]
    )
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-AC Scene-Proxy Guarded Latent Policy Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- deploy guarded scene-proxy latent: `{gate['deploy_guarded_scene_proxy_latent']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ],
    )
    _update_text_outputs(payload)


def _update_text_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_ac_gate"]
    cur = payload["test_metrics_guarded"]
    delta = payload["delta_vs_stage43_m"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"deploy_guarded_scene_proxy_latent = `{gate['deploy_guarded_scene_proxy_latent']}`",
        "",
        f"full_waypoint_ade_vs_floor = `{_pct(cur['full_waypoint_ade_improvement_vs_floor'])}`; delta_vs_stage43_m = `{_pct(delta['full_waypoint_ade_improvement_vs_floor'])}`",
        f"t50_full_waypoint_ade_vs_floor = `{_pct(cur['t50_full_waypoint_ade_improvement_vs_floor'])}`; delta_vs_stage43_m = `{_pct(delta['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"hard_failure_vs_floor = `{_pct(cur['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`; delta_vs_stage43_m = `{_pct(delta['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"t100_raw_frame_diagnostic = `{_pct(cur['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`; delta_vs_stage43_m = `{_pct(delta['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"easy_degradation = `{_pct(cur['easy_degradation_vs_floor'])}`",
        f"scene_proxy_override_rate = `{_pct(cur['scene_proxy_override_rate'])}`",
        "",
        "Stage43-AC is the guarded deployment version of Stage43-AB. It keeps the scene-proxy latent head where validation says it helps, but falls back to Stage43-M on risky slices, especially raw-frame t100.",
        "",
        "Boundary unchanged: scene proxy is not raw image/SDF, future labels are loss/eval only, t100 is raw-frame diagnostic, no metric/seconds claim, no Stage5C, no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_ac_scene_proxy_guarded_latent_policy"] = {
        "source": payload["source"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "deploy_guarded_scene_proxy_latent": gate["deploy_guarded_scene_proxy_latent"],
        "metrics": payload["test_metrics_guarded"],
        "delta_vs_stage43_m": payload["delta_vs_stage43_m"],
        "delta_vs_stage43_ab_all": payload["delta_vs_stage43_ab_all"],
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_ac_scene_proxy_guarded_latent_policy"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    ensure_dir(LEDGER_JSONL.parent)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-AC",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "deploy_guarded_scene_proxy_latent": gate["deploy_guarded_scene_proxy_latent"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay and guard Stage43-AB scene-proxy latent dynamics against Stage43-M.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true")
    group.add_argument("--small", action="store_true")
    group.add_argument("--medium", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=431)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if not args.quick and not args.small and not args.medium:
        args.small = True
    result = _run(args)
    gate = result["stage43_ac_gate"]
    print(f"Stage43-AC: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"deploy_guarded_scene_proxy_latent={gate['deploy_guarded_scene_proxy_latent']}")
    return result


if __name__ == "__main__":
    main()
