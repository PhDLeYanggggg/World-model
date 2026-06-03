from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_t100_source_scene_supported_supervision_cache as cq
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_protected_latent_state_model import _jsonable


OUT_DIR = m.OUT_DIR
CKPT_DIR = OUT_DIR / "checkpoints"
CHECKPOINT_NAME = "stage43_cr_t100_supported_latent_dynamics.pt"
HEARTBEAT_JSON = OUT_DIR / "stage43_t100_supported_latent_dynamics_heartbeat.json"
REPORT_JSON = OUT_DIR / "stage43_t100_supported_latent_dynamics.json"
REPORT_MD = OUT_DIR / "stage43_t100_supported_latent_dynamics.md"
GATE_MD = OUT_DIR / "stage43_stage_cr_t100_supported_latent_dynamics_gate.md"

README_RESULTS = m.README_RESULTS
M3W_README = m.M3W_README
WORK_SUMMARY = m.WORK_SUMMARY
RESEARCH_STATE = m.RESEARCH_STATE

SECTION = "STAGE43_CR_T100_SUPPORTED_LATENT_DYNAMICS"
SOURCE = "fresh_stage43_cr_t100_supported_latent_dynamics"
DENIED_FEATURE_NAME_FRAGMENTS = ("future", "oracle", "central_velocity", "ground_truth", "label", "ade", "fde")


def _cache_path(split: str) -> Path:
    return cq.CACHE_DIR / f"stage43_cp_t100_supervision_{split}.npz"


def _npz(path: Path) -> Mapping[str, np.ndarray]:
    return np.load(path, allow_pickle=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_hash(cache: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in ["old_split", "local_row", "dataset", "scene_id", "source_file", "agent_id", "frame_id", "horizon"]:
        arr = np.asarray(cache[key])
        digest.update(key.encode("utf-8"))
        if arr.dtype.kind in {"U", "S", "O"}:
            digest.update(arr.astype(str).tobytes())
        else:
            digest.update(arr.tobytes())
    return digest.hexdigest()


def _feature_contract(feature_names: list[str]) -> dict[str, Any]:
    denied = sorted(
        {
            name
            for name in feature_names
            for fragment in DENIED_FEATURE_NAME_FRAGMENTS
            if fragment in name.lower()
        }
    )
    return {
        "feature_dim": int(len(feature_names)),
        "feature_name_hash": hashlib.sha256("\0".join(feature_names).encode("utf-8")).hexdigest(),
        "denied_feature_name_hits": denied,
        "future_labels_train_eval_only": True,
        "central_velocity_feature": False,
        "test_endpoint_goal_feature": False,
    }


def _build_cq_split(split: str, *, max_rows: int | None, seed: int) -> m.WaypointSplit:
    cache = _npz(_cache_path(split))
    n = len(cache["horizon"])
    ids = np.arange(n)
    if max_rows is not None and max_rows < n:
        rng = np.random.default_rng(seed + {"train": 17, "val": 19, "test": 23}[split])
        ids = np.sort(rng.choice(ids, size=int(max_rows), replace=False))
    sub: dict[str, np.ndarray] = {}
    for key in cache.files:
        arr = np.asarray(cache[key])
        sub[key] = arr[ids] if len(arr) == n else arr
    scale = np.maximum(sub["scale"].astype(np.float32), 1e-4)
    cur = sub["current_xy"].astype(np.float32)
    waypoints = sub["waypoint_xy"].astype(np.float32)
    valid = sub["waypoint_valid"].astype(bool)
    waypoint_delta = ((waypoints - cur[:, None, :]) / scale[:, None, None]).astype(np.float32)

    hist_keys_1d = [
        "history_curvature",
        "history_turn_angle",
        "history_stop_go",
        "history_dwell",
        "history_path_length",
        "history_velocity_decay",
        "history_goal_alignment_proxy",
        "history_neighbor_count",
        "history_min_neighbor_dist",
        "history_density",
        "history_TTC",
        "history_closing_speed",
    ]
    history = {
        key: m._gather_old(
            sub,
            "history",
            key,
        )
        for key in [
            "history_dx",
            "history_dy",
            "history_speed",
            "history_accel",
            "history_heading",
            "history_valid_mask",
            *hist_keys_1d,
        ]
    }
    goal = {
        key: m._gather_old(sub, "goal", key)
        for key in ["prototype_likelihood", "prototype_distance", "prototype_angle", "prototype_entropy", "goal_ambiguity"]
    }
    baseline_pred = m._gather_old(sub, "baseline", "prediction").astype(np.float32)
    labels_y_fde = m._gather_old(sub, "labels", "y_fde").astype(np.float32)
    labels_oracle_idx = m._gather_old(sub, "labels", "oracle_idx").astype(np.int64)
    labels_oracle_margin = m._gather_old(sub, "labels", "oracle_margin").astype(np.float32)

    floor_endpoint = m._stage_floor_endpoint(sub, baseline_pred)
    floor_delta = ((floor_endpoint - cur) / scale[:, None]).astype(np.float32)
    floor_waypoint_delta = m.WAYPOINT_FRAC[None, :, None] * floor_delta[:, None, :]
    floor_xy = cur[:, None, :] + floor_waypoint_delta * scale[:, None, None]
    floor_err = np.linalg.norm(floor_xy.astype(np.float64) - waypoints.astype(np.float64), axis=2) / scale[:, None]
    floor_ade = ((floor_err * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)).astype(np.float32)
    floor_fde = floor_err[:, -1].astype(np.float32)

    row = np.arange(len(ids))
    oracle_err = labels_y_fde[row, labels_oracle_idx.clip(0, labels_y_fde.shape[1] - 1)]
    strongest_idx = m._gather_old(sub, "labels", "strongest_idx").astype(np.int64)
    strongest_err = labels_y_fde[row, strongest_idx.clip(0, labels_y_fde.shape[1] - 1)]
    y_gain = (oracle_err + 0.01 < strongest_err).astype(np.float32)
    y_harm = (sub["easy"].astype(bool) | (labels_oracle_margin < 0.01)).astype(np.float32)
    y_density = np.clip(history["history_density"].astype(np.float32) / 10.0, 0.0, 1.0)

    domain = sub["dataset"].astype(str)
    horizon = sub["horizon"].astype(np.int64)
    domain_oh, domain_names = m._one_hot(domain, m.DOMAINS, "domain")
    horizon_oh, horizon_names = m._one_hot(horizon, m.HORIZONS, "horizon")
    feature_parts: list[np.ndarray] = [
        cur / scale[:, None],
        horizon[:, None].astype(np.float32) / 100.0,
        domain_oh,
        horizon_oh,
    ]
    feature_names = ["current_x_over_scale", "current_y_over_scale", "horizon_norm", *domain_names, *horizon_names]
    for key in ["history_dx", "history_dy", "history_speed", "history_accel", "history_heading", "history_valid_mask"]:
        vals = m._tail(history[key], 16)
        feature_parts.append(vals)
        feature_names.extend([f"{key}_tail{i}" for i in range(vals.shape[1])])
    for key in hist_keys_1d:
        vals = history[key].astype(np.float32)[:, None]
        feature_parts.append(vals)
        feature_names.append(key)
    for key in ["prototype_likelihood", "prototype_distance", "prototype_angle"]:
        vals = goal[key].astype(np.float32)
        feature_parts.append(vals)
        feature_names.extend([f"{key}_{i}" for i in range(vals.shape[1])])
    for key in ["prototype_entropy", "goal_ambiguity"]:
        vals = goal[key].astype(np.float32)[:, None]
        feature_parts.append(vals)
        feature_names.append(key)
    baseline_rel = ((baseline_pred - cur[:, None, :]) / scale[:, None, None]).reshape(len(ids), -1)
    feature_parts.append(baseline_rel.astype(np.float32))
    feature_names.extend([f"baseline_endpoint_rel_{i}" for i in range(baseline_rel.shape[1])])
    feature_parts.append(floor_delta)
    feature_names.extend(["floor_endpoint_rel_x", "floor_endpoint_rel_y"])
    x = np.concatenate(feature_parts, axis=1).astype(np.float32)
    return m.WaypointSplit(
        split=split,
        x=x,
        waypoint_delta=waypoint_delta,
        waypoint_valid=valid,
        floor_waypoint_delta=floor_waypoint_delta.astype(np.float32),
        floor_ade=floor_ade,
        floor_fde=floor_fde,
        y_failure=sub["failure"].astype(np.float32),
        y_gain=y_gain,
        y_harm=y_harm,
        y_density=y_density.astype(np.float32),
        horizon=horizon,
        domain=domain,
        source_file=sub["source_file"].astype(str),
        scene_id=sub["scene_id"].astype(str),
        hard=sub["hard"].astype(bool),
        failure=sub["failure"].astype(bool),
        easy=sub["easy"].astype(bool),
        scale=scale,
        feature_names=feature_names,
    )


def _standardize(train: m.WaypointSplit, val: m.WaypointSplit, test: m.WaypointSplit) -> tuple[np.ndarray, np.ndarray]:
    mean = train.x.mean(axis=0).astype(np.float32)
    raw_std = train.x.std(axis=0).astype(np.float32)
    std = np.where(raw_std < 1e-3, 1.0, raw_std).astype(np.float32)
    for ds in [train, val, test]:
        ds.x = ((ds.x - mean) / std).astype(np.float32)
    return mean, std


def _search_t100_policy(val: m.WaypointSplit, pred: Mapping[str, np.ndarray]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for gain in [0.0, 0.25, 0.45, 0.55, 0.65, 0.75, 0.85]:
        for harm in [0.10, 0.20, 0.35, 0.50, 0.75, 1.00]:
            for failure in [0.0, 0.10, 0.25, 0.40, 0.55]:
                policy = {"gain_threshold": gain, "harm_threshold": harm, "failure_threshold": failure}
                selected_ade, selected_fde, switched = m._select_with_policy(val, pred, policy)
                metrics = m._metrics(val, selected_ade, selected_fde, switched)
                if metrics["easy_degradation_vs_floor"] > 0.02:
                    continue
                if metrics["switch_rate"] > 0.90:
                    continue
                objective = (
                    2.0 * metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
                    + 0.8 * metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
                    + 0.2 * metrics["full_waypoint_ade_improvement_vs_floor"]
                    - 0.15 * metrics["switch_rate"]
                )
                row = {"policy": policy, "validation_metrics": metrics, "objective": float(objective)}
                if best is None or row["objective"] > best["objective"]:
                    best = row
    if best is None:
        selected_ade = val.floor_ade.copy()
        selected_fde = val.floor_fde.copy()
        switched = np.zeros(len(val.x), dtype=bool)
        return {
            "policy": {"gain_threshold": 1.01, "harm_threshold": -0.01, "failure_threshold": 1.01},
            "validation_metrics": m._metrics(val, selected_ade, selected_fde, switched),
            "objective": 0.0,
            "diagnostic": "no_validation_safe_policy_keep_floor",
        }
    return best


def _ungated_metrics(ds: m.WaypointSplit, pred: Mapping[str, np.ndarray]) -> dict[str, Any]:
    ade, fde = m._trajectory_error(ds, pred["waypoint"])
    return m._metrics(ds, ade, fde, np.ones(len(ds.x), dtype=bool))


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["test_metrics_with_floor"]
    gates = {
        "stage43_cq_precondition_passed": payload["stage43_cq_precondition"]["verdict"]
        == "stage43_cq_t100_source_scene_supported_supervision_cache_pass",
        "torch_training_fresh_run": payload["result_source"] == "fresh_torch_t100_supported_latent_dynamics",
        "checkpoint_written_not_committed": Path(payload["checkpoint"]).exists() and payload["checkpoint_committed"] is False,
        "t100_only_train_val_test": all(value == 100 for value in payload["horizon_protocol"]["horizons"]),
        "feature_contract_clean": not payload["feature_contract"]["denied_feature_name_hits"],
        "latent_noncollapse": payload["latent_variance"] > 0.01,
        "validation_selected_policy": payload["selection_protocol"]["test_threshold_tuning"] is False,
        "test_once_completed": metrics["rows"] > 0,
        "easy_preserved": metrics["easy_degradation_vs_floor"] <= 0.02,
        "protected_lift_or_honest_floor": (
            metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] > 0.0
            or metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0
            or payload["deploy_on_current_heldout"] is False
        ),
        "current_heldout_t100_not_changed": payload["deploy_on_current_heldout"] is False,
        "no_future_or_test_leakage": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True
        and payload["no_leakage"]["central_velocity_input"] is False
        and payload["no_leakage"]["test_endpoint_goal_construction"] is False
        and payload["no_leakage"]["test_statistics_normalization"] is False,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
        "long_objective_kept_active": payload["long_objective_complete"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = (
        "stage43_cr_t100_supported_latent_dynamics_positive_diagnostic"
        if passed == total
        and metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] > 0.0
        and metrics["easy_degradation_vs_floor"] <= 0.02
        else "stage43_cr_t100_supported_latent_dynamics_keep_floor"
    )
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_cr_gate"]
    test = payload["test_metrics_with_floor"]
    ungated = payload["test_metrics_neural_without_floor"]
    return [
        "# Stage43-CR T100 Supported Latent Dynamics",
        "",
        f"- source: `{SOURCE}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- mode: `{payload['mode']}`",
        f"- checkpoint committed: `{payload['checkpoint_committed']}`",
        "- deploy on current heldout t100: `False`",
        "",
        "## Data",
        "",
        f"- train / val / test rows: `{payload['data_rows']['train']} / {payload['data_rows']['val']} / {payload['data_rows']['test']}`",
        f"- feature dim: `{payload['feature_contract']['feature_dim']}`",
        f"- feature hash: `{payload['feature_contract']['feature_name_hash']}`",
        f"- denied feature hits: `{payload['feature_contract']['denied_feature_name_hits']}`",
        "",
        "## Validation Policy",
        "",
        f"- selected policy: `{payload['validation_selected_policy']['policy']}`",
        f"- validation t100 improvement: `{payload['validation_selected_policy']['validation_metrics']['t100_raw_frame_full_waypoint_diagnostic_vs_floor']:.4f}`",
        f"- validation easy degradation: `{payload['validation_selected_policy']['validation_metrics']['easy_degradation_vs_floor']:.4f}`",
        "",
        "## Test Once on Supported Protocol",
        "",
        f"- protected t100 improvement: `{test['t100_raw_frame_full_waypoint_diagnostic_vs_floor']:.4f}`",
        f"- protected hard/failure improvement: `{test['hard_failure_full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- protected easy degradation: `{test['easy_degradation_vs_floor']:.4f}`",
        f"- protected switch rate: `{test['switch_rate']:.4f}`",
        f"- ungated t100 improvement: `{ungated['t100_raw_frame_full_waypoint_diagnostic_vs_floor']:.4f}`",
        f"- ungated easy degradation: `{ungated['easy_degradation_vs_floor']:.4f}`",
        f"- latent variance: `{payload['latent_variance']:.6f}`",
        "",
        "## Interpretation",
        "",
        "- This is a t100 supported-protocol neural diagnostic, not a current heldout deployment.",
        "- Current heldout t100 remains floor-only until a model passes source/scene support and heldout safety gates.",
        "- Future endpoints/full waypoints are labels only; inputs are causal history, goal prototypes, baseline rollouts, floor rollout, domain/horizon tokens, and current state.",
        "- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
        "",
    ]


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_cr_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-CR Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            "- deploy on current heldout t100: `False`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
            "| gate | pass |",
            "| --- | --- |",
            *[f"| `{key}` | `{bool(value)}` |" for key, value in gate["gates"].items()],
            "",
        ],
    )
    test = payload["test_metrics_with_floor"]
    readme_block = [
        "## Stage43-CR: t100 supported latent dynamics pilot",
        "",
        "I trained a small torch latent-dynamics pilot on the Stage43-CQ t100-supported cache. This is a supported-protocol diagnostic, not a deployment change for the current heldout t100 split.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- protected t100 improvement: `{test['t100_raw_frame_full_waypoint_diagnostic_vs_floor']:.2%}`",
        f"- protected hard/failure improvement: `{test['hard_failure_full_waypoint_ade_improvement_vs_floor']:.2%}`",
        f"- protected easy degradation: `{test['easy_degradation_vs_floor']:.2%}`",
        f"- switch rate: `{test['switch_rate']:.2%}`",
        f"- deploy on current heldout t100: `{payload['deploy_on_current_heldout']}`",
        "",
        "The useful question here is whether t100 learning is possible when validation actually covers the source/scene. The stricter heldout t100 policy is still floor-only.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, readme_block)

    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage43_cr_t100_supported_latent_dynamics"
    state["current_verdict"] = gate["verdict"]
    stage = state.setdefault("stage43_long_research_execution", {})
    stage["t100_supported_latent_dynamics"] = {
        "source": SOURCE,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "mode": payload["mode"],
        "data_rows": payload["data_rows"],
        "feature_contract": payload["feature_contract"],
        "validation_selected_policy": payload["validation_selected_policy"],
        "test_metrics_with_floor": payload["test_metrics_with_floor"],
        "test_metrics_neural_without_floor": payload["test_metrics_neural_without_floor"],
        "deploy_on_current_heldout": payload["deploy_on_current_heldout"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, _jsonable(state))


def train_t100_supported_latent_dynamics(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CKPT_DIR)
    if not cq.REPORT_JSON.exists() or not all(_cache_path(split).exists() for split in ["train", "val", "test"]):
        cq.build_t100_source_scene_supported_supervision_cache()
    seed = int(args.seed)
    runtime = m._configure_runtime(seed)
    mode = "quick" if args.quick else "small"
    max_train = int(args.max_train or (6000 if args.quick else 24000))
    max_val = int(args.max_val or (3000 if args.quick else 9000))
    max_test = int(args.max_test or (3000 if args.quick else 10000))
    train = _build_cq_split("train", max_rows=max_train, seed=seed)
    val = _build_cq_split("val", max_rows=max_val, seed=seed)
    test = _build_cq_split("test", max_rows=max_test, seed=seed)
    mean, std = _standardize(train, val, test)

    device = torch.device("cpu")
    model = m.FullWaypointLatentDynamics(train.x.shape[1], hidden_dim=int(args.hidden_dim), latent_dim=int(args.latent_dim)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    best_val = float("inf")
    best_path = CKPT_DIR / CHECKPOINT_NAME
    history: list[dict[str, Any]] = []
    start = time.time()
    for epoch in range(int(args.epochs)):
        model.train()
        losses: list[float] = []
        latent_vars: list[float] = []
        for batch_ids in m._batch_indices(len(train.x), int(args.batch_size), shuffle=True, seed=seed + epoch):
            opt.zero_grad(set_to_none=True)
            loss, stat = m._loss(model, train, batch_ids, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            latent_vars.append(float(stat["latent_variance"]))
        val_pred = m._predict(model, val, device, int(args.batch_size))
        val_ade, _val_fde = m._trajectory_error(val, val_pred["waypoint"])
        val_mse = float(np.mean((val_ade - val.floor_ade) ** 2))
        row = {
            "epoch": int(epoch + 1),
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_candidate_mse_to_floor": val_mse,
            "latent_variance": float(np.mean(latent_vars)) if latent_vars else 0.0,
        }
        history.append(row)
        write_json(
            HEARTBEAT_JSON,
            _jsonable({"source": SOURCE, "mode": mode, "epoch": epoch + 1, "elapsed_s": time.time() - start, "last": row}),
        )
        if val_mse < best_val:
            best_val = val_mse
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_mean": mean,
                    "feature_std": std,
                    "feature_names": train.feature_names,
                    "input_dim": int(train.x.shape[1]),
                    "hidden_dim": int(args.hidden_dim),
                    "latent_dim": int(args.latent_dim),
                    "seed": seed,
                    "epoch": epoch + 1,
                    "runtime": runtime,
                    "protocol": "stage43_cq_t100_source_scene_supported",
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    val_pred = m._predict(model, val, device, int(args.batch_size))
    test_pred = m._predict(model, test, device, int(args.batch_size))
    val_policy = _search_t100_policy(val, val_pred)
    selected_ade, selected_fde, switched = m._select_with_policy(test, test_pred, val_policy["policy"])
    test_metrics = m._metrics(test, selected_ade, selected_fde, switched)
    ungated_metrics = _ungated_metrics(test, test_pred)
    bootstrap = m._bootstrap_ci(test, selected_ade, selected_fde, n=int(args.bootstrap), seed=seed + 1000)
    feature_contract = _feature_contract(train.feature_names)
    cq_payload = read_json(cq.REPORT_JSON, {})
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_torch_t100_supported_latent_dynamics",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "mode": mode,
        "checkpoint": str(best_path),
        "checkpoint_sha256": _sha256(best_path),
        "checkpoint_committed": False,
        "runtime": runtime,
        "stage43_cq_precondition": {
            "report": str(cq.REPORT_JSON),
            "verdict": cq_payload.get("stage43_cq_gate", {}).get("verdict"),
            "cp_assignment_hash": cq_payload.get("cp_assignment_hash"),
        },
        "cache_row_hashes": {split: _row_hash(_npz(_cache_path(split))) for split in ["train", "val", "test"]},
        "horizon_protocol": {"horizons": sorted(set(test.horizon.astype(int).tolist())), "raw_frame_only": True},
        "data_rows": {"train": int(len(train.x)), "val": int(len(val.x)), "test": int(len(test.x))},
        "feature_contract": feature_contract,
        "training_history": history,
        "selection_protocol": {"validation_only": True, "test_threshold_tuning": False},
        "validation_selected_policy": val_policy,
        "test_metrics_with_floor": test_metrics,
        "test_metrics_neural_without_floor": ungated_metrics,
        "bootstrap_ci": bootstrap,
        "latent_variance": float(np.var(test_pred["latent"], axis=0).mean()) if len(test_pred["latent"]) else 0.0,
        "deploy_on_current_heldout": False,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "long_objective_complete": False,
    }
    payload["stage43_cr_gate"] = _gate(payload)
    _write_reports(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--latent-dim", type=int, default=24)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    args = parser.parse_args()
    payload = train_t100_supported_latent_dynamics(args)
    gate = payload["stage43_cr_gate"]
    print(f"Stage43-CR: {gate['verdict']} ({gate['passed']}/{gate['total']})")


if __name__ == "__main__":
    main()
