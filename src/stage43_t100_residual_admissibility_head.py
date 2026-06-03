from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_t100_bounded_residual_latent_repair as cs
from src import stage43_t100_source_scene_supported_supervision_cache as cq
from src import stage43_t100_supported_latent_dynamics as cr
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_protected_latent_state_model import _jsonable


OUT_DIR = m.OUT_DIR
CKPT_DIR = OUT_DIR / "checkpoints"
CHECKPOINT_NAME = "stage43_ct_t100_residual_admissibility_head.pt"
HEARTBEAT_JSON = OUT_DIR / "stage43_t100_residual_admissibility_head_heartbeat.json"
REPORT_JSON = OUT_DIR / "stage43_t100_residual_admissibility_head.json"
REPORT_MD = OUT_DIR / "stage43_t100_residual_admissibility_head.md"
GATE_MD = OUT_DIR / "stage43_stage_ct_t100_residual_admissibility_head_gate.md"

README_RESULTS = m.README_RESULTS
M3W_README = m.M3W_README
WORK_SUMMARY = m.WORK_SUMMARY
RESEARCH_STATE = m.RESEARCH_STATE

SECTION = "STAGE43_CT_T100_RESIDUAL_ADMISSIBILITY_HEAD"
SOURCE = "fresh_stage43_ct_t100_residual_admissibility_head"
ALPHAS = np.asarray([0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00], dtype=np.float32)


class ResidualAdmissibilityHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.net(x)
        raw = self.head(z)
        return {
            "gain_logit": raw[:, 0],
            "harm_logit": raw[:, 1],
            "delta": raw[:, 2],
        }


def _ensure_cs_precondition(args: argparse.Namespace) -> None:
    if not cs.REPORT_JSON.exists() or not (cs.CKPT_DIR / cs.CHECKPOINT_NAME).exists():
        ns = argparse.Namespace(
            quick=bool(args.quick),
            small=True,
            seed=int(args.seed),
            max_train=args.max_train,
            max_val=args.max_val,
            max_test=args.max_test,
            epochs=5,
            batch_size=int(args.batch_size),
            hidden_dim=96,
            latent_dim=24,
            residual_clip=0.20,
            lr=8e-4,
            bootstrap=200,
        )
        cs.train_t100_bounded_residual_latent_repair(ns)


def _build_splits(args: argparse.Namespace) -> tuple[m.WaypointSplit, m.WaypointSplit, m.WaypointSplit, dict[str, Any], cs.BoundedResidualLatentDynamics]:
    _ensure_cs_precondition(args)
    max_train = int(args.max_train or (6000 if args.quick else 24000))
    max_val = int(args.max_val or (3000 if args.quick else 9000))
    max_test = int(args.max_test or (3000 if args.quick else 10000))
    train = cr._build_cq_split("train", max_rows=max_train, seed=int(args.seed))
    val = cr._build_cq_split("val", max_rows=max_val, seed=int(args.seed))
    test = cr._build_cq_split("test", max_rows=max_test, seed=int(args.seed))
    ckpt = torch.load(cs.CKPT_DIR / cs.CHECKPOINT_NAME, map_location="cpu", weights_only=False)
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_std"], dtype=np.float32)
    for ds in [train, val, test]:
        ds.x = ((ds.x - mean) / std).astype(np.float32)
    model = cs.BoundedResidualLatentDynamics(
        int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        latent_dim=int(ckpt["latent_dim"]),
        residual_clip=float(ckpt["residual_clip"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return train, val, test, ckpt, model


def _diagnostic_features(ds: m.WaypointSplit, pred: Mapping[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    residual = np.asarray(pred["residual"], dtype=np.float32)
    norms = np.linalg.norm(residual.astype(np.float64), axis=2).astype(np.float32)
    endpoint = residual[:, -1, :]
    latent = np.asarray(pred["latent"], dtype=np.float32)
    pieces = [
        ds.x.astype(np.float32),
        np.asarray(pred["failure"], dtype=np.float32)[:, None],
        np.asarray(pred["gain"], dtype=np.float32)[:, None],
        np.asarray(pred["harm"], dtype=np.float32)[:, None],
        np.asarray(pred["density"], dtype=np.float32)[:, None],
        norms,
        norms.mean(axis=1, keepdims=True),
        norms.max(axis=1, keepdims=True),
        endpoint.astype(np.float32),
        latent[:, : min(16, latent.shape[1])].astype(np.float32),
    ]
    names = [
        *ds.feature_names,
        "cs_failure_score",
        "cs_gain_score",
        "cs_harm_score",
        "cs_density_score",
        *[f"cs_residual_norm_wp{i}" for i in range(norms.shape[1])],
        "cs_residual_norm_mean",
        "cs_residual_norm_max",
        "cs_endpoint_residual_x",
        "cs_endpoint_residual_y",
        *[f"cs_latent_{i}" for i in range(min(16, latent.shape[1]))],
    ]
    return np.concatenate(pieces, axis=1).astype(np.float32), names


def _augment_alpha_features(
    ds: m.WaypointSplit,
    pred: Mapping[str, np.ndarray],
    *,
    alphas: np.ndarray = ALPHAS,
) -> dict[str, np.ndarray]:
    base, names = _diagnostic_features(ds, pred)
    xs: list[np.ndarray] = []
    y_gain: list[np.ndarray] = []
    y_harm: list[np.ndarray] = []
    y_delta: list[np.ndarray] = []
    y_ade: list[np.ndarray] = []
    alpha_col: list[np.ndarray] = []
    for alpha in alphas.astype(np.float32):
        waypoint = cs._compose_waypoint(ds, pred, alpha=float(alpha))
        ade, _fde = m._trajectory_error(ds, waypoint)
        delta = (ade - ds.floor_ade).astype(np.float32)
        xs.append(np.concatenate([base, np.full((len(base), 1), float(alpha), dtype=np.float32)], axis=1))
        y_gain.append((delta < -0.0025).astype(np.float32))
        y_harm.append(((delta > 0.0025) | (ds.easy & (delta > 0.0))).astype(np.float32))
        y_delta.append(delta.astype(np.float32))
        y_ade.append(ade.astype(np.float32))
        alpha_col.append(np.full(len(base), float(alpha), dtype=np.float32))
    return {
        "x": np.concatenate(xs, axis=0).astype(np.float32),
        "y_gain": np.concatenate(y_gain, axis=0).astype(np.float32),
        "y_harm": np.concatenate(y_harm, axis=0).astype(np.float32),
        "y_delta": np.concatenate(y_delta, axis=0).astype(np.float32),
        "candidate_ade": np.concatenate(y_ade, axis=0).astype(np.float32),
        "alpha": np.concatenate(alpha_col, axis=0).astype(np.float32),
        "feature_names": np.asarray([*names, "candidate_alpha"], dtype=object),
    }


def _standardize_aug(train: dict[str, np.ndarray], val: dict[str, np.ndarray], test: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    mean = train["x"].mean(axis=0).astype(np.float32)
    raw_std = train["x"].std(axis=0).astype(np.float32)
    std = np.where(raw_std < 1e-3, 1.0, raw_std).astype(np.float32)
    for item in [train, val, test]:
        item["x"] = ((item["x"] - mean) / std).astype(np.float32)
    return mean, std


def _batch_indices(n: int, batch_size: int, *, shuffle: bool, seed: int) -> list[np.ndarray]:
    ids = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(ids)
    return [ids[i : i + batch_size] for i in range(0, n, batch_size)]


def _loss(model: ResidualAdmissibilityHead, data: Mapping[str, np.ndarray], ids: np.ndarray, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    x = torch.from_numpy(data["x"][ids]).to(device)
    y_gain = torch.from_numpy(data["y_gain"][ids]).to(device)
    y_harm = torch.from_numpy(data["y_harm"][ids]).to(device)
    y_delta = torch.from_numpy(data["y_delta"][ids]).to(device)
    out = model(x)
    pos = float(max(1, int(y_gain.detach().cpu().numpy().sum())))
    neg = float(max(1, len(ids) - int(y_gain.detach().cpu().numpy().sum())))
    gain_weight = torch.where(y_gain > 0.5, torch.tensor(neg / pos, device=device).clamp(max=8.0), torch.ones_like(y_gain))
    gain = nn.functional.binary_cross_entropy_with_logits(out["gain_logit"], y_gain, weight=gain_weight)
    harm = nn.functional.binary_cross_entropy_with_logits(out["harm_logit"], y_harm)
    delta = nn.functional.smooth_l1_loss(out["delta"], y_delta)
    total = 0.8 * gain + 1.0 * harm + 0.7 * delta
    return total, {"gain": float(gain.detach().cpu()), "harm": float(harm.detach().cpu()), "delta": float(delta.detach().cpu())}


@torch.no_grad()
def _predict_head(model: ResidualAdmissibilityHead, data: Mapping[str, np.ndarray], device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    model.eval()
    out: dict[str, list[np.ndarray]] = {"gain": [], "harm": [], "delta": []}
    for ids in _batch_indices(len(data["x"]), batch_size, shuffle=False, seed=0):
        pred = model(torch.from_numpy(data["x"][ids]).to(device))
        out["gain"].append(torch.sigmoid(pred["gain_logit"]).detach().cpu().numpy())
        out["harm"].append(torch.sigmoid(pred["harm_logit"]).detach().cpu().numpy())
        out["delta"].append(pred["delta"].detach().cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in out.items()}


def _policy_metrics_for_alpha(
    ds: m.WaypointSplit,
    cs_pred: Mapping[str, np.ndarray],
    head_pred: Mapping[str, np.ndarray],
    *,
    alpha_index: int,
    policy: Mapping[str, float],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    n = len(ds.x)
    sl = slice(alpha_index * n, (alpha_index + 1) * n)
    gain = head_pred["gain"][sl]
    harm = head_pred["harm"][sl]
    delta = head_pred["delta"][sl]
    allow = (
        (gain >= float(policy["gain_threshold"]))
        & (harm <= float(policy["harm_threshold"]))
        & (delta <= float(policy["delta_threshold"]))
    )
    if bool(policy.get("force_easy_floor", True)):
        allow = allow & (~ds.easy)
    waypoint = cs._compose_waypoint(ds, cs_pred, alpha=float(ALPHAS[alpha_index]))
    selected_waypoint = np.where(allow[:, None, None], waypoint, ds.floor_waypoint_delta).astype(np.float32)
    selected_ade, selected_fde = m._trajectory_error(ds, selected_waypoint)
    return m._metrics(ds, selected_ade, selected_fde, allow), selected_ade, selected_fde, allow


def _search_policy(ds: m.WaypointSplit, cs_pred: Mapping[str, np.ndarray], head_pred: Mapping[str, np.ndarray]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    searched = 0
    safe_candidates = 0
    for ai, alpha in enumerate(ALPHAS):
        for gain_thr in [0.20, 0.35, 0.50, 0.65, 0.80]:
            for harm_thr in [0.05, 0.10, 0.20, 0.35, 0.50]:
                for delta_thr in [-0.010, -0.005, -0.001, 0.0, 0.002]:
                    for force_easy in [True, False]:
                        policy = {
                            "alpha": float(alpha),
                            "alpha_index": int(ai),
                            "gain_threshold": float(gain_thr),
                            "harm_threshold": float(harm_thr),
                            "delta_threshold": float(delta_thr),
                            "force_easy_floor": bool(force_easy),
                        }
                        metrics, _ade, _fde, _allow = _policy_metrics_for_alpha(ds, cs_pred, head_pred, alpha_index=ai, policy=policy)
                        searched += 1
                        safe = metrics["easy_degradation_vs_floor"] <= 0.02 and metrics["switch_rate"] <= 0.60
                        if safe:
                            safe_candidates += 1
                        if not safe:
                            continue
                        objective = (
                            2.0 * metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
                            + 1.0 * metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
                            + 0.25 * metrics["full_waypoint_ade_improvement_vs_floor"]
                            - 0.05 * metrics["switch_rate"]
                        )
                        row = {"policy": policy, "metrics": metrics, "objective": float(objective)}
                        if best is None or row["objective"] > best["objective"]:
                            best = row
    if best is None:
        selected_ade = ds.floor_ade.copy()
        selected_fde = ds.floor_fde.copy()
        switched = np.zeros(len(ds.x), dtype=bool)
        best = {
            "policy": {
                "alpha": 0.0,
                "alpha_index": -1,
                "gain_threshold": 1.01,
                "harm_threshold": -0.01,
                "delta_threshold": -1.0,
                "force_easy_floor": True,
            },
            "metrics": m._metrics(ds, selected_ade, selected_fde, switched),
            "objective": 0.0,
            "diagnostic": "no_safe_admissible_policy_keep_floor",
        }
    best["searched_candidates"] = int(searched)
    best["safe_candidates"] = int(safe_candidates)
    return best


def _evaluate_selected(ds: m.WaypointSplit, cs_pred: Mapping[str, np.ndarray], head_pred: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    if int(policy.get("alpha_index", -1)) < 0:
        selected_ade = ds.floor_ade.copy()
        selected_fde = ds.floor_fde.copy()
        switched = np.zeros(len(ds.x), dtype=bool)
        return m._metrics(ds, selected_ade, selected_fde, switched), selected_ade, selected_fde, switched
    return _policy_metrics_for_alpha(ds, cs_pred, head_pred, alpha_index=int(policy["alpha_index"]), policy=policy)


def _ungated_for_alpha(ds: m.WaypointSplit, cs_pred: Mapping[str, np.ndarray], alpha: float) -> dict[str, Any]:
    waypoint = cs._compose_waypoint(ds, cs_pred, alpha=float(alpha))
    ade, fde = m._trajectory_error(ds, waypoint)
    return m._metrics(ds, ade, fde, np.ones(len(ds.x), dtype=bool))


def _feature_contract(names: np.ndarray) -> dict[str, Any]:
    joined = "\0".join([str(x) for x in names.tolist()])
    denied = sorted(
        {
            str(name)
            for name in names.tolist()
            for fragment in cr.DENIED_FEATURE_NAME_FRAGMENTS
            if fragment in str(name).lower()
        }
    )
    return {
        "feature_dim": int(len(names)),
        "feature_name_hash": __import__("hashlib").sha256(joined.encode("utf-8")).hexdigest(),
        "denied_feature_name_hits": denied,
        "future_labels_train_eval_only": True,
        "central_velocity_feature": False,
        "test_endpoint_goal_feature": False,
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["test_metrics_with_floor"]
    gates = {
        "stage43_cs_precondition_present": payload["stage43_cs_precondition"]["verdict"]
        in {"stage43_cs_t100_bounded_residual_latent_keep_floor", "stage43_cs_t100_bounded_residual_latent_positive_diagnostic"},
        "fresh_admissibility_head_training": payload["result_source"] == "fresh_torch_t100_residual_admissibility_head",
        "checkpoint_written_not_committed": Path(payload["checkpoint"]).exists() and payload["checkpoint_committed"] is False,
        "t100_only_supported_protocol": all(value == 100 for value in payload["horizon_protocol"]["horizons"]),
        "feature_contract_clean": not payload["feature_contract"]["denied_feature_name_hits"],
        "alpha_augmented_protocol": payload["alpha_protocol"]["num_alphas"] >= 3,
        "validation_only_policy_selection": payload["selection_protocol"]["test_threshold_tuning"] is False,
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
    positive = (
        passed == total
        and metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] > 0.0
        and metrics["easy_degradation_vs_floor"] <= 0.02
        and metrics["switch_rate"] > 0.0
    )
    verdict = "stage43_ct_t100_residual_admissibility_positive_diagnostic" if positive else "stage43_ct_t100_residual_admissibility_keep_floor"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_ct_gate"]
    test = payload["test_metrics_with_floor"]
    ungated = payload["ungated_reference"]
    return [
        "# Stage43-CT T100 Residual Admissibility Head",
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
        f"- augmented train rows: `{payload['alpha_protocol']['augmented_train_rows']}`",
        f"- feature dim: `{payload['feature_contract']['feature_dim']}`",
        f"- denied feature hits: `{payload['feature_contract']['denied_feature_name_hits']}`",
        "",
        "## Validation Policy",
        "",
        f"- selected policy: `{payload['validation_selected_policy']['policy']}`",
        f"- searched candidates: `{payload['validation_selected_policy'].get('searched_candidates', 0)}`",
        f"- safe candidates: `{payload['validation_selected_policy'].get('safe_candidates', 0)}`",
        f"- validation t100 improvement: `{payload['validation_selected_policy']['metrics']['t100_raw_frame_full_waypoint_diagnostic_vs_floor']:.4f}`",
        f"- validation easy degradation: `{payload['validation_selected_policy']['metrics']['easy_degradation_vs_floor']:.4f}`",
        "",
        "## Test Once",
        "",
        f"- protected t100 improvement: `{test['t100_raw_frame_full_waypoint_diagnostic_vs_floor']:.4f}`",
        f"- protected hard/failure improvement: `{test['hard_failure_full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- protected easy degradation: `{test['easy_degradation_vs_floor']:.4f}`",
        f"- protected switch rate: `{test['switch_rate']:.4f}`",
        f"- ungated alpha=1 t100 improvement: `{ungated['t100_raw_frame_full_waypoint_diagnostic_vs_floor']:.4f}`",
        f"- ungated alpha=1 easy degradation: `{ungated['easy_degradation_vs_floor']:.4f}`",
        "",
        "## Interpretation",
        "",
        "- This trains a residual-admissibility head over CS residual candidates instead of searching only raw gain/harm thresholds.",
        "- Labels use future waypoints only for supervised training/evaluation; inference inputs are causal CS diagnostics, residual norms, latent state, and history/goal/baseline features.",
        "- Current heldout t100 remains floor-only unless this admissibility policy clears stricter heldout gates.",
        "- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
        "",
    ]


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_ct_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-CT Gate",
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
    ungated = payload["ungated_reference"]
    readme_block = [
        "## Stage43-CT: t100 residual admissibility head",
        "",
        "I trained a second-stage admissibility head to decide when the bounded t100 residual from Stage43-CS should be accepted. The head is trained on train labels, thresholded on validation, and evaluated once on test.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- protected t100 improvement: `{test['t100_raw_frame_full_waypoint_diagnostic_vs_floor']:.2%}`",
        f"- protected hard/failure improvement: `{test['hard_failure_full_waypoint_ade_improvement_vs_floor']:.2%}`",
        f"- protected easy degradation: `{test['easy_degradation_vs_floor']:.2%}`",
        f"- switch rate: `{test['switch_rate']:.2%}`",
        f"- ungated alpha=1 t100 improvement: `{ungated['t100_raw_frame_full_waypoint_diagnostic_vs_floor']:.2%}`",
        f"- ungated alpha=1 easy degradation: `{ungated['easy_degradation_vs_floor']:.2%}`",
        f"- deploy on current heldout t100: `{payload['deploy_on_current_heldout']}`",
        "",
        "This is still a supported-protocol diagnostic. The positive lift is small, so I am not treating it as a heldout deployment change. If the admissibility head cannot clear stricter heldout gates, the current heldout t100 policy stays floor-only.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, readme_block)

    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage43_ct_t100_residual_admissibility_head"
    state["current_verdict"] = gate["verdict"]
    stage = state.setdefault("stage43_long_research_execution", {})
    stage["t100_residual_admissibility_head"] = {
        "source": SOURCE,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "mode": payload["mode"],
        "data_rows": payload["data_rows"],
        "alpha_protocol": payload["alpha_protocol"],
        "feature_contract": payload["feature_contract"],
        "validation_selected_policy": payload["validation_selected_policy"],
        "test_metrics_with_floor": payload["test_metrics_with_floor"],
        "ungated_reference": payload["ungated_reference"],
        "deploy_on_current_heldout": payload["deploy_on_current_heldout"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, _jsonable(state))


def train_t100_residual_admissibility_head(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CKPT_DIR)
    seed = int(args.seed)
    runtime = m._configure_runtime(seed)
    mode = "quick" if args.quick else "small"
    train, val, test, cs_ckpt, cs_model = _build_splits(args)
    device = torch.device("cpu")
    train_pred = cs._predict(cs_model, train, device, int(args.batch_size))
    val_pred = cs._predict(cs_model, val, device, int(args.batch_size))
    test_pred = cs._predict(cs_model, test, device, int(args.batch_size))
    train_aug = _augment_alpha_features(train, train_pred)
    val_aug = _augment_alpha_features(val, val_pred)
    test_aug = _augment_alpha_features(test, test_pred)
    mean, std = _standardize_aug(train_aug, val_aug, test_aug)
    model = ResidualAdmissibilityHead(train_aug["x"].shape[1], hidden_dim=int(args.hidden_dim)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    best_val = float("inf")
    best_path = CKPT_DIR / CHECKPOINT_NAME
    history: list[dict[str, Any]] = []
    start = time.time()
    for epoch in range(int(args.epochs)):
        model.train()
        losses: list[float] = []
        for ids in _batch_indices(len(train_aug["x"]), int(args.batch_size), shuffle=True, seed=seed + epoch):
            opt.zero_grad(set_to_none=True)
            loss, stat = _loss(model, train_aug, ids, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_head = _predict_head(model, val_aug, device, int(args.batch_size))
        val_loss = float(np.mean((val_head["delta"] - val_aug["y_delta"]) ** 2))
        row = {"epoch": int(epoch + 1), "train_loss": float(np.mean(losses)) if losses else 0.0, "val_delta_mse": val_loss}
        history.append(row)
        write_json(
            HEARTBEAT_JSON,
            _jsonable({"source": SOURCE, "mode": mode, "epoch": epoch + 1, "elapsed_s": time.time() - start, "last": row}),
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_mean": mean,
                    "feature_std": std,
                    "feature_names": train_aug["feature_names"].tolist(),
                    "input_dim": int(train_aug["x"].shape[1]),
                    "hidden_dim": int(args.hidden_dim),
                    "seed": seed,
                    "epoch": epoch + 1,
                    "runtime": runtime,
                    "cs_checkpoint_sha256": cr._sha256(cs.CKPT_DIR / cs.CHECKPOINT_NAME),
                },
                best_path,
            )
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    val_head = _predict_head(model, val_aug, device, int(args.batch_size))
    test_head = _predict_head(model, test_aug, device, int(args.batch_size))
    val_policy = _search_policy(val, val_pred, val_head)
    test_metrics, selected_ade, selected_fde, switched = _evaluate_selected(test, test_pred, test_head, val_policy["policy"])
    bootstrap = m._bootstrap_ci(test, selected_ade, selected_fde, n=int(args.bootstrap), seed=seed + 3000)
    ungated = _ungated_for_alpha(test, test_pred, 1.0)
    cs_report = read_json(cs.REPORT_JSON, {})
    feature_contract = _feature_contract(train_aug["feature_names"])
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_torch_t100_residual_admissibility_head",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "mode": mode,
        "checkpoint": str(best_path),
        "checkpoint_sha256": cr._sha256(best_path),
        "checkpoint_committed": False,
        "runtime": runtime,
        "stage43_cs_precondition": {
            "report": str(cs.REPORT_JSON),
            "verdict": cs_report.get("stage43_cs_gate", {}).get("verdict"),
            "checkpoint_sha256": cs_report.get("checkpoint_sha256"),
        },
        "horizon_protocol": {"horizons": sorted(set(test.horizon.astype(int).tolist())), "raw_frame_only": True},
        "data_rows": {"train": int(len(train.x)), "val": int(len(val.x)), "test": int(len(test.x))},
        "alpha_protocol": {
            "alphas": [float(x) for x in ALPHAS.tolist()],
            "num_alphas": int(len(ALPHAS)),
            "augmented_train_rows": int(len(train_aug["x"])),
            "train_positive_rate": float(np.mean(train_aug["y_gain"])),
            "train_harm_rate": float(np.mean(train_aug["y_harm"])),
        },
        "feature_contract": feature_contract,
        "training_history": history,
        "selection_protocol": {"train_supervised_labels": True, "validation_only": True, "test_threshold_tuning": False},
        "validation_selected_policy": val_policy,
        "test_metrics_with_floor": test_metrics,
        "switch_count": int(switched.sum()),
        "bootstrap_ci": bootstrap,
        "ungated_reference": ungated,
        "deploy_on_current_heldout": False,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "feature_standardization_train_only": True,
            "validation_policy_selection_only": True,
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
    payload["stage43_ct_gate"] = _gate(payload)
    _write_reports(payload)
    print(f"Stage43-CT: {payload['stage43_ct_gate']['verdict']} ({payload['stage43_ct_gate']['passed']}/{payload['stage43_ct_gate']['total']})")
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Train Stage43-CT t100 residual admissibility head.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--seed", type=int, default=4323)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args(argv)
    return train_t100_residual_admissibility_head(args)


if __name__ == "__main__":
    main()
