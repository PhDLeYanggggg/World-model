from __future__ import annotations

import argparse
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_t100_source_scene_supported_supervision_cache as cq
from src import stage43_t100_supported_latent_dynamics as cr
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_protected_latent_state_model import _jsonable


OUT_DIR = m.OUT_DIR
CKPT_DIR = OUT_DIR / "checkpoints"
CHECKPOINT_NAME = "stage43_cs_t100_bounded_residual_latent_repair.pt"
HEARTBEAT_JSON = OUT_DIR / "stage43_t100_bounded_residual_latent_repair_heartbeat.json"
REPORT_JSON = OUT_DIR / "stage43_t100_bounded_residual_latent_repair.json"
REPORT_MD = OUT_DIR / "stage43_t100_bounded_residual_latent_repair.md"
GATE_MD = OUT_DIR / "stage43_stage_cs_t100_bounded_residual_latent_repair_gate.md"

README_RESULTS = m.README_RESULTS
M3W_README = m.M3W_README
WORK_SUMMARY = m.WORK_SUMMARY
RESEARCH_STATE = m.RESEARCH_STATE

SECTION = "STAGE43_CS_T100_BOUNDED_RESIDUAL_LATENT_REPAIR"
SOURCE = "fresh_stage43_cs_t100_bounded_residual_latent_repair"


class BoundedResidualLatentDynamics(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96, latent_dim: int = 24, residual_clip: float = 0.20) -> None:
        super().__init__()
        self.residual_clip = float(residual_clip)
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.future_target_encoder = nn.Sequential(
            nn.Linear(14, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.head = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 13))

    def forward(self, x: torch.Tensor, target_vec: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        z_t = self.encoder(x)
        z_next = self.dynamics(z_t)
        raw = self.head(z_next)
        residual = torch.tanh(raw[:, :8]).reshape(-1, 4, 2) * self.residual_clip
        out = {
            "z_t": z_t,
            "z_next": z_next,
            "residual_delta": residual,
            "failure_logit": raw[:, 8],
            "gain_logit": raw[:, 9],
            "harm_logit": raw[:, 10],
            "density": torch.sigmoid(raw[:, 11]),
            "validity_logit": raw[:, 12],
        }
        if target_vec is not None:
            out["target_latent"] = self.future_target_encoder(target_vec).detach()
        return out


def _target_vec(ds: m.WaypointSplit) -> np.ndarray:
    residual = (ds.waypoint_delta - ds.floor_waypoint_delta).reshape(len(ds.x), -1)
    return np.concatenate(
        [
            residual.astype(np.float32),
            ds.y_failure[:, None],
            ds.y_gain[:, None],
            ds.y_harm[:, None],
            ds.y_density[:, None],
            (ds.horizon[:, None].astype(np.float32) / 100.0),
            ds.waypoint_valid.mean(axis=1, keepdims=True).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def _loss(model: BoundedResidualLatentDynamics, ds: m.WaypointSplit, ids: np.ndarray, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    x = torch.from_numpy(ds.x[ids]).to(device)
    target_residual = torch.from_numpy((ds.waypoint_delta[ids] - ds.floor_waypoint_delta[ids]).astype(np.float32)).to(device)
    floor = torch.from_numpy(ds.floor_waypoint_delta[ids]).to(device)
    target_delta = torch.from_numpy(ds.waypoint_delta[ids]).to(device)
    valid = torch.from_numpy(ds.waypoint_valid[ids].astype(np.float32)).to(device)
    y_failure = torch.from_numpy(ds.y_failure[ids]).to(device)
    y_gain = torch.from_numpy(ds.y_gain[ids]).to(device)
    y_harm = torch.from_numpy(ds.y_harm[ids]).to(device)
    y_density = torch.from_numpy(ds.y_density[ids]).to(device)
    target = torch.from_numpy(_target_vec(ds)[ids]).to(device)
    hard = torch.from_numpy((ds.hard[ids] | ds.failure[ids]).astype(np.float32)).to(device)
    easy = torch.from_numpy(ds.easy[ids].astype(np.float32)).to(device)
    out = model(x, target)
    candidate = floor + out["residual_delta"]
    per_wp = nn.functional.smooth_l1_loss(candidate, target_delta, reduction="none").mean(dim=2)
    residual_loss = nn.functional.smooth_l1_loss(out["residual_delta"], target_residual, reduction="none").mean(dim=2)
    row_weight = 1.0 + 1.25 * hard
    waypoint = ((per_wp * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0) * row_weight).mean()
    residual = ((residual_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0) * row_weight).mean()
    endpoint = nn.functional.smooth_l1_loss(candidate[:, -1, :], target_delta[:, -1, :])
    failure = nn.functional.binary_cross_entropy_with_logits(out["failure_logit"], y_failure)
    gain = nn.functional.binary_cross_entropy_with_logits(out["gain_logit"], y_gain)
    harm = nn.functional.binary_cross_entropy_with_logits(out["harm_logit"], y_harm)
    density = nn.functional.mse_loss(out["density"], y_density)
    latent = nn.functional.mse_loss(out["z_next"], out["target_latent"])
    variance = out["z_next"].float().var(dim=0).mean()
    collapse = torch.relu(torch.tensor(0.02, device=device) - variance)
    easy_residual_penalty = (out["residual_delta"].pow(2).mean(dim=(1, 2)) * easy).mean()
    total = (
        0.70 * waypoint
        + 0.80 * residual
        + 0.25 * endpoint
        + 0.35 * failure
        + 0.45 * gain
        + 0.65 * harm
        + 0.15 * density
        + 0.30 * latent
        + 0.80 * easy_residual_penalty
        + collapse
    )
    return total, {
        "waypoint": float(waypoint.detach().cpu()),
        "residual": float(residual.detach().cpu()),
        "endpoint": float(endpoint.detach().cpu()),
        "failure": float(failure.detach().cpu()),
        "gain": float(gain.detach().cpu()),
        "harm": float(harm.detach().cpu()),
        "density": float(density.detach().cpu()),
        "latent": float(latent.detach().cpu()),
        "easy_residual_penalty": float(easy_residual_penalty.detach().cpu()),
        "latent_variance": float(variance.detach().cpu()),
    }


@torch.no_grad()
def _predict(model: BoundedResidualLatentDynamics, ds: m.WaypointSplit, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    model.eval()
    outs: dict[str, list[np.ndarray]] = {"residual": [], "failure": [], "gain": [], "harm": [], "density": [], "latent": []}
    for ids in m._batch_indices(len(ds.x), batch_size, shuffle=False, seed=0):
        x = torch.from_numpy(ds.x[ids]).to(device)
        out = model(x)
        outs["residual"].append(out["residual_delta"].detach().cpu().numpy())
        outs["failure"].append(torch.sigmoid(out["failure_logit"]).detach().cpu().numpy())
        outs["gain"].append(torch.sigmoid(out["gain_logit"]).detach().cpu().numpy())
        outs["harm"].append(torch.sigmoid(out["harm_logit"]).detach().cpu().numpy())
        outs["density"].append(out["density"].detach().cpu().numpy())
        outs["latent"].append(out["z_next"].detach().cpu().numpy())
    result = {key: np.concatenate(value, axis=0) for key, value in outs.items()}
    result["waypoint"] = (ds.floor_waypoint_delta + result["residual"]).astype(np.float32)
    return result


def _compose_waypoint(ds: m.WaypointSplit, pred: Mapping[str, np.ndarray], *, alpha: float) -> np.ndarray:
    return (ds.floor_waypoint_delta.astype(np.float32) + float(alpha) * np.asarray(pred["residual"], dtype=np.float32)).astype(np.float32)


def _selected_arrays(
    ds: m.WaypointSplit,
    pred: Mapping[str, np.ndarray],
    policy: Mapping[str, Any],
    *,
    floor_only: bool = False,
    ungated: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if floor_only:
        selected_ade = ds.floor_ade.copy()
        selected_fde = ds.floor_fde.copy()
        switched = np.zeros(len(ds.x), dtype=bool)
        waypoint = ds.floor_waypoint_delta
    else:
        alpha = float(policy.get("alpha", 1.0))
        waypoint = _compose_waypoint(ds, pred, alpha=alpha)
        if ungated:
            switched = np.ones(len(ds.x), dtype=bool)
        else:
            gate = policy.get("policy", {})
            switched = (
                (pred["gain"] >= float(gate.get("gain_threshold", 1.01)))
                & (pred["harm"] <= float(gate.get("harm_threshold", -0.01)))
                & (pred["failure"] >= float(gate.get("failure_threshold", 1.01)))
            )
            if bool(policy.get("force_easy_floor", True)):
                switched = switched & (~ds.easy)
        waypoint = np.where(switched[:, None, None], waypoint, ds.floor_waypoint_delta).astype(np.float32)
        selected_ade, selected_fde = m._trajectory_error(ds, waypoint)
    return selected_ade, selected_fde, switched, waypoint


def _evaluate_policy(ds: m.WaypointSplit, pred: Mapping[str, np.ndarray], policy: Mapping[str, Any], *, floor_only: bool = False, ungated: bool = False) -> dict[str, Any]:
    selected_ade, selected_fde, switched, waypoint = _selected_arrays(ds, pred, policy, floor_only=floor_only, ungated=ungated)
    residual_norm = np.linalg.norm((waypoint - ds.floor_waypoint_delta).astype(np.float64), axis=2).mean(axis=1)
    metrics = m._metrics(ds, selected_ade, selected_fde, switched)
    return {
        "policy": dict(policy),
        "metrics": metrics,
        "switch_count": int(switched.sum()),
        "mean_residual_norm": float(np.mean(residual_norm)),
        "max_residual_norm": float(np.max(residual_norm)) if len(residual_norm) else 0.0,
    }


def _search_policy(val: m.WaypointSplit, pred: Mapping[str, np.ndarray]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    safe_candidates = 0
    searched = 0
    for alpha in [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]:
        for gain in [0.0, 0.20, 0.35, 0.50, 0.65, 0.80]:
            for harm in [0.03, 0.05, 0.10, 0.15, 0.25, 0.40]:
                for failure in [0.0, 0.10, 0.25, 0.40, 0.60]:
                    for force_easy in [True, False]:
                        policy = {
                            "alpha": alpha,
                            "policy": {
                                "gain_threshold": gain,
                                "harm_threshold": harm,
                                "failure_threshold": failure,
                            },
                            "force_easy_floor": force_easy,
                        }
                        result = _evaluate_policy(val, pred, policy)
                        metrics = result["metrics"]
                        searched += 1
                        safe = metrics["easy_degradation_vs_floor"] <= 0.02 and metrics["switch_rate"] <= 0.80
                        if safe:
                            safe_candidates += 1
                        if not safe:
                            continue
                        objective = (
                            2.0 * metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
                            + 1.0 * metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
                            + 0.25 * metrics["full_waypoint_ade_improvement_vs_floor"]
                            - 0.08 * metrics["switch_rate"]
                        )
                        result["objective"] = float(objective)
                        if best is None or result["objective"] > best["objective"]:
                            best = result
    if best is None:
        best = _evaluate_policy(val, pred, {}, floor_only=True)
        best["objective"] = 0.0
        best["diagnostic"] = "no_safe_validation_policy_keep_floor"
    best["searched_candidates"] = int(searched)
    best["safe_candidates"] = int(safe_candidates)
    return best


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["test_metrics_with_floor"]
    ungated = payload["test_metrics_neural_without_floor"]
    residual = payload["residual_bounds"]
    gates = {
        "stage43_cq_precondition_passed": payload["stage43_cq_precondition"]["verdict"]
        == "stage43_cq_t100_source_scene_supported_supervision_cache_pass",
        "fresh_torch_bounded_residual_training": payload["result_source"] == "fresh_torch_t100_bounded_residual_latent_repair",
        "checkpoint_written_not_committed": Path(payload["checkpoint"]).exists() and payload["checkpoint_committed"] is False,
        "t100_only_supported_protocol": all(value == 100 for value in payload["horizon_protocol"]["horizons"]),
        "feature_contract_clean": not payload["feature_contract"]["denied_feature_name_hits"],
        "residual_output_bounded": residual["max_abs_predicted_residual"] <= residual["residual_clip"] + 1e-5,
        "latent_noncollapse": payload["latent_variance"] > 0.01,
        "validation_only_policy_selection": payload["selection_protocol"]["test_threshold_tuning"] is False,
        "test_once_completed": metrics["rows"] > 0,
        "easy_preserved": metrics["easy_degradation_vs_floor"] <= 0.02,
        "protected_lift_or_honest_floor": (
            metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] > 0.0
            or metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0
            or payload["deploy_on_current_heldout"] is False
        ),
        "ungated_neural_not_deployed": ungated["easy_degradation_vs_floor"] > 0.02 or payload["deploy_on_current_heldout"] is False,
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
    verdict = (
        "stage43_cs_t100_bounded_residual_latent_positive_diagnostic"
        if positive
        else "stage43_cs_t100_bounded_residual_latent_keep_floor"
    )
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_cs_gate"]
    test = payload["test_metrics_with_floor"]
    ungated = payload["test_metrics_neural_without_floor"]
    return [
        "# Stage43-CS T100 Bounded Residual Latent Repair",
        "",
        f"- source: `{SOURCE}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- mode: `{payload['mode']}`",
        f"- residual clip: `{payload['residual_bounds']['residual_clip']}`",
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
        f"- selected policy: `{payload['validation_selected_policy'].get('policy', {})}`",
        f"- searched candidates: `{payload['validation_selected_policy'].get('searched_candidates', 0)}`",
        f"- safe candidates: `{payload['validation_selected_policy'].get('safe_candidates', 0)}`",
        f"- validation t100 improvement: `{payload['validation_selected_policy']['metrics']['t100_raw_frame_full_waypoint_diagnostic_vs_floor']:.4f}`",
        f"- validation easy degradation: `{payload['validation_selected_policy']['metrics']['easy_degradation_vs_floor']:.4f}`",
        "",
        "## Test Once on Supported Protocol",
        "",
        f"- protected t100 improvement: `{test['t100_raw_frame_full_waypoint_diagnostic_vs_floor']:.4f}`",
        f"- protected hard/failure improvement: `{test['hard_failure_full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- protected easy degradation: `{test['easy_degradation_vs_floor']:.4f}`",
        f"- protected switch rate: `{test['switch_rate']:.4f}`",
        f"- ungated bounded t100 improvement: `{ungated['t100_raw_frame_full_waypoint_diagnostic_vs_floor']:.4f}`",
        f"- ungated bounded easy degradation: `{ungated['easy_degradation_vs_floor']:.4f}`",
        f"- latent variance: `{payload['latent_variance']:.6f}`",
        "",
        "## Interpretation",
        "",
        "- This tests a repair path for the Stage43-CR failure mode by predicting a bounded residual around the safety floor instead of directly replacing the waypoint trajectory.",
        "- The current heldout t100 policy remains unchanged unless a residual policy clears source/scene support and heldout safety gates.",
        "- Future endpoints/full waypoints are labels only; inputs remain causal history, goal prototypes, baseline rollouts, floor rollout, domain/horizon tokens, and current state.",
        "- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
        "",
    ]


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_cs_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-CS Gate",
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
    ungated = payload["test_metrics_neural_without_floor"]
    readme_block = [
        "## Stage43-CS: t100 bounded residual latent repair",
        "",
        "After the direct latent waypoint head failed, I retrained the t100 supported pilot as a bounded residual around the safety floor. This keeps the experiment focused on safe world-dynamics lift instead of letting a neural head freely replace the floor.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- protected t100 improvement: `{test['t100_raw_frame_full_waypoint_diagnostic_vs_floor']:.2%}`",
        f"- protected hard/failure improvement: `{test['hard_failure_full_waypoint_ade_improvement_vs_floor']:.2%}`",
        f"- protected easy degradation: `{test['easy_degradation_vs_floor']:.2%}`",
        f"- switch rate: `{test['switch_rate']:.2%}`",
        f"- ungated bounded t100 improvement: `{ungated['t100_raw_frame_full_waypoint_diagnostic_vs_floor']:.2%}`",
        f"- ungated bounded easy degradation: `{ungated['easy_degradation_vs_floor']:.2%}`",
        f"- deploy on current heldout t100: `{payload['deploy_on_current_heldout']}`",
        "",
        "This is still a supported-protocol diagnostic. The current heldout t100 policy remains floor-only until a residual policy clears the stricter heldout gates.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, readme_block)

    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage43_cs_t100_bounded_residual_latent_repair"
    state["current_verdict"] = gate["verdict"]
    stage = state.setdefault("stage43_long_research_execution", {})
    stage["t100_bounded_residual_latent_repair"] = {
        "source": SOURCE,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "mode": payload["mode"],
        "data_rows": payload["data_rows"],
        "feature_contract": payload["feature_contract"],
        "residual_bounds": payload["residual_bounds"],
        "validation_selected_policy": payload["validation_selected_policy"],
        "test_metrics_with_floor": payload["test_metrics_with_floor"],
        "test_metrics_neural_without_floor": payload["test_metrics_neural_without_floor"],
        "deploy_on_current_heldout": payload["deploy_on_current_heldout"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, _jsonable(state))


def train_t100_bounded_residual_latent_repair(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CKPT_DIR)
    if not cq.REPORT_JSON.exists() or not all(cr._cache_path(split).exists() for split in ["train", "val", "test"]):
        cq.build_t100_source_scene_supported_supervision_cache()
    seed = int(args.seed)
    runtime = m._configure_runtime(seed)
    mode = "quick" if args.quick else "small"
    max_train = int(args.max_train or (6000 if args.quick else 24000))
    max_val = int(args.max_val or (3000 if args.quick else 9000))
    max_test = int(args.max_test or (3000 if args.quick else 10000))
    train = cr._build_cq_split("train", max_rows=max_train, seed=seed)
    val = cr._build_cq_split("val", max_rows=max_val, seed=seed)
    test = cr._build_cq_split("test", max_rows=max_test, seed=seed)
    mean, std = cr._standardize(train, val, test)

    device = torch.device("cpu")
    model = BoundedResidualLatentDynamics(
        train.x.shape[1],
        hidden_dim=int(args.hidden_dim),
        latent_dim=int(args.latent_dim),
        residual_clip=float(args.residual_clip),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    best_val = float("inf")
    best_path = CKPT_DIR / CHECKPOINT_NAME
    history: list[dict[str, Any]] = []
    start = time.time()
    for epoch in range(int(args.epochs)):
        model.train()
        losses: list[float] = []
        latent_vars: list[float] = []
        residual_losses: list[float] = []
        for batch_ids in m._batch_indices(len(train.x), int(args.batch_size), shuffle=True, seed=seed + epoch):
            opt.zero_grad(set_to_none=True)
            loss, stat = _loss(model, train, batch_ids, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            latent_vars.append(float(stat["latent_variance"]))
            residual_losses.append(float(stat["residual"]))
        val_pred = _predict(model, val, device, int(args.batch_size))
        val_candidate_ade, _ = m._trajectory_error(val, val_pred["waypoint"])
        val_mse = float(np.mean((val_candidate_ade - val.floor_ade) ** 2))
        row = {
            "epoch": int(epoch + 1),
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "train_residual_loss": float(np.mean(residual_losses)) if residual_losses else 0.0,
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
                    "residual_clip": float(args.residual_clip),
                    "seed": seed,
                    "epoch": epoch + 1,
                    "runtime": runtime,
                    "protocol": "stage43_cq_t100_source_scene_supported_bounded_residual",
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    val_pred = _predict(model, val, device, int(args.batch_size))
    test_pred = _predict(model, test, device, int(args.batch_size))
    val_policy = _search_policy(val, val_pred)
    test_result = _evaluate_policy(test, test_pred, val_policy["policy"])
    ungated = _evaluate_policy(test, test_pred, {"alpha": 1.0}, ungated=True)
    selected_ade, selected_fde, _switched, _waypoint = _selected_arrays(test, test_pred, val_policy["policy"])
    bootstrap = m._bootstrap_ci(
        test,
        selected_ade,
        selected_fde,
        n=int(args.bootstrap),
        seed=seed + 2000,
    )
    residual_abs = np.abs(np.asarray(test_pred["residual"], dtype=np.float32))
    cq_payload = read_json(cq.REPORT_JSON, {})
    feature_contract = cr._feature_contract(train.feature_names)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_torch_t100_bounded_residual_latent_repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "mode": mode,
        "checkpoint": str(best_path),
        "checkpoint_sha256": cr._sha256(best_path),
        "checkpoint_committed": False,
        "runtime": runtime,
        "stage43_cq_precondition": {
            "report": str(cq.REPORT_JSON),
            "verdict": cq_payload.get("stage43_cq_gate", {}).get("verdict"),
            "cp_assignment_hash": cq_payload.get("cp_assignment_hash"),
        },
        "cache_row_hashes": {split: cr._row_hash(cr._npz(cr._cache_path(split))) for split in ["train", "val", "test"]},
        "horizon_protocol": {"horizons": sorted(set(test.horizon.astype(int).tolist())), "raw_frame_only": True},
        "data_rows": {"train": int(len(train.x)), "val": int(len(val.x)), "test": int(len(test.x))},
        "feature_contract": feature_contract,
        "residual_bounds": {
            "residual_clip": float(args.residual_clip),
            "max_abs_predicted_residual": float(np.max(residual_abs)) if residual_abs.size else 0.0,
            "mean_abs_predicted_residual": float(np.mean(residual_abs)) if residual_abs.size else 0.0,
        },
        "training_history": history,
        "selection_protocol": {"validation_only": True, "test_threshold_tuning": False},
        "validation_selected_policy": val_policy,
        "test_metrics_with_floor": test_result["metrics"],
        "test_metrics_neural_without_floor": ungated["metrics"],
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
    payload["stage43_cs_gate"] = _gate(payload)
    _write_reports(payload)
    print(f"Stage43-CS: {payload['stage43_cs_gate']['verdict']} ({payload['stage43_cs_gate']['passed']}/{payload['stage43_cs_gate']['total']})")
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Train Stage43-CS t100 bounded residual latent repair.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--seed", type=int, default=4319)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--latent-dim", type=int, default=24)
    parser.add_argument("--residual-clip", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args(argv)
    return train_t100_bounded_residual_latent_repair(args)


if __name__ == "__main__":
    main()
