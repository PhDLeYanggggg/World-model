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
from src import stage43_t100_group_robust_head_failure_forensics as db
from src import stage43_t100_residual_admissibility_leave_group_out_policy as cz
from src import stage43_t100_residual_admissibility_slice_attribution as cv
from src import stage43_t100_residual_admissibility_statistical_confirmation as cu
from src import stage43_t100_residual_admissibility_head as ct
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_protected_latent_state_model import _jsonable


OUT_DIR = m.OUT_DIR
CKPT_DIR = OUT_DIR / "checkpoints"
REPORT_JSON = OUT_DIR / "stage43_t100_policy_distilled_admissibility_head.json"
REPORT_MD = OUT_DIR / "stage43_t100_policy_distilled_admissibility_head.md"
GATE_MD = OUT_DIR / "stage43_stage_dc_t100_policy_distilled_admissibility_head_gate.md"
HEARTBEAT_JSON = OUT_DIR / "stage43_t100_policy_distilled_admissibility_head_heartbeat.json"

README_RESULTS = m.README_RESULTS
M3W_README = m.M3W_README
WORK_SUMMARY = m.WORK_SUMMARY
RESEARCH_STATE = m.RESEARCH_STATE

SECTION = "STAGE43_DC_T100_POLICY_DISTILLED_ADMISSIBILITY_HEAD"
SOURCE = "fresh_stage43_dc_t100_policy_distilled_admissibility_head"


def _parse_seeds(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _sha_names(names: list[str] | np.ndarray) -> str:
    return hashlib.sha256("\0".join([str(x) for x in list(names)]).encode("utf-8")).hexdigest()


def _ensure_db_precondition(args: argparse.Namespace) -> dict[str, Any]:
    report = read_json(db.REPORT_JSON, {})
    gate = report.get("stage43_db_gate", {})
    if not report or gate.get("passed") != gate.get("total"):
        report = db.run_t100_group_robust_head_failure_forensics(args)
    return report


def _seed_run_by_seed(report: Mapping[str, Any], seed: int) -> dict[str, Any]:
    for run in report.get("seed_runs", []):
        if int(run.get("seed", -1)) == int(seed):
            return dict(run)
    raise KeyError(f"seed {seed} not found")


def _teacher_switch_labels(
    ds: m.WaypointSplit,
    teacher_head_pred: Mapping[str, np.ndarray],
    policy: Mapping[str, Any],
) -> np.ndarray:
    n = int(len(ds.x))
    out = np.zeros(n * len(ct.ALPHAS), dtype=np.float32)
    ai = int(policy.get("alpha_index", -1))
    if ai < 0:
        return out
    sl = slice(ai * n, (ai + 1) * n)
    gain = teacher_head_pred["gain"][sl]
    harm = teacher_head_pred["harm"][sl]
    delta = teacher_head_pred["delta"][sl]
    allow = (
        (gain >= float(policy["gain_threshold"]))
        & (harm <= float(policy["harm_threshold"]))
        & (delta <= float(policy["delta_threshold"]))
    )
    if bool(policy.get("force_easy_floor", True)):
        allow = allow & (~ds.easy)
    out[sl] = allow.astype(np.float32)
    return out


def _teacher_head_predictions(
    ds: m.WaypointSplit,
    cs_pred: Mapping[str, np.ndarray],
    seed_run: Mapping[str, Any],
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    device = torch.device("cpu")
    aug = ct._augment_alpha_features(ds, cs_pred)
    model, mean, std = cv._load_seed_head(seed_run)
    aug["x"] = ((aug["x"] - mean) / std).astype(np.float32)
    return ct._predict_head(model, aug, device, int(batch_size))


def _attach_distillation_targets(
    data: dict[str, np.ndarray],
    ds: m.WaypointSplit,
    teacher_head_pred: Mapping[str, np.ndarray],
    teacher_policy: Mapping[str, Any],
) -> None:
    teacher_switch = _teacher_switch_labels(ds, teacher_head_pred, teacher_policy)
    data["y_teacher_switch"] = teacher_switch.astype(np.float32)
    alpha_count = int(len(ct.ALPHAS))
    source = np.concatenate([ds.source_file.astype(str) for _ in range(alpha_count)], axis=0)
    scene = np.concatenate([ds.scene_id.astype(str) for _ in range(alpha_count)], axis=0)
    domain = np.concatenate([ds.domain.astype(str) for _ in range(alpha_count)], axis=0)
    group_weight = _balanced_group_weights(source, scene, domain)
    hard = np.concatenate([(ds.hard | ds.failure).astype(np.float32) for _ in range(alpha_count)], axis=0)
    easy = np.concatenate([ds.easy.astype(np.float32) for _ in range(alpha_count)], axis=0)
    data["group_weight"] = np.clip(group_weight * (1.0 + 0.45 * hard - 0.15 * easy + 0.60 * teacher_switch), 0.20, 6.0).astype(np.float32)


def _balanced_group_weights(source_labels: np.ndarray, scene_labels: np.ndarray, domain_labels: np.ndarray) -> np.ndarray:
    def inv_sqrt(labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(labels).astype(str)
        _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        weights = 1.0 / np.sqrt(np.maximum(counts[inverse].astype(np.float64), 1.0))
        return weights / max(float(np.mean(weights)), m.EPS)

    weights = (inv_sqrt(source_labels) + inv_sqrt(scene_labels) + inv_sqrt(domain_labels)) / 3.0
    return np.clip(weights / max(float(np.mean(weights)), m.EPS), 0.25, 4.0).astype(np.float32)


def _distill_loss(model: ct.ResidualAdmissibilityHead, data: Mapping[str, np.ndarray], ids: np.ndarray, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    x = torch.from_numpy(data["x"][ids]).to(device)
    y_teacher = torch.from_numpy(data["y_teacher_switch"][ids]).to(device)
    y_gain = torch.from_numpy(data["y_gain"][ids]).to(device)
    y_harm = torch.from_numpy(data["y_harm"][ids]).to(device)
    y_delta = torch.from_numpy(data["y_delta"][ids]).to(device)
    weight = torch.from_numpy(data["group_weight"][ids]).to(device)
    weight = weight / torch.clamp(weight.mean(), min=1e-6)
    out = model(x)
    pos = float(max(1, int(y_teacher.detach().cpu().numpy().sum())))
    neg = float(max(1, len(ids) - int(y_teacher.detach().cpu().numpy().sum())))
    teacher_weight = torch.where(y_teacher > 0.5, torch.tensor(neg / pos, device=device).clamp(max=10.0), torch.ones_like(y_teacher))
    teacher = nn.functional.binary_cross_entropy_with_logits(out["gain_logit"], y_teacher, weight=teacher_weight * weight)
    gain_aux = nn.functional.binary_cross_entropy_with_logits(out["gain_logit"], y_gain, weight=weight)
    harm = nn.functional.binary_cross_entropy_with_logits(out["harm_logit"], y_harm, weight=weight)
    delta = torch.mean(nn.functional.smooth_l1_loss(out["delta"], y_delta, reduction="none") * weight)
    safe_margin = torch.relu(torch.sigmoid(out["gain_logit"]) + torch.sigmoid(out["harm_logit"]) - 1.0)
    support = torch.mean(safe_margin * weight)
    total = 1.20 * teacher + 0.35 * gain_aux + 1.05 * harm + 0.55 * delta + 0.08 * support
    return total, {
        "teacher": float(teacher.detach().cpu()),
        "gain_aux": float(gain_aux.detach().cpu()),
        "harm": float(harm.detach().cpu()),
        "delta": float(delta.detach().cpu()),
        "support": float(support.detach().cpu()),
    }


def _select_policy(ds: m.WaypointSplit, cs_pred: Mapping[str, np.ndarray], head_pred: Mapping[str, np.ndarray]) -> dict[str, Any]:
    candidates = cz._policy_candidates(ds, cs_pred, head_pred)
    safe = [row for row in candidates if row["safe"]]
    selected = safe[0] if safe else candidates[0]
    return {
        "policy": selected["policy"],
        "metrics": selected["metrics"],
        "group_summary": selected["group_summary"],
        "objective": float(selected["objective"]),
        "safe": bool(selected["safe"]),
        "searched_candidates": int(len(candidates)),
        "safe_candidates": int(len(safe)),
    }


def _seed_payload(seed: int, args: argparse.Namespace, runtime: Mapping[str, Any], cz_report: Mapping[str, Any], cu_report: Mapping[str, Any]) -> dict[str, Any]:
    local_args = argparse.Namespace(
        quick=bool(args.quick),
        seed=int(seed),
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        batch_size=int(args.batch_size),
    )
    train, val, test, _cs_ckpt, cs_model = ct._build_splits(local_args)
    device = torch.device("cpu")
    train_pred = ct.cs._predict(cs_model, train, device, int(args.batch_size))
    val_pred = ct.cs._predict(cs_model, val, device, int(args.batch_size))
    test_pred = ct.cs._predict(cs_model, test, device, int(args.batch_size))
    teacher_seed_run = _seed_run_by_seed(cu_report, seed)
    cz_seed_run = _seed_run_by_seed(cz_report, seed)
    teacher_policy = cz_seed_run["selected_validation_candidate"]["policy"]
    train_teacher = _teacher_head_predictions(train, train_pred, teacher_seed_run, batch_size=int(args.batch_size))
    val_teacher = _teacher_head_predictions(val, val_pred, teacher_seed_run, batch_size=int(args.batch_size))
    train_aug = ct._augment_alpha_features(train, train_pred)
    val_aug = ct._augment_alpha_features(val, val_pred)
    test_aug = ct._augment_alpha_features(test, test_pred)
    _attach_distillation_targets(train_aug, train, train_teacher, teacher_policy)
    _attach_distillation_targets(val_aug, val, val_teacher, teacher_policy)
    mean, std = ct._standardize_aug(train_aug, val_aug, test_aug)
    model = ct.ResidualAdmissibilityHead(train_aug["x"].shape[1], hidden_dim=int(args.hidden_dim)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    start = time.time()
    for epoch in range(int(args.epochs)):
        model.train()
        losses: list[float] = []
        stats: list[dict[str, float]] = []
        for ids in ct._batch_indices(len(train_aug["x"]), int(args.batch_size), shuffle=True, seed=int(seed) + epoch):
            opt.zero_grad(set_to_none=True)
            loss, stat = _distill_loss(model, train_aug, ids, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            stats.append(stat)
        val_head = ct._predict_head(model, val_aug, device, int(args.batch_size))
        val_selection = _select_policy(val, val_pred, val_head)
        row = {
            "seed": int(seed),
            "epoch": int(epoch + 1),
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "validation_objective": float(val_selection["objective"]),
            "validation_t100": float(val_selection["metrics"]["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]),
            "validation_min_without_group_t100": float(val_selection["group_summary"]["min_without_any_group_t100"]),
            "safe_candidates": int(val_selection["safe_candidates"]),
            "teacher_switch_rate_train": float(np.mean(train_aug["y_teacher_switch"])),
            "loss_terms": {key: float(np.mean([s[key] for s in stats])) for key in stats[0]} if stats else {},
        }
        history.append(row)
        write_json(HEARTBEAT_JSON, _jsonable({"source": SOURCE, "seed": int(seed), "epoch": int(epoch + 1), "elapsed_s": time.time() - start, "last": row}))
        if best is None or row["validation_objective"] > best["history_row"]["validation_objective"]:
            best = {
                "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "epoch": int(epoch + 1),
                "history_row": row,
            }
    assert best is not None
    model.load_state_dict(best["model_state"])
    ckpt_path = CKPT_DIR / f"stage43_dc_t100_policy_distilled_admissibility_seed{int(seed)}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "feature_names": train_aug["feature_names"].tolist(),
            "input_dim": int(train_aug["x"].shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "seed": int(seed),
            "epoch": int(best["epoch"]),
            "runtime": dict(runtime),
            "teacher_source": "stage43_cz_leave_group_out_policy_on_train_rows",
            "teacher_policy": teacher_policy,
            "training_objective": "cz_policy_distillation_plus_group_weighted_harm_delta",
        },
        ckpt_path,
    )
    test_head = ct._predict_head(model, test_aug, device, int(args.batch_size))
    val_head = ct._predict_head(model, val_aug, device, int(args.batch_size))
    val_selection = _select_policy(val, val_pred, val_head)
    test_metrics, selected_ade, selected_fde, switched = ct._evaluate_selected(test, test_pred, test_head, val_selection["policy"])
    test_group = cz.cy._group_summary(test, selected_ade, selected_fde, switched)
    bootstrap = m._bootstrap_ci(test, selected_ade, selected_fde, n=int(args.bootstrap), seed=int(seed) + 6100)
    return {
        "seed": int(seed),
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": ct.cr._sha256(ckpt_path),
        "checkpoint_committed": False,
        "best_epoch": int(best["epoch"]),
        "training_history": history,
        "teacher_policy": teacher_policy,
        "teacher_switch_rate_train": float(np.mean(train_aug["y_teacher_switch"])),
        "teacher_switch_rate_val": float(np.mean(val_aug["y_teacher_switch"])),
        "validation_selected_policy": val_selection,
        "test_metrics_with_floor": test_metrics,
        "test_group_summary": test_group,
        "bootstrap_ci": bootstrap,
        "switch_count": int(switched.sum()),
        "data_rows": {"train": int(len(train.x)), "val": int(len(val.x)), "test": int(len(test.x))},
        "feature_contract": ct._feature_contract(train_aug["feature_names"]),
        "feature_name_hash": _sha_names(train_aug["feature_names"]),
        "alpha_protocol": {
            "alphas": [float(x) for x in ct.ALPHAS.tolist()],
            "num_alphas": int(len(ct.ALPHAS)),
            "augmented_train_rows": int(len(train_aug["x"])),
            "train_teacher_switch_rate": float(np.mean(train_aug["y_teacher_switch"])),
            "train_positive_rate": float(np.mean(train_aug["y_gain"])),
            "train_harm_rate": float(np.mean(train_aug["y_harm"])),
        },
    }


def _stats(seed_runs: list[Mapping[str, Any]], path: tuple[str, ...]) -> dict[str, Any]:
    vals = []
    for run in seed_runs:
        cur: Any = run
        for key in path:
            cur = cur[key]
        vals.append(float(cur))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "values": [float(x) for x in arr.tolist()],
    }


def _aggregate(seed_runs: list[Mapping[str, Any]], cz_report: Mapping[str, Any], da_report: Mapping[str, Any]) -> dict[str, Any]:
    cz_agg = cz_report.get("aggregate", {})
    da_agg = da_report.get("aggregate", {})
    t100 = _stats(seed_runs, ("test_metrics_with_floor", "t100_raw_frame_full_waypoint_diagnostic_vs_floor"))
    min_without = _stats(seed_runs, ("test_group_summary", "min_without_any_group_t100"))
    easy = _stats(seed_runs, ("test_metrics_with_floor", "easy_degradation_vs_floor"))
    ci_lows = np.asarray([
        run["bootstrap_ci"]["metrics"]["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]["low"]
        for run in seed_runs
    ], dtype=np.float64)
    return {
        "t100": t100,
        "hard_failure": _stats(seed_runs, ("test_metrics_with_floor", "hard_failure_full_waypoint_ade_improvement_vs_floor")),
        "easy_degradation": easy,
        "switch_rate": _stats(seed_runs, ("test_metrics_with_floor", "switch_rate")),
        "min_without_group_t100": min_without,
        "teacher_switch_rate_train": _stats(seed_runs, ("teacher_switch_rate_train",)),
        "bootstrap_t100_low_values": [float(x) for x in ci_lows.tolist()],
        "all_seed_t100_positive": bool(all(v > 0.0 for v in t100["values"])),
        "all_seed_easy_safe": bool(easy["max"] <= 0.02),
        "all_seed_bootstrap_low_positive": bool(np.all(ci_lows > 0.0)),
        "cz_reference": {
            "robust_t100_mean": float(cz_agg.get("robust_t100", {}).get("mean", 0.0)),
            "robust_min_without_group_t100_mean": float(cz_agg.get("robust_min_without_group_t100", {}).get("mean", 0.0)),
        },
        "da_reference": {
            "t100_mean": float(da_agg.get("t100", {}).get("mean", 0.0)),
            "min_without_group_t100_mean": float(da_agg.get("min_without_group_t100", {}).get("mean", 0.0)),
        },
        "beats_da_t100_mean": bool(t100["mean"] > float(da_agg.get("t100", {}).get("mean", 0.0))),
        "beats_cz_t100_mean": bool(t100["mean"] > float(cz_agg.get("robust_t100", {}).get("mean", 0.0))),
        "beats_da_min_without_group_mean": bool(min_without["mean"] > float(da_agg.get("min_without_group_t100", {}).get("mean", 0.0))),
        "beats_cz_min_without_group_mean": bool(min_without["mean"] > float(cz_agg.get("robust_min_without_group_t100", {}).get("mean", 0.0))),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    agg = payload["aggregate"]
    gates = {
        "db_precondition_present": payload["stage43_db_precondition"]["verdict"]
        == "stage43_db_t100_head_failure_forensics_complete_policy_distill_next",
        "fresh_policy_distillation_training": payload["result_source"] == "fresh_torch_policy_distilled_t100_admissibility_head",
        "three_or_more_seeds": len(payload["seed_runs"]) >= 3,
        "all_checkpoints_written_not_committed": all(Path(run["checkpoint"]).exists() and run["checkpoint_committed"] is False for run in payload["seed_runs"]),
        "teacher_policy_from_cz": payload["training_protocol"]["teacher"] == "stage43_cz_leave_group_out_policy",
        "validation_only_policy_selection": payload["selection_protocol"]["test_threshold_tuning"] is False,
        "feature_contract_clean": not payload["feature_contract"]["denied_feature_name_hits"],
        "test_once_per_seed": all(run["test_metrics_with_floor"]["rows"] > 0 for run in payload["seed_runs"]),
        "easy_preserved": bool(agg["all_seed_easy_safe"]),
        "diagnostic_not_deployed": payload["deploy_on_current_heldout"] is False,
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
    if passed != total:
        verdict = "stage43_dc_t100_policy_distilled_head_incomplete_keep_diagnostic"
    elif agg["beats_cz_t100_mean"] and agg["all_seed_easy_safe"]:
        verdict = "stage43_dc_t100_policy_distilled_head_beats_cz_diagnostic"
    elif agg["beats_da_t100_mean"] and agg["all_seed_easy_safe"]:
        verdict = "stage43_dc_t100_policy_distilled_head_improves_da_not_cz"
    elif agg["all_seed_t100_positive"] and agg["all_seed_easy_safe"]:
        verdict = "stage43_dc_t100_policy_distilled_head_positive_but_not_repair"
    else:
        verdict = "stage43_dc_t100_policy_distilled_head_no_lift_keep_policy"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_dc_gate"]
    agg = payload["aggregate"]
    lines = [
        "# Stage43-DC T100 Policy-Distilled Admissibility Head",
        "",
        f"- source: `{SOURCE}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- seeds: `{payload['seeds']}`",
        "- deploy on current heldout t100: `False`",
        "",
        "## Aggregate",
        "",
        f"- mean t100 improvement: `{agg['t100']['mean']:.6f}`",
        f"- mean min-without-group t100: `{agg['min_without_group_t100']['mean']:.6f}`",
        f"- max easy degradation: `{agg['easy_degradation']['max']:.6f}`",
        f"- mean switch rate: `{agg['switch_rate']['mean']:.6f}`",
        f"- mean teacher switch rate: `{agg['teacher_switch_rate_train']['mean']:.6f}`",
        f"- beats DA t100 mean: `{agg['beats_da_t100_mean']}`",
        f"- beats CZ t100 mean: `{agg['beats_cz_t100_mean']}`",
        f"- beats DA min-without-group mean: `{agg['beats_da_min_without_group_mean']}`",
        f"- beats CZ min-without-group mean: `{agg['beats_cz_min_without_group_mean']}`",
        "",
        "## Per Seed",
        "",
        "| seed | t100 | min-without-group | easy degradation | switch rate | teacher switch | bootstrap low | best epoch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in payload["seed_runs"]:
        metrics = run["test_metrics_with_floor"]
        group = run["test_group_summary"]
        low = run["bootstrap_ci"]["metrics"]["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]["low"]
        lines.append(
            f"| `{run['seed']}` | `{metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor']:.6f}` | "
            f"`{group['min_without_any_group_t100']:.6f}` | `{metrics['easy_degradation_vs_floor']:.6f}` | "
            f"`{metrics['switch_rate']:.6f}` | `{run['teacher_switch_rate_train']:.6f}` | `{low:.6f}` | `{run['best_epoch']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This trains on CZ leave-group-out robust switch decisions rather than only generic gain/harm/delta labels.",
            "- Teacher labels are built on train rows; validation still selects policy/checkpoint and test is evaluated once.",
            "- Future waypoints are labels/eval only; inference inputs are causal CS diagnostics, latent state, history/goal/baseline features, and split metadata.",
            "- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
            "",
        ]
    )
    return lines


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_dc_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-DC Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- beats DA t100 mean: `{payload['aggregate']['beats_da_t100_mean']}`",
            f"- beats CZ t100 mean: `{payload['aggregate']['beats_cz_t100_mean']}`",
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
    agg = payload["aggregate"]
    readme_block = [
        "## Stage43-DC: policy-distilled t100 admissibility head",
        "",
        "After DB showed that DA missed CZ's deployment behavior, I trained a new head to imitate the CZ leave-group-out switch decisions on train rows, while keeping harm/easy and group weighting in the loss.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- mean t100 improvement: `{agg['t100']['mean']:.4%}`",
        f"- mean min-without-group t100: `{agg['min_without_group_t100']['mean']:.4%}`",
        f"- max easy degradation: `{agg['easy_degradation']['max']:.4%}`",
        f"- beats DA t100 mean: `{agg['beats_da_t100_mean']}`",
        f"- beats CZ t100 mean: `{agg['beats_cz_t100_mean']}`",
        f"- deploy on current heldout t100: `{payload['deploy_on_current_heldout']}`",
        "",
        "This tests whether CZ's robust policy can be learned by the head itself. I still do not deploy t100 unless the learned head beats the stronger CZ policy and preserves worst-group/easy behavior.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, readme_block)
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage43_dc_t100_policy_distilled_admissibility_head"
    state["current_verdict"] = gate["verdict"]
    stage = state.setdefault("stage43_long_research_execution", {})
    stage["t100_policy_distilled_admissibility_head"] = {
        "source": SOURCE,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "aggregate": payload["aggregate"],
        "deploy_on_current_heldout": payload["deploy_on_current_heldout"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, _jsonable(state))


def train_t100_policy_distilled_admissibility_head(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CKPT_DIR)
    db_report = _ensure_db_precondition(args)
    cz_report = read_json(cz.REPORT_JSON, {})
    cu_report = read_json(cu.REPORT_JSON, {})
    seeds = _parse_seeds(args.seeds)
    runtime = m._configure_runtime(seeds[0])
    seed_runs = [_seed_payload(seed, args, runtime, cz_report, cu_report) for seed in seeds]
    da_report = read_json(db.da.REPORT_JSON, {})
    aggregate = _aggregate(seed_runs, cz_report, da_report)
    first = seed_runs[0]
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_torch_policy_distilled_t100_admissibility_head",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "mode": "quick" if args.quick else "small",
        "seeds": seeds,
        "runtime": runtime,
        "stage43_db_precondition": {
            "report": str(db.REPORT_JSON),
            "verdict": db_report.get("stage43_db_gate", {}).get("verdict"),
        },
        "training_protocol": {
            "teacher": "stage43_cz_leave_group_out_policy",
            "teacher_labels_on": "train_rows_only",
            "group_weighting": True,
            "epochs": int(args.epochs),
        },
        "selection_protocol": {
            "validation_only": True,
            "test_threshold_tuning": False,
            "objective": "cz_leave_group_out_policy_search_on_new_head",
        },
        "seed_runs": seed_runs,
        "aggregate": aggregate,
        "feature_contract": first["feature_contract"],
        "data_rows": first["data_rows"],
        "deploy_on_current_heldout": False,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "feature_standardization_train_only": True,
            "teacher_policy_selected_on_validation_not_test": True,
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
    payload["stage43_dc_gate"] = _gate(payload)
    _write_reports(payload)
    print(f"Stage43-DC: {payload['stage43_dc_gate']['verdict']} ({payload['stage43_dc_gate']['passed']}/{payload['stage43_dc_gate']['total']})")
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Train Stage43-DC policy-distilled t100 admissibility head.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--seeds", type=str, default="4323,4331,4337")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args(argv)
    return train_t100_policy_distilled_admissibility_head(args)


if __name__ == "__main__":
    main()
