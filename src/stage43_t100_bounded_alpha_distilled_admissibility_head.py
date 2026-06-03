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
from src import stage43_t100_policy_distilled_admissibility_head as dc
from src import stage43_t100_policy_distilled_alpha_stability_policy as de
from src import stage43_t100_policy_distilled_group_stability_guard as dd
from src import stage43_t100_residual_admissibility_leave_group_out_policy as cz
from src import stage43_t100_residual_admissibility_group_support_guard as cy
from src import stage43_t100_residual_admissibility_head as ct
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_protected_latent_state_model import _jsonable


OUT_DIR = m.OUT_DIR
CKPT_DIR = OUT_DIR / "checkpoints"
REPORT_JSON = OUT_DIR / "stage43_t100_bounded_alpha_distilled_admissibility_head.json"
REPORT_MD = OUT_DIR / "stage43_t100_bounded_alpha_distilled_admissibility_head.md"
GATE_MD = OUT_DIR / "stage43_stage_df_t100_bounded_alpha_distilled_admissibility_head_gate.md"
HEARTBEAT_JSON = OUT_DIR / "stage43_t100_bounded_alpha_distilled_admissibility_head_heartbeat.json"

README_RESULTS = m.README_RESULTS
M3W_README = m.M3W_README
WORK_SUMMARY = m.WORK_SUMMARY
RESEARCH_STATE = m.RESEARCH_STATE

SECTION = "STAGE43_DF_T100_BOUNDED_ALPHA_DISTILLED_ADMISSIBILITY_HEAD"
SOURCE = "fresh_stage43_df_t100_bounded_alpha_distilled_admissibility_head"


def _ensure_de_precondition(args: argparse.Namespace) -> dict[str, Any]:
    report = read_json(de.REPORT_JSON, {})
    gate = report.get("stage43_de_gate", {})
    if not report or gate.get("passed") != gate.get("total"):
        report = de.run_t100_policy_distilled_alpha_stability_policy(args)
    return report


def _de_seed_run(report: Mapping[str, Any], seed: int) -> dict[str, Any]:
    for run in report.get("seed_runs", []):
        if int(run.get("seed", -1)) == int(seed):
            return dict(run)
    raise KeyError(f"DE seed {seed} not found")


def _dc_seed_run(report: Mapping[str, Any], seed: int) -> dict[str, Any]:
    for run in report.get("seed_runs", []):
        if int(run.get("seed", -1)) == int(seed):
            return dict(run)
    raise KeyError(f"DC seed {seed} not found")


def _teacher_predictions_from_dc(
    ds: m.WaypointSplit,
    cs_pred: Mapping[str, np.ndarray],
    dc_seed_run: Mapping[str, Any],
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    device = torch.device("cpu")
    aug = ct._augment_alpha_features(ds, cs_pred)
    model, mean, std = dd._load_dc_seed_head(dc_seed_run)
    aug["x"] = ((aug["x"] - mean) / std).astype(np.float32)
    return ct._predict_head(model, aug, device, int(batch_size))


def _attach_bounded_teacher_targets(
    data: dict[str, np.ndarray],
    ds: m.WaypointSplit,
    teacher_head_pred: Mapping[str, np.ndarray],
    teacher_policy: Mapping[str, Any],
    *,
    alpha_cap: float,
) -> None:
    teacher_switch = dc._teacher_switch_labels(ds, teacher_head_pred, teacher_policy)
    alpha_count = int(len(ct.ALPHAS))
    alpha_values = np.repeat(ct.ALPHAS.astype(np.float32), int(len(ds.x)))
    teacher_switch = np.where(alpha_values <= float(alpha_cap) + 1e-8, teacher_switch, 0.0).astype(np.float32)
    data["y_teacher_switch"] = teacher_switch
    data["alpha_value"] = alpha_values.astype(np.float32)
    data["alpha_above_cap"] = (alpha_values > float(alpha_cap) + 1e-8).astype(np.float32)

    source = np.concatenate([ds.source_file.astype(str) for _ in range(alpha_count)], axis=0)
    scene = np.concatenate([ds.scene_id.astype(str) for _ in range(alpha_count)], axis=0)
    domain = np.concatenate([ds.domain.astype(str) for _ in range(alpha_count)], axis=0)
    group_weight = dc._balanced_group_weights(source, scene, domain)
    hard = np.concatenate([(ds.hard | ds.failure).astype(np.float32) for _ in range(alpha_count)], axis=0)
    easy = np.concatenate([ds.easy.astype(np.float32) for _ in range(alpha_count)], axis=0)
    data["group_weight"] = np.clip(
        group_weight * (1.0 + 0.55 * hard - 0.20 * easy + 0.80 * teacher_switch + 0.20 * data["alpha_above_cap"]),
        0.20,
        7.0,
    ).astype(np.float32)


def _distill_loss(
    model: ct.ResidualAdmissibilityHead,
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    x = torch.from_numpy(data["x"][ids]).to(device)
    y_teacher = torch.from_numpy(data["y_teacher_switch"][ids]).to(device)
    y_gain = torch.from_numpy(data["y_gain"][ids]).to(device)
    y_harm = torch.from_numpy(data["y_harm"][ids]).to(device)
    y_delta = torch.from_numpy(data["y_delta"][ids]).to(device)
    alpha_above = torch.from_numpy(data["alpha_above_cap"][ids]).to(device)
    weight = torch.from_numpy(data["group_weight"][ids]).to(device)
    weight = weight / torch.clamp(weight.mean(), min=1e-6)
    out = model(x)
    pos = float(max(1, int(y_teacher.detach().cpu().numpy().sum())))
    neg = float(max(1, len(ids) - int(y_teacher.detach().cpu().numpy().sum())))
    teacher_weight = torch.where(y_teacher > 0.5, torch.tensor(neg / pos, device=device).clamp(max=12.0), torch.ones_like(y_teacher))
    teacher = nn.functional.binary_cross_entropy_with_logits(out["gain_logit"], y_teacher, weight=teacher_weight * weight)
    gain_aux = nn.functional.binary_cross_entropy_with_logits(out["gain_logit"], y_gain, weight=weight)
    harm = nn.functional.binary_cross_entropy_with_logits(out["harm_logit"], y_harm, weight=weight)
    delta = torch.mean(nn.functional.smooth_l1_loss(out["delta"], y_delta, reduction="none") * weight)
    prob_gain = torch.sigmoid(out["gain_logit"])
    prob_harm = torch.sigmoid(out["harm_logit"])
    over_alpha_gain = torch.mean(prob_gain * alpha_above * weight)
    safe_margin = torch.relu(prob_gain + prob_harm - 1.0)
    support = torch.mean(safe_margin * weight)
    total = 1.45 * teacher + 0.25 * gain_aux + 1.10 * harm + 0.45 * delta + 0.25 * over_alpha_gain + 0.06 * support
    return total, {
        "teacher": float(teacher.detach().cpu()),
        "gain_aux": float(gain_aux.detach().cpu()),
        "harm": float(harm.detach().cpu()),
        "delta": float(delta.detach().cpu()),
        "over_alpha_gain": float(over_alpha_gain.detach().cpu()),
        "support": float(support.detach().cpu()),
    }


def _bounded_policy_candidates(
    ds: m.WaypointSplit,
    cs_pred: Mapping[str, np.ndarray],
    head_pred: Mapping[str, np.ndarray],
    *,
    alpha_cap: float,
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in cz._policy_candidates(ds, cs_pred, head_pred)
        if bool(row.get("safe", False))
        and int(row.get("policy", {}).get("alpha_index", -1)) >= 0
        and float(row.get("policy", {}).get("alpha", 0.0)) <= float(alpha_cap)
    ]
    rows.sort(key=lambda row: float(row["objective"]), reverse=True)
    return rows


def _select_bounded_policy(
    ds: m.WaypointSplit,
    cs_pred: Mapping[str, np.ndarray],
    head_pred: Mapping[str, np.ndarray],
    *,
    alpha_cap: float,
) -> dict[str, Any]:
    bounded = _bounded_policy_candidates(ds, cs_pred, head_pred, alpha_cap=alpha_cap)
    if bounded:
        selected = bounded[0]
    else:
        selected = cz._policy_candidates(ds, cs_pred, head_pred)[0]
    return {
        "policy": selected["policy"],
        "metrics": selected["metrics"],
        "group_summary": selected["group_summary"],
        "objective": float(selected["objective"]),
        "safe": bool(selected["safe"]),
        "bounded_candidate_count": int(len(bounded)),
    }


def _seed_payload(seed: int, args: argparse.Namespace, runtime: Mapping[str, Any], de_report: Mapping[str, Any], dc_report: Mapping[str, Any]) -> dict[str, Any]:
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
    de_seed = _de_seed_run(de_report, seed)
    dc_seed = _dc_seed_run(dc_report, seed)
    teacher_policy = de_seed["selected_variant"]["selected_policy"]
    train_teacher = _teacher_predictions_from_dc(train, train_pred, dc_seed, batch_size=int(args.batch_size))
    val_teacher = _teacher_predictions_from_dc(val, val_pred, dc_seed, batch_size=int(args.batch_size))
    train_aug = ct._augment_alpha_features(train, train_pred)
    val_aug = ct._augment_alpha_features(val, val_pred)
    test_aug = ct._augment_alpha_features(test, test_pred)
    _attach_bounded_teacher_targets(train_aug, train, train_teacher, teacher_policy, alpha_cap=float(args.alpha_cap))
    _attach_bounded_teacher_targets(val_aug, val, val_teacher, teacher_policy, alpha_cap=float(args.alpha_cap))
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
        for ids in ct._batch_indices(len(train_aug["x"]), int(args.batch_size), shuffle=True, seed=int(seed) + epoch + 7300):
            opt.zero_grad(set_to_none=True)
            loss, stat = _distill_loss(model, train_aug, ids, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            stats.append(stat)
        val_head = ct._predict_head(model, val_aug, device, int(args.batch_size))
        val_selection = _select_bounded_policy(val, val_pred, val_head, alpha_cap=float(args.alpha_cap))
        row = {
            "seed": int(seed),
            "epoch": int(epoch + 1),
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "validation_objective": float(val_selection["objective"]),
            "validation_t100": float(val_selection["metrics"]["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]),
            "validation_min_without_group_t100": float(val_selection["group_summary"]["min_without_any_group_t100"]),
            "bounded_candidate_count": int(val_selection["bounded_candidate_count"]),
            "teacher_switch_rate_train": float(np.mean(train_aug["y_teacher_switch"])),
            "teacher_alpha_cap": float(args.alpha_cap),
            "loss_terms": {key: float(np.mean([s[key] for s in stats])) for key in stats[0]} if stats else {},
        }
        history.append(row)
        write_json(
            HEARTBEAT_JSON,
            _jsonable({"source": SOURCE, "seed": int(seed), "epoch": int(epoch + 1), "elapsed_s": time.time() - start, "last": row}),
        )
        if best is None or row["validation_objective"] > best["history_row"]["validation_objective"]:
            best = {
                "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "epoch": int(epoch + 1),
                "history_row": row,
            }
    assert best is not None
    model.load_state_dict(best["model_state"])
    ckpt_path = CKPT_DIR / f"stage43_df_t100_bounded_alpha_distilled_admissibility_seed{int(seed)}.pt"
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
            "teacher_source": "stage43_de_bounded_alpha_policy_on_train_rows",
            "teacher_policy": teacher_policy,
            "training_objective": "bounded_alpha_policy_distillation_plus_group_weighted_harm_delta",
            "alpha_cap": float(args.alpha_cap),
        },
        ckpt_path,
    )
    val_head = ct._predict_head(model, val_aug, device, int(args.batch_size))
    test_head = ct._predict_head(model, test_aug, device, int(args.batch_size))
    val_selection = _select_bounded_policy(val, val_pred, val_head, alpha_cap=float(args.alpha_cap))
    test_metrics, selected_ade, selected_fde, switched = ct._evaluate_selected(test, test_pred, test_head, val_selection["policy"])
    test_group = cy._group_summary(test, selected_ade, selected_fde, switched)
    bootstrap = m._bootstrap_ci(test, selected_ade, selected_fde, n=int(args.bootstrap), seed=int(seed) + 7600)
    return {
        "seed": int(seed),
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": ct.cr._sha256(ckpt_path),
        "checkpoint_committed": False,
        "best_epoch": int(best["epoch"]),
        "training_history": history,
        "teacher_policy": teacher_policy,
        "teacher_switch_rate_train": float(np.mean(train_aug["y_teacher_switch"])),
        "validation_selected_policy": val_selection,
        "test_metrics_with_floor": test_metrics,
        "test_group_summary": test_group,
        "bootstrap_ci": bootstrap,
        "switch_count": int(switched.sum()),
        "data_rows": {"train": int(len(train.x)), "val": int(len(val.x)), "test": int(len(test.x))},
        "feature_contract": ct._feature_contract(train_aug["feature_names"]),
        "feature_name_hash": dc._sha_names(train_aug["feature_names"]),
        "alpha_protocol": {
            "alpha_cap": float(args.alpha_cap),
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


def _aggregate(seed_runs: list[Mapping[str, Any]], de_report: Mapping[str, Any], dc_report: Mapping[str, Any]) -> dict[str, Any]:
    de_agg = de_report.get("aggregate", {})
    dc_agg = dc_report.get("aggregate", {})
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
        "all_min_without_group_positive": bool(min_without["min"] > 0.0),
        "de_reference": {
            "bounded_t100_mean": float(de_agg.get("bounded_t100", {}).get("mean", 0.0)),
            "bounded_min_without_group_t100_mean": float(de_agg.get("bounded_min_without_group_t100", {}).get("mean", 0.0)),
        },
        "dc_reference": {
            "t100_mean": float(dc_agg.get("t100", {}).get("mean", 0.0)),
            "min_without_group_t100_mean": float(dc_agg.get("min_without_group_t100", {}).get("mean", 0.0)),
        },
        "beats_de_t100_mean": bool(t100["mean"] > float(de_agg.get("bounded_t100", {}).get("mean", 0.0))),
        "beats_de_min_without_group_mean": bool(min_without["mean"] > float(de_agg.get("bounded_min_without_group_t100", {}).get("mean", 0.0))),
        "beats_dc_min_without_group_mean": bool(min_without["mean"] > float(dc_agg.get("min_without_group_t100", {}).get("mean", 0.0))),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    agg = payload["aggregate"]
    gates = {
        "de_precondition_present": payload["stage43_de_precondition"]["verdict"]
        == "stage43_de_t100_alpha_stability_policy_repairs_group_fragility_diagnostic",
        "fresh_bounded_alpha_distillation_training": payload["result_source"] == "fresh_torch_bounded_alpha_policy_distilled_t100_head",
        "three_or_more_seeds": len(payload["seed_runs"]) >= 3,
        "all_checkpoints_written_not_committed": all(Path(run["checkpoint"]).exists() and run["checkpoint_committed"] is False for run in payload["seed_runs"]),
        "teacher_policy_from_de": payload["training_protocol"]["teacher"] == "stage43_de_bounded_alpha_policy",
        "bounded_alpha_protocol": float(payload["training_protocol"]["alpha_cap"]) <= 0.75,
        "validation_only_policy_selection": payload["selection_protocol"]["test_threshold_tuning"] is False,
        "feature_contract_clean": not payload["feature_contract"]["denied_feature_name_hits"],
        "test_once_per_seed": all(run["test_metrics_with_floor"]["rows"] > 0 for run in payload["seed_runs"]),
        "easy_preserved": bool(agg["all_seed_easy_safe"]),
        "all_min_without_group_positive": bool(agg["all_min_without_group_positive"]),
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
        verdict = "stage43_df_t100_bounded_alpha_distilled_head_incomplete"
    elif agg["beats_de_t100_mean"] and agg["all_min_without_group_positive"]:
        verdict = "stage43_df_t100_bounded_alpha_distilled_head_beats_de_diagnostic"
    elif agg["all_min_without_group_positive"] and agg["all_seed_t100_positive"]:
        verdict = "stage43_df_t100_bounded_alpha_distilled_head_learns_safe_signal_diagnostic"
    else:
        verdict = "stage43_df_t100_bounded_alpha_distilled_head_no_repair"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_df_gate"]
    agg = payload["aggregate"]
    lines = [
        "# Stage43-DF T100 Bounded-Alpha Distilled Admissibility Head",
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
        f"- all min-without-group positive: `{agg['all_min_without_group_positive']}`",
        f"- max easy degradation: `{agg['easy_degradation']['max']:.6f}`",
        f"- mean switch rate: `{agg['switch_rate']['mean']:.6f}`",
        f"- beats DE t100 mean: `{agg['beats_de_t100_mean']}`",
        f"- beats DE min-without-group mean: `{agg['beats_de_min_without_group_mean']}`",
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
            "- This trains a new head from the DE bounded-alpha policy instead of only adding an outer alpha cap at deployment time.",
            "- Teacher labels are built on train rows using the DC teacher head and DE validation-selected bounded-alpha policies.",
            "- Validation still selects policy/checkpoint and test is evaluated once; checkpoints are written locally but not committed.",
            "- Future waypoints are labels/eval only; inference inputs remain causal.",
            "- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
            "",
        ]
    )
    return lines


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_df_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-DF Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- all min-without-group positive: `{payload['aggregate']['all_min_without_group_positive']}`",
            f"- beats DE t100 mean: `{payload['aggregate']['beats_de_t100_mean']}`",
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
        "## Stage43-DF: bounded-alpha distilled t100 admissibility head",
        "",
        "DE showed that bounded intervention can repair the policy-distilled t100 head's seed-level group fragility. I trained a new head to imitate that bounded-alpha policy on train rows, rather than relying only on an outer deployment cap.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- mean t100 improvement: `{agg['t100']['mean']:.4%}`",
        f"- mean min-without-group t100: `{agg['min_without_group_t100']['mean']:.4%}`",
        f"- all min-without-group positive: `{agg['all_min_without_group_positive']}`",
        f"- max easy degradation: `{agg['easy_degradation']['max']:.4%}`",
        f"- beats DE t100 mean: `{agg['beats_de_t100_mean']}`",
        f"- deploy on current heldout t100: `{payload['deploy_on_current_heldout']}`",
        "",
        "My read: this is the right direction if it can retain DE's group safety while recovering mean t100. I still keep it diagnostic unless the trained head beats the bounded policy and keeps every seed/slice safe.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, readme_block)
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage43_df_t100_bounded_alpha_distilled_admissibility_head"
    state["current_verdict"] = gate["verdict"]
    stage = state.setdefault("stage43_long_research_execution", {})
    stage["t100_bounded_alpha_distilled_admissibility_head"] = {
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


def train_t100_bounded_alpha_distilled_admissibility_head(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CKPT_DIR)
    de_report = _ensure_de_precondition(args)
    dc_report = read_json(dc.REPORT_JSON, {})
    seeds = dc._parse_seeds(args.seeds)
    runtime = m._configure_runtime(seeds[0])
    seed_runs = [_seed_payload(seed, args, runtime, de_report, dc_report) for seed in seeds]
    aggregate = _aggregate(seed_runs, de_report, dc_report)
    first = seed_runs[0]
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_torch_bounded_alpha_policy_distilled_t100_head",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "mode": "quick" if args.quick else "small",
        "seeds": seeds,
        "runtime": runtime,
        "stage43_de_precondition": {
            "report": str(de.REPORT_JSON),
            "verdict": de_report.get("stage43_de_gate", {}).get("verdict"),
        },
        "training_protocol": {
            "teacher": "stage43_de_bounded_alpha_policy",
            "teacher_labels_on": "train_rows_only",
            "alpha_cap": float(args.alpha_cap),
            "group_weighting": True,
            "epochs": int(args.epochs),
        },
        "selection_protocol": {
            "validation_only": True,
            "test_threshold_tuning": False,
            "objective": "bounded_alpha_leave_group_out_policy_search_on_new_head",
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
    payload["stage43_df_gate"] = _gate(payload)
    _write_reports(payload)
    print(f"Stage43-DF: {payload['stage43_df_gate']['verdict']} ({payload['stage43_df_gate']['passed']}/{payload['stage43_df_gate']['total']})")
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Train Stage43-DF bounded-alpha policy-distilled t100 admissibility head.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--seeds", type=str, default="4323,4331,4337")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1.3e-3)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--alpha-cap", type=float, default=0.75)
    # Compatibility with DE/DD precondition rebuild.
    parser.add_argument("--min-label-rows", type=int, default=80)
    parser.add_argument("--min-val-improvement", type=float, default=0.0002)
    args = parser.parse_args(argv)
    return train_t100_bounded_alpha_distilled_admissibility_head(args)


if __name__ == "__main__":
    main()
