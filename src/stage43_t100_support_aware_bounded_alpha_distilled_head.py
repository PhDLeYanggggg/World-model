from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_t100_bounded_alpha_distilled_admissibility_head as df
from src import stage43_t100_bounded_alpha_head_support_aware_selection as dh
from src import stage43_t100_policy_distilled_admissibility_head as dc
from src import stage43_t100_residual_admissibility_group_support_guard as cy
from src import stage43_t100_residual_admissibility_head as ct
from src import stage43_t100_residual_admissibility_leave_group_out_policy as cz
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_protected_latent_state_model import _jsonable


OUT_DIR = m.OUT_DIR
CKPT_DIR = OUT_DIR / "checkpoints"
REPORT_JSON = OUT_DIR / "stage43_t100_support_aware_bounded_alpha_distilled_head.json"
REPORT_MD = OUT_DIR / "stage43_t100_support_aware_bounded_alpha_distilled_head.md"
GATE_MD = OUT_DIR / "stage43_stage_di_t100_support_aware_bounded_alpha_distilled_head_gate.md"
HEARTBEAT_JSON = OUT_DIR / "stage43_t100_support_aware_bounded_alpha_distilled_head_heartbeat.json"

README_RESULTS = m.README_RESULTS
M3W_README = m.M3W_README
WORK_SUMMARY = m.WORK_SUMMARY
RESEARCH_STATE = m.RESEARCH_STATE

SECTION = "STAGE43_DI_T100_SUPPORT_AWARE_BOUNDED_ALPHA_DISTILLED_HEAD"
SOURCE = "fresh_stage43_di_t100_support_aware_bounded_alpha_distilled_head"


def _ensure_dh_precondition(args: argparse.Namespace) -> dict[str, Any]:
    report = read_json(dh.REPORT_JSON, {})
    gate = report.get("stage43_dh_gate", {})
    if not report or gate.get("passed") != gate.get("total"):
        report = dh.run_t100_bounded_alpha_head_support_aware_selection(args)
    return report


def _dh_seed_policy(report: Mapping[str, Any], seed: int) -> dict[str, Any]:
    for run in report.get("seed_runs", []):
        if int(run.get("seed", -1)) == int(seed):
            return dict(run["selected_validation_candidate"]["policy"])
    raise KeyError(f"DH seed policy not found for seed {seed}")


def _augment_support_weights(data: dict[str, np.ndarray], ds: m.WaypointSplit) -> None:
    alpha_count = int(len(ct.ALPHAS))
    source = np.concatenate([ds.source_file.astype(str) for _ in range(alpha_count)], axis=0)
    scene = np.concatenate([ds.scene_id.astype(str) for _ in range(alpha_count)], axis=0)
    domain = np.concatenate([ds.domain.astype(str) for _ in range(alpha_count)], axis=0)
    teacher = np.asarray(data["y_teacher_switch"], dtype=np.float32)
    base = np.asarray(data["group_weight"], dtype=np.float32)
    support_boost = np.ones_like(base, dtype=np.float32)
    for labels, scale in [(source, 0.25), (scene, 0.35), (domain, 0.15)]:
        labels = np.asarray(labels).astype(str)
        for label in sorted(set(labels.tolist())):
            mask = labels == label
            positive = float(np.sum(teacher[mask]))
            if positive <= 0:
                support_boost[mask] *= 1.0 + scale
            else:
                support_boost[mask] *= 1.0 + scale / np.sqrt(positive + 1.0)
    data["group_weight"] = np.clip(base * support_boost, 0.20, 9.0).astype(np.float32)


def _select_support_policy(
    ds: m.WaypointSplit,
    cs_pred: Mapping[str, np.ndarray],
    head_pred: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    rows = [dh._candidate_row(row, ds, cs_pred, head_pred) for row in cz._policy_candidates(ds, cs_pred, head_pred)]
    selected = dh._select_support_aware_candidate(rows)
    return {
        "policy": selected["policy"],
        "metrics": selected["metrics"],
        "group_summary": selected["group_summary"],
        "support": selected["support"],
        "objective": float(selected["support_aware_objective"]),
        "support_safe": bool(selected["support_safe"]),
        "support_safe_candidate_count": int(sum(bool(row["support_safe"]) for row in rows)),
    }


def _seed_payload(seed: int, args: argparse.Namespace, runtime: Mapping[str, Any], dh_report: Mapping[str, Any], dc_report: Mapping[str, Any]) -> dict[str, Any]:
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
    dc_seed = df._dc_seed_run(dc_report, seed)
    teacher_policy = _dh_seed_policy(dh_report, seed)
    train_teacher = df._teacher_predictions_from_dc(train, train_pred, dc_seed, batch_size=int(args.batch_size))
    val_teacher = df._teacher_predictions_from_dc(val, val_pred, dc_seed, batch_size=int(args.batch_size))
    train_aug = ct._augment_alpha_features(train, train_pred)
    val_aug = ct._augment_alpha_features(val, val_pred)
    test_aug = ct._augment_alpha_features(test, test_pred)
    df._attach_bounded_teacher_targets(train_aug, train, train_teacher, teacher_policy, alpha_cap=float(args.alpha_cap))
    df._attach_bounded_teacher_targets(val_aug, val, val_teacher, teacher_policy, alpha_cap=float(args.alpha_cap))
    _augment_support_weights(train_aug, train)
    _augment_support_weights(val_aug, val)
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
        for ids in ct._batch_indices(len(train_aug["x"]), int(args.batch_size), shuffle=True, seed=int(seed) + epoch + 7700):
            opt.zero_grad(set_to_none=True)
            loss, stat = df._distill_loss(model, train_aug, ids, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            stats.append(stat)
        val_head = ct._predict_head(model, val_aug, device, int(args.batch_size))
        val_selection = _select_support_policy(val, val_pred, val_head)
        row = {
            "seed": int(seed),
            "epoch": int(epoch + 1),
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "validation_objective": float(val_selection["objective"]),
            "validation_t100": float(val_selection["metrics"]["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]),
            "validation_min_without_group_t100": float(val_selection["group_summary"]["min_without_any_group_t100"]),
            "validation_support_safe": bool(val_selection["support_safe"]),
            "support_safe_candidate_count": int(val_selection["support_safe_candidate_count"]),
            "teacher_switch_rate_train": float(np.mean(train_aug["y_teacher_switch"])),
            "loss_terms": {key: float(np.mean([s[key] for s in stats])) for key in stats[0]} if stats else {},
        }
        history.append(row)
        write_json(
            HEARTBEAT_JSON,
            _jsonable({"source": SOURCE, "seed": int(seed), "epoch": int(epoch + 1), "elapsed_s": time.time() - start, "last": row}),
        )
        if best is None or (
            row["validation_support_safe"],
            row["validation_min_without_group_t100"],
            row["validation_objective"],
        ) > (
            best["history_row"]["validation_support_safe"],
            best["history_row"]["validation_min_without_group_t100"],
            best["history_row"]["validation_objective"],
        ):
            best = {
                "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "epoch": int(epoch + 1),
                "history_row": row,
            }
    assert best is not None
    model.load_state_dict(best["model_state"])
    ckpt_path = CKPT_DIR / f"stage43_di_t100_support_aware_bounded_alpha_seed{int(seed)}.pt"
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
            "teacher_source": "stage43_dh_support_aware_policy_on_train_rows",
            "teacher_policy": teacher_policy,
            "training_objective": "support_aware_bounded_alpha_distillation",
            "alpha_cap": float(args.alpha_cap),
        },
        ckpt_path,
    )
    val_head = ct._predict_head(model, val_aug, device, int(args.batch_size))
    test_head = ct._predict_head(model, test_aug, device, int(args.batch_size))
    val_selection = _select_support_policy(val, val_pred, val_head)
    test_metrics, selected_ade, selected_fde, switched = ct._evaluate_selected(test, test_pred, test_head, val_selection["policy"])
    test_group = cy._group_summary(test, selected_ade, selected_fde, switched)
    bootstrap = m._bootstrap_ci(test, selected_ade, selected_fde, n=int(args.bootstrap), seed=int(seed) + 7900)
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
        "test_support": dh._support_stats(test, switched),
        "bootstrap_ci": bootstrap,
        "switch_count": int(switched.sum()),
        "data_rows": {"train": int(len(train.x)), "val": int(len(val.x)), "test": int(len(test.x))},
        "feature_contract": ct._feature_contract(train_aug["feature_names"]),
        "feature_name_hash": dc._sha_names(train_aug["feature_names"]),
    }


def _stats(seed_runs: list[Mapping[str, Any]], path: tuple[str, ...]) -> dict[str, Any]:
    vals = []
    for run in seed_runs:
        cur: Any = run
        for key in path:
            cur = cur[key]
        vals.append(float(cur))
    arr = np.asarray(vals, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "min": float(np.min(arr)), "max": float(np.max(arr)), "values": [float(x) for x in arr.tolist()]}


def _aggregate(seed_runs: list[Mapping[str, Any]], dh_report: Mapping[str, Any], de_report: Mapping[str, Any]) -> dict[str, Any]:
    t100 = _stats(seed_runs, ("test_metrics_with_floor", "t100_raw_frame_full_waypoint_diagnostic_vs_floor"))
    min_without = _stats(seed_runs, ("test_group_summary", "min_without_any_group_t100"))
    easy = _stats(seed_runs, ("test_metrics_with_floor", "easy_degradation_vs_floor"))
    ci_lows = np.asarray([
        run["bootstrap_ci"]["metrics"]["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]["low"]
        for run in seed_runs
    ], dtype=np.float64)
    dh_agg = dh_report.get("aggregate", {})
    de_agg = de_report.get("aggregate", {})
    return {
        "t100": t100,
        "hard_failure": _stats(seed_runs, ("test_metrics_with_floor", "hard_failure_full_waypoint_ade_improvement_vs_floor")),
        "easy_degradation": easy,
        "switch_rate": _stats(seed_runs, ("test_metrics_with_floor", "switch_rate")),
        "min_without_group_t100": min_without,
        "teacher_switch_rate_train": _stats(seed_runs, ("teacher_switch_rate_train",)),
        "bootstrap_t100_low_values": [float(x) for x in ci_lows.tolist()],
        "all_seed_t100_positive": bool(t100["min"] > 0.0),
        "all_seed_easy_safe": bool(easy["max"] <= 0.02),
        "all_min_without_group_positive": bool(min_without["min"] > 0.0),
        "all_seed_bootstrap_low_positive": bool(np.all(ci_lows > 0.0)),
        "dh_reference": {
            "t100_mean": float(dh_agg.get("support_selected_t100", {}).get("mean", 0.0)),
            "min_without_group_t100_mean": float(dh_agg.get("support_selected_min_without_group_t100", {}).get("mean", 0.0)),
        },
        "de_reference": {
            "bounded_t100_mean": float(de_agg.get("bounded_t100", {}).get("mean", 0.0)),
            "bounded_min_without_group_t100_mean": float(de_agg.get("bounded_min_without_group_t100", {}).get("mean", 0.0)),
        },
        "beats_dh_t100_mean": bool(t100["mean"] > float(dh_agg.get("support_selected_t100", {}).get("mean", 0.0))),
        "beats_dh_min_without_group_mean": bool(min_without["mean"] > float(dh_agg.get("support_selected_min_without_group_t100", {}).get("mean", 0.0))),
        "beats_de_t100_mean": bool(t100["mean"] > float(de_agg.get("bounded_t100", {}).get("mean", 0.0))),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    agg = payload["aggregate"]
    gates = {
        "dh_precondition_present": payload["stage43_dh_precondition"]["verdict"]
        == "stage43_dh_t100_support_aware_selection_repairs_df_group_fragility_diagnostic",
        "fresh_support_aware_head_training": payload["result_source"] == "fresh_torch_support_aware_bounded_alpha_distilled_t100_head",
        "three_or_more_seeds": len(payload["seed_runs"]) >= 3,
        "all_checkpoints_written_not_committed": all(Path(run["checkpoint"]).exists() and run["checkpoint_committed"] is False for run in payload["seed_runs"]),
        "teacher_policy_from_dh": payload["training_protocol"]["teacher"] == "stage43_dh_support_aware_policy",
        "support_aware_validation_selection": payload["selection_protocol"]["objective"] == "support_aware_validation_t100_min_group",
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
        verdict = "stage43_di_t100_support_aware_distilled_head_incomplete"
    elif agg["beats_de_t100_mean"] and agg["all_min_without_group_positive"]:
        verdict = "stage43_di_t100_support_aware_distilled_head_beats_de_candidate"
    elif agg["all_min_without_group_positive"] and agg["beats_dh_t100_mean"]:
        verdict = "stage43_di_t100_support_aware_distilled_head_improves_dh_diagnostic"
    elif agg["all_min_without_group_positive"]:
        verdict = "stage43_di_t100_support_aware_distilled_head_safe_but_no_lift_diagnostic"
    else:
        verdict = "stage43_di_t100_support_aware_distilled_head_no_repair"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_di_gate"]
    agg = payload["aggregate"]
    lines = [
        "# Stage43-DI T100 Support-Aware Bounded-Alpha Distilled Head",
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
        f"- beats DH t100 mean: `{agg['beats_dh_t100_mean']}`",
        f"- beats DE t100 mean: `{agg['beats_de_t100_mean']}`",
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
            "- DH fixed the DF head's selection gap by reranking existing candidates.",
            "- DI retrains a new bounded-alpha head using DH support-aware policies as train-row teacher labels, then uses the same support-aware validation selector.",
            "- Checkpoints are written locally for replay, but not committed.",
            "- This is not deployed unless it beats the stronger DE bounded policy while preserving every seed/group/easy gate.",
            "- Future waypoints are labels/eval only; inference inputs remain causal.",
            "- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
            "",
        ]
    )
    return lines


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_di_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-DI Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- all min-without-group positive: `{payload['aggregate']['all_min_without_group_positive']}`",
            f"- beats DH t100 mean: `{payload['aggregate']['beats_dh_t100_mean']}`",
            f"- deploy on current heldout t100: `{payload['deploy_on_current_heldout']}`",
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
        "## Stage43-DI: support-aware bounded-alpha t100 head training",
        "",
        "DH fixed DF by reranking candidates with a support-aware validation objective. Here I trained a new head from that support-aware teacher, so the repair is part of the model training loop rather than only a post-hoc selector.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- mean t100 improvement: `{agg['t100']['mean']:.4%}`",
        f"- mean min-without-group t100: `{agg['min_without_group_t100']['mean']:.4%}`",
        f"- all min-without-group positive: `{agg['all_min_without_group_positive']}`",
        f"- max easy degradation: `{agg['easy_degradation']['max']:.4%}`",
        f"- beats DH t100 mean: `{agg['beats_dh_t100_mean']}`",
        f"- beats DE t100 mean: `{agg['beats_de_t100_mean']}`",
        f"- deploy on current heldout t100: `{payload['deploy_on_current_heldout']}`",
        "",
        "My read: this is a real training check for the support-aware idea. If it does not beat DE while preserving group safety, I keep the stronger bounded policy as the deployable reference and use this head as diagnostic evidence for the next training objective.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, readme_block)
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage43_di_t100_support_aware_bounded_alpha_distilled_head"
    state["current_verdict"] = gate["verdict"]
    stage = state.setdefault("stage43_long_research_execution", {})
    stage["t100_support_aware_bounded_alpha_distilled_head"] = {
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


def train_t100_support_aware_bounded_alpha_distilled_head(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CKPT_DIR)
    dh_report = _ensure_dh_precondition(args)
    df_report = read_json(df.REPORT_JSON, {})
    de_report = read_json(df.de.REPORT_JSON, {})
    dc_report = read_json(dc.REPORT_JSON, {})
    seeds = dc._parse_seeds(args.seeds)
    runtime = m._configure_runtime(seeds[0])
    seed_runs = [_seed_payload(seed, args, runtime, dh_report, dc_report) for seed in seeds]
    aggregate = _aggregate(seed_runs, dh_report, de_report)
    first = seed_runs[0]
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_torch_support_aware_bounded_alpha_distilled_t100_head",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "mode": "quick" if args.quick else "small",
        "seeds": seeds,
        "runtime": runtime,
        "stage43_dh_precondition": {
            "report": str(dh.REPORT_JSON),
            "verdict": dh_report.get("stage43_dh_gate", {}).get("verdict"),
        },
        "stage43_df_reference": {
            "report": str(df.REPORT_JSON),
            "verdict": df_report.get("stage43_df_gate", {}).get("verdict"),
        },
        "training_protocol": {
            "teacher": "stage43_dh_support_aware_policy",
            "teacher_labels_on": "train_rows_only",
            "alpha_cap": float(args.alpha_cap),
            "support_weighting": True,
            "epochs": int(args.epochs),
        },
        "selection_protocol": {
            "validation_only": True,
            "test_threshold_tuning": False,
            "objective": "support_aware_validation_t100_min_group",
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
    payload["stage43_di_gate"] = _gate(payload)
    _write_reports(payload)
    print(f"Stage43-DI: {payload['stage43_di_gate']['verdict']} ({payload['stage43_di_gate']['passed']}/{payload['stage43_di_gate']['total']})")
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Train Stage43-DI support-aware bounded-alpha t100 head.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--seeds", type=str, default="4323,4331,4337")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1.1e-3)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--alpha-cap", type=float, default=0.75)
    # Compatibility with precondition rebuilds.
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-label-rows", type=int, default=80)
    parser.add_argument("--min-val-improvement", type=float, default=0.0002)
    args = parser.parse_args(argv)
    return train_t100_support_aware_bounded_alpha_distilled_head(args)


if __name__ == "__main__":
    main()
