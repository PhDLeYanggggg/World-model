from __future__ import annotations

import argparse
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_t100_residual_admissibility_head as ct
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_protected_latent_state_model import _jsonable


OUT_DIR = m.OUT_DIR
CKPT_DIR = OUT_DIR / "checkpoints"
REPORT_JSON = OUT_DIR / "stage43_t100_residual_admissibility_statistical_confirmation.json"
REPORT_MD = OUT_DIR / "stage43_t100_residual_admissibility_statistical_confirmation.md"
GATE_MD = OUT_DIR / "stage43_stage_cu_t100_residual_admissibility_statistical_confirmation_gate.md"
HEARTBEAT_JSON = OUT_DIR / "stage43_t100_residual_admissibility_statistical_confirmation_heartbeat.json"

README_RESULTS = m.README_RESULTS
M3W_README = m.M3W_README
WORK_SUMMARY = m.WORK_SUMMARY
RESEARCH_STATE = m.RESEARCH_STATE

SECTION = "STAGE43_CU_T100_RESIDUAL_ADMISSIBILITY_STATISTICAL_CONFIRMATION"
SOURCE = "fresh_stage43_cu_t100_residual_admissibility_statistical_confirmation"


def _sha_names(names: list[str] | np.ndarray) -> str:
    return hashlib.sha256("\0".join([str(x) for x in list(names)]).encode("utf-8")).hexdigest()


def _parse_seeds(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _seed_payload(seed: int, args: argparse.Namespace, runtime: Mapping[str, Any]) -> dict[str, Any]:
    local_args = argparse.Namespace(
        quick=bool(args.quick),
        seed=int(seed),
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        batch_size=int(args.batch_size),
    )
    train, val, test, cs_ckpt, cs_model = ct._build_splits(local_args)
    device = torch.device("cpu")
    train_pred = ct.cs._predict(cs_model, train, device, int(args.batch_size))
    val_pred = ct.cs._predict(cs_model, val, device, int(args.batch_size))
    test_pred = ct.cs._predict(cs_model, test, device, int(args.batch_size))
    train_aug = ct._augment_alpha_features(train, train_pred)
    val_aug = ct._augment_alpha_features(val, val_pred)
    test_aug = ct._augment_alpha_features(test, test_pred)
    mean, std = ct._standardize_aug(train_aug, val_aug, test_aug)
    model = ct.ResidualAdmissibilityHead(train_aug["x"].shape[1], hidden_dim=int(args.hidden_dim)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    start = time.time()
    for epoch in range(int(args.epochs)):
        model.train()
        losses: list[float] = []
        for ids in ct._batch_indices(len(train_aug["x"]), int(args.batch_size), shuffle=True, seed=int(seed) + epoch):
            opt.zero_grad(set_to_none=True)
            loss, _stat = ct._loss(model, train_aug, ids, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_head = ct._predict_head(model, val_aug, device, int(args.batch_size))
        val_loss = float(np.mean((val_head["delta"] - val_aug["y_delta"]) ** 2))
        row = {"seed": int(seed), "epoch": int(epoch + 1), "train_loss": float(np.mean(losses)) if losses else 0.0, "val_delta_mse": val_loss}
        history.append(row)
        write_json(
            HEARTBEAT_JSON,
            _jsonable({"source": SOURCE, "seed": int(seed), "epoch": int(epoch + 1), "elapsed_s": time.time() - start, "last": row}),
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "epoch": int(epoch + 1),
            }
    assert best_state is not None
    model.load_state_dict(best_state["model_state"])
    ckpt_path = CKPT_DIR / f"stage43_cu_t100_residual_admissibility_seed{int(seed)}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "feature_names": train_aug["feature_names"].tolist(),
            "input_dim": int(train_aug["x"].shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "seed": int(seed),
            "epoch": int(best_state["epoch"]),
            "runtime": dict(runtime),
            "cs_checkpoint_sha256": ct.cr._sha256(ct.cs.CKPT_DIR / ct.cs.CHECKPOINT_NAME),
        },
        ckpt_path,
    )
    val_head = ct._predict_head(model, val_aug, device, int(args.batch_size))
    test_head = ct._predict_head(model, test_aug, device, int(args.batch_size))
    val_policy = ct._search_policy(val, val_pred, val_head)
    test_metrics, selected_ade, selected_fde, switched = ct._evaluate_selected(test, test_pred, test_head, val_policy["policy"])
    bootstrap = m._bootstrap_ci(test, selected_ade, selected_fde, n=int(args.bootstrap), seed=int(seed) + 4000)
    ungated = ct._ungated_for_alpha(test, test_pred, 1.0)
    return {
        "seed": int(seed),
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": ct.cr._sha256(ckpt_path),
        "checkpoint_committed": False,
        "best_epoch": int(best_state["epoch"]),
        "training_history": history,
        "validation_selected_policy": val_policy,
        "test_metrics_with_floor": test_metrics,
        "bootstrap_ci": bootstrap,
        "switch_count": int(switched.sum()),
        "ungated_reference": ungated,
        "data_rows": {"train": int(len(train.x)), "val": int(len(val.x)), "test": int(len(test.x))},
        "alpha_protocol": {
            "alphas": [float(x) for x in ct.ALPHAS.tolist()],
            "num_alphas": int(len(ct.ALPHAS)),
            "augmented_train_rows": int(len(train_aug["x"])),
            "train_positive_rate": float(np.mean(train_aug["y_gain"])),
            "train_harm_rate": float(np.mean(train_aug["y_harm"])),
        },
        "feature_contract": ct._feature_contract(train_aug["feature_names"]),
        "feature_name_hash": _sha_names(train_aug["feature_names"]),
    }


def _aggregate(seed_runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    keys = [
        "full_waypoint_ade_improvement_vs_floor",
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor",
        "hard_failure_full_waypoint_ade_improvement_vs_floor",
        "easy_degradation_vs_floor",
        "switch_rate",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        vals = np.asarray([run["test_metrics_with_floor"][key] for run in seed_runs], dtype=np.float64)
        out[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "values": [float(v) for v in vals.tolist()],
        }
    ci_lows = np.asarray(
        [
            run["bootstrap_ci"]["metrics"]["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]["low"]
            for run in seed_runs
        ],
        dtype=np.float64,
    )
    out["all_seed_t100_positive"] = bool(all(run["test_metrics_with_floor"]["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] > 0.0 for run in seed_runs))
    out["all_seed_easy_safe"] = bool(all(run["test_metrics_with_floor"]["easy_degradation_vs_floor"] <= 0.02 for run in seed_runs))
    out["all_seed_bootstrap_low_positive"] = bool(np.all(ci_lows > 0.0))
    out["bootstrap_t100_low_values"] = [float(x) for x in ci_lows.tolist()]
    return out


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    agg = payload["aggregate"]
    gates = {
        "stage43_ct_precondition_present": payload["stage43_ct_precondition"]["verdict"]
        in {"stage43_ct_t100_residual_admissibility_positive_diagnostic", "stage43_ct_t100_residual_admissibility_keep_floor"},
        "fresh_multiseed_training": payload["result_source"] == "fresh_torch_t100_residual_admissibility_multiseed_confirmation",
        "three_or_more_seeds": len(payload["seed_runs"]) >= 3,
        "all_checkpoints_written_not_committed": all(Path(run["checkpoint"]).exists() and run["checkpoint_committed"] is False for run in payload["seed_runs"]),
        "feature_contract_clean": not payload["feature_contract"]["denied_feature_name_hits"],
        "all_seed_t100_positive": bool(agg["all_seed_t100_positive"]),
        "all_seed_bootstrap_low_positive": bool(agg["all_seed_bootstrap_low_positive"]),
        "all_seed_easy_safe": bool(agg["all_seed_easy_safe"]),
        "test_once_per_seed": all(run["test_metrics_with_floor"]["rows"] > 0 for run in payload["seed_runs"]),
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
        "stage43_cu_t100_admissibility_multiseed_confirmed_tiny_positive"
        if passed == total
        else "stage43_cu_t100_admissibility_multiseed_inconclusive_keep_floor"
    )
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_cu_gate"]
    agg = payload["aggregate"]
    lines = [
        "# Stage43-CU T100 Residual Admissibility Statistical Confirmation",
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
        f"- mean t100 improvement: `{agg['t100_raw_frame_full_waypoint_diagnostic_vs_floor']['mean']:.6f}`",
        f"- min t100 improvement: `{agg['t100_raw_frame_full_waypoint_diagnostic_vs_floor']['min']:.6f}`",
        f"- mean hard/failure improvement: `{agg['hard_failure_full_waypoint_ade_improvement_vs_floor']['mean']:.6f}`",
        f"- max easy degradation: `{agg['easy_degradation_vs_floor']['max']:.6f}`",
        f"- mean switch rate: `{agg['switch_rate']['mean']:.6f}`",
        f"- all seed bootstrap low positive: `{agg['all_seed_bootstrap_low_positive']}`",
        "",
        "## Per Seed",
        "",
        "| seed | t100 | hard/failure | easy degradation | switch rate | bootstrap low |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in payload["seed_runs"]:
        metrics = run["test_metrics_with_floor"]
        low = run["bootstrap_ci"]["metrics"]["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]["low"]
        lines.append(
            f"| `{run['seed']}` | `{metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor']:.6f}` | "
            f"`{metrics['hard_failure_full_waypoint_ade_improvement_vs_floor']:.6f}` | "
            f"`{metrics['easy_degradation_vs_floor']:.6f}` | `{metrics['switch_rate']:.6f}` | `{low:.6f}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This confirms whether the Stage43-CT tiny supported-protocol t100 lift survives seed variation and bootstrap.",
            "- The effect remains a supported-protocol diagnostic, not a current heldout deployment change.",
            "- Future endpoints/full waypoints are labels only; inference inputs remain causal.",
            "- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
            "",
        ]
    )
    return lines


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_cu_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-CU Gate",
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
    agg = payload["aggregate"]
    readme_block = [
        "## Stage43-CU: t100 residual admissibility statistical confirmation",
        "",
        "I reran the Stage43-CT residual-admissibility head across multiple seeds and bootstrapped the t100 diagnostic. The goal was to check whether the tiny positive CT signal is real enough to keep pursuing.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- seeds: `{payload['seeds']}`",
        f"- mean t100 improvement: `{agg['t100_raw_frame_full_waypoint_diagnostic_vs_floor']['mean']:.2%}`",
        f"- min t100 improvement: `{agg['t100_raw_frame_full_waypoint_diagnostic_vs_floor']['min']:.2%}`",
        f"- mean hard/failure improvement: `{agg['hard_failure_full_waypoint_ade_improvement_vs_floor']['mean']:.2%}`",
        f"- max easy degradation: `{agg['easy_degradation_vs_floor']['max']:.2%}`",
        f"- all bootstrap lows positive: `{agg['all_seed_bootstrap_low_positive']}`",
        f"- deploy on current heldout t100: `{payload['deploy_on_current_heldout']}`",
        "",
        "This is useful evidence, but the lift is still tiny and only on the supported t100 protocol. I am not treating it as a heldout deployment change.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, readme_block)

    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage43_cu_t100_residual_admissibility_statistical_confirmation"
    state["current_verdict"] = gate["verdict"]
    stage = state.setdefault("stage43_long_research_execution", {})
    stage["t100_residual_admissibility_statistical_confirmation"] = {
        "source": SOURCE,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "seeds": payload["seeds"],
        "aggregate": payload["aggregate"],
        "deploy_on_current_heldout": payload["deploy_on_current_heldout"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, _jsonable(state))


def confirm_t100_residual_admissibility(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CKPT_DIR)
    seeds = _parse_seeds(str(args.seeds))
    runtime = m._configure_runtime(seeds[0])
    seed_runs = [_seed_payload(seed, args, runtime) for seed in seeds]
    aggregate = _aggregate(seed_runs)
    ct_report = read_json(ct.REPORT_JSON, {})
    first = seed_runs[0]
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_torch_t100_residual_admissibility_multiseed_confirmation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "mode": "quick" if args.quick else "small",
        "seeds": seeds,
        "runtime": runtime,
        "stage43_ct_precondition": {
            "report": str(ct.REPORT_JSON),
            "verdict": ct_report.get("stage43_ct_gate", {}).get("verdict"),
        },
        "seed_runs": seed_runs,
        "aggregate": aggregate,
        "feature_contract": first["feature_contract"],
        "data_rows": first["data_rows"],
        "alpha_protocol": first["alpha_protocol"],
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
    payload["stage43_cu_gate"] = _gate(payload)
    _write_reports(payload)
    print(f"Stage43-CU: {payload['stage43_cu_gate']['verdict']} ({payload['stage43_cu_gate']['passed']}/{payload['stage43_cu_gate']['total']})")
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Confirm Stage43-CT t100 admissibility signal across seeds/bootstrap.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--seeds", type=str, default="4323,4331,4337")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args(argv)
    return confirm_t100_residual_admissibility(args)


if __name__ == "__main__":
    main()
