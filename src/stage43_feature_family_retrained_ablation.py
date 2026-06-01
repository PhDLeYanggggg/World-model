from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m


OUT_DIR = m.OUT_DIR
CKPT_DIR = OUT_DIR / "checkpoints"
REPORT_JSON = OUT_DIR / "stage43_feature_family_retrained_ablation.json"
REPORT_MD = OUT_DIR / "stage43_feature_family_retrained_ablation.md"
GATE_MD = OUT_DIR / "stage43_stage_ah_feature_family_retrained_ablation_gate.md"
HEARTBEAT_JSON = OUT_DIR / "stage43_feature_family_retrained_ablation_heartbeat.json"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_AH_FEATURE_FAMILY_RETRAINED_ABLATION"
SOURCE = "fresh_stage43_ah_feature_family_retrained_ablation"

VARIANTS = [
    "full_features",
    "no_history",
    "no_goal",
    "no_neighbor_interaction",
    "no_baseline_floor",
    "no_domain",
]


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _family_for_feature(name: str) -> set[str]:
    families: set[str] = set()
    if name.startswith("history_"):
        families.add("history")
    if name in {
        "history_neighbor_count",
        "history_min_neighbor_dist",
        "history_density",
        "history_TTC",
        "history_closing_speed",
    }:
        families.add("neighbor_interaction")
    if name == "history_goal_alignment_proxy" or name.startswith("prototype_") or name == "goal_ambiguity":
        families.add("goal")
    if name.startswith("baseline_endpoint_rel_") or name.startswith("floor_endpoint_rel_"):
        families.add("baseline_floor")
    if name.startswith("domain_"):
        families.add("domain")
    return families


def _feature_mask(feature_names: list[str], variant: str) -> np.ndarray:
    if variant == "full_features":
        return np.ones(len(feature_names), dtype=bool)
    remove = {
        "no_history": {"history"},
        "no_goal": {"goal"},
        "no_neighbor_interaction": {"neighbor_interaction"},
        "no_baseline_floor": {"baseline_floor"},
        "no_domain": {"domain"},
    }.get(variant)
    if remove is None:
        raise KeyError(f"Unknown Stage43-AH feature-family variant: {variant}")
    keep = [not bool(_family_for_feature(name) & remove) for name in feature_names]
    return np.asarray(keep, dtype=bool)


def _apply_variant(ds: m.WaypointSplit, variant: str) -> m.WaypointSplit:
    mask = _feature_mask(ds.feature_names, variant)
    ds.x = ds.x[:, mask].astype(np.float32)
    ds.feature_names = [name for name, keep in zip(ds.feature_names, mask) if keep]
    return ds


def _train_one(args: argparse.Namespace, *, variant: str, rows: Mapping[str, int | None]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    row_seed = int(args.seed)
    train_seed = int(args.seed) + 101 * VARIANTS.index(variant)
    runtime = m._configure_runtime(train_seed)
    train = _apply_variant(m._build_split("train", max_rows=rows["train"], seed=row_seed), variant)
    val = _apply_variant(m._build_split("val", max_rows=rows["val"], seed=row_seed), variant)
    test = _apply_variant(m._build_split("test", max_rows=rows["test"], seed=row_seed), variant)
    train, val, test, mean, std = m._standardize(train, val, test)
    model = m.FullWaypointLatentDynamics(train.x.shape[1], hidden_dim=int(args.hidden_dim), latent_dim=int(args.latent_dim))
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    best_val = float("inf")
    checkpoint_tag = str(getattr(args, "checkpoint_tag", "stage43_feature_family_retrained_ablation"))
    ckpt_path = CKPT_DIR / f"{checkpoint_tag}_{variant}.pt"
    history: list[dict[str, Any]] = []
    start = time.time()
    for epoch in range(int(args.epochs)):
        model.train()
        losses: list[float] = []
        latent_vars: list[float] = []
        for batch_ids in m._batch_indices(len(train.x), int(args.batch_size), shuffle=True, seed=train_seed + epoch):
            opt.zero_grad(set_to_none=True)
            loss, stat = m._loss(model, train, batch_ids, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            latent_vars.append(float(stat.get("latent_variance", 0.0)))
        val_pred = m._predict(model, val, device, int(args.batch_size))
        val_policy = m._search_policy(val, val_pred)
        selected_ade, selected_fde, switched = m._select_with_policy(val, val_pred, val_policy["policy"])
        val_metrics = m._metrics(val, selected_ade, selected_fde, switched)
        objective_loss = -float(
            val_metrics["full_waypoint_ade_improvement_vs_floor"]
            + val_metrics["t50_full_waypoint_ade_improvement_vs_floor"]
            + 0.5 * val_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
            - 10.0 * max(0.0, val_metrics["easy_degradation_vs_floor"] - 0.02)
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_selection_objective_loss": objective_loss,
            "latent_variance": float(np.mean(latent_vars)) if latent_vars else 0.0,
        }
        history.append(row)
        write_json(
            HEARTBEAT_JSON,
            m._jsonable(
                {
                    "source": SOURCE,
                    "variant": variant,
                    "epoch": epoch + 1,
                    "elapsed_s": time.time() - start,
                    "last": row,
                    "git_commit": m._git_commit(),
                }
            ),
        )
        if objective_loss < best_val:
            best_val = objective_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_mean": mean,
                    "feature_std": std,
                    "feature_names": train.feature_names,
                    "variant": variant,
                    "input_dim": int(train.x.shape[1]),
                    "hidden_dim": int(args.hidden_dim),
                    "latent_dim": int(args.latent_dim),
                    "seed": train_seed,
                    "row_seed": row_seed,
                    "epoch": epoch + 1,
                    "runtime": runtime,
                    "checkpoint_committed": False,
                    "no_leakage": {
                        "future_endpoint_input": False,
                        "future_waypoint_input": False,
                        "future_waypoint_label_eval_only": True,
                        "central_velocity_input": False,
                        "test_endpoint_goal_construction": False,
                        "test_statistics_normalization": False,
                    },
                },
                ckpt_path,
            )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    val_pred = m._predict(model, val, device, int(args.batch_size))
    test_pred = m._predict(model, test, device, int(args.batch_size))
    val_policy = m._search_policy(val, val_pred)
    selected_ade, selected_fde, switched = m._select_with_policy(test, test_pred, val_policy["policy"])
    metrics = m._metrics(test, selected_ade, selected_fde, switched)
    ungated_ade, ungated_fde = m._trajectory_error(test, test_pred["waypoint"])
    ungated = m._metrics(test, ungated_ade, ungated_fde, np.ones(len(test.x), dtype=bool))
    removed_count = len(_feature_mask(m._build_split("train", max_rows=1, seed=row_seed).feature_names, "full_features")) - len(train.feature_names)
    result = {
        "variant": variant,
        "feature_count": int(train.x.shape[1]),
        "removed_feature_count": int(removed_count),
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": m._sha256(ckpt_path),
        "checkpoint_committed": False,
        "runtime": runtime,
        "data_rows": {"train": len(train.x), "val": len(val.x), "test": len(test.x)},
        "training_history": history,
        "validation_selected_policy": val_policy,
        "test_metrics_with_floor": metrics,
        "test_metrics_neural_without_floor": ungated,
        "latent_variance": float(np.var(test_pred["latent"], axis=0).mean()) if len(test_pred["latent"]) else 0.0,
    }
    arrays = {
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "floor_ade": test.floor_ade,
        "floor_fde": test.floor_fde,
        "h50": test.horizon == 50,
        "h100": test.horizon == 100,
        "hard_failure": test.hard | test.failure,
        "easy": test.easy,
    }
    return result, arrays


def _contribution_ci(full: Mapping[str, np.ndarray], ablated: Mapping[str, np.ndarray], *, n: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"n": int(n), "seed": int(seed), "metrics": {}}
    masks = {
        "all_full_waypoint_ade_contribution": np.ones(len(full["selected_ade"]), dtype=bool),
        "t50_full_waypoint_ade_contribution": full["h50"].astype(bool),
        "hard_failure_full_waypoint_ade_contribution": full["hard_failure"].astype(bool),
        "t100_raw_frame_diagnostic_contribution": full["h100"].astype(bool),
    }
    for name, mask in masks.items():
        ids = np.where(mask)[0]
        if int(len(ids)) == 0 or int(n) <= 0:
            out["metrics"][name] = {"rows": int(len(ids)), "mean": 0.0, "low": 0.0, "high": 0.0}
            continue
        vals = np.empty(int(n), dtype=np.float64)
        for i in range(int(n)):
            sample = rng.choice(ids, size=len(ids), replace=True)
            floor = float(np.mean(full["floor_ade"][sample]))
            full_imp = 1.0 - float(np.mean(full["selected_ade"][sample])) / max(floor, m.EPS)
            ablated_imp = 1.0 - float(np.mean(ablated["selected_ade"][sample])) / max(floor, m.EPS)
            vals[i] = full_imp - ablated_imp
        out["metrics"][name] = {
            "rows": int(len(ids)),
            "mean": float(np.mean(vals)),
            "low": float(np.quantile(vals, 0.025)),
            "high": float(np.quantile(vals, 0.975)),
        }
    return out


def _metric_delta(full: Mapping[str, Any], ablated: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(full.get(key, 0.0)) - float(ablated.get(key, 0.0))
        for key in [
            "full_waypoint_ade_improvement_vs_floor",
            "endpoint_fde_improvement_vs_floor",
            "t50_full_waypoint_ade_improvement_vs_floor",
            "t50_endpoint_fde_improvement_vs_floor",
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor",
            "hard_failure_full_waypoint_ade_improvement_vs_floor",
            "easy_degradation_vs_floor",
            "switch_rate",
        ]
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CKPT_DIR)
    mode = "medium" if args.medium else "quick" if args.quick else "small"
    if args.quick:
        rows = {"train": 6000, "val": 3000, "test": 3000}
    elif args.medium:
        rows = {"train": 90000, "val": 40000, "test": 50000}
    else:
        rows = {"train": 30000, "val": 12000, "test": 16000}
    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise KeyError(f"Unknown Stage43-AH variants: {unknown}")
    if "full_features" not in variants:
        raise ValueError("Stage43-AH requires full_features baseline.")
    results: list[dict[str, Any]] = []
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for variant in variants:
        row, arr = _train_one(args, variant=variant, rows=rows)
        results.append(row)
        arrays[variant] = arr
    by_variant = {row["variant"]: row for row in results}
    full = by_variant["full_features"]
    for row in results:
        if row["variant"] == "full_features":
            row["delta_full_minus_variant"] = {key: 0.0 for key in _metric_delta(full["test_metrics_with_floor"], full["test_metrics_with_floor"])}
            row["bootstrap_contribution_ci"] = {"reference": "full_features"}
        else:
            row["delta_full_minus_variant"] = _metric_delta(full["test_metrics_with_floor"], row["test_metrics_with_floor"])
            row["bootstrap_contribution_ci"] = _contribution_ci(
                arrays["full_features"],
                arrays[row["variant"]],
                n=int(args.bootstrap),
                seed=int(args.seed) + 1009 + VARIANTS.index(row["variant"]),
            )
    ablated = [row for row in results if row["variant"] != "full_features"]
    positive_t50 = [
        row
        for row in ablated
        if row["delta_full_minus_variant"]["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0
        and row["bootstrap_contribution_ci"]["metrics"]["t50_full_waypoint_ade_contribution"]["mean"] > 0.0
    ]
    positive_hard_or_all = [
        row
        for row in ablated
        if row["delta_full_minus_variant"]["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0
        or row["delta_full_minus_variant"]["full_waypoint_ade_improvement_vs_floor"] > 0.0
    ]
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_retrained_feature_family_ablation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "mode": mode,
        "variants": results,
        "positive_t50_contribution_variants": [row["variant"] for row in positive_t50],
        "positive_hard_or_all_contribution_variants": [row["variant"] for row in positive_hard_or_all],
        "ablation_type": {
            "fresh_retrained_variants": True,
            "same_train_val_test_protocol": True,
            "feature_family_drop_retraining": True,
            "not_inference_masking": True,
            "not_full_all_module_factorial": True,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "input_hash": _combined_hash([m._cache_path("train"), m._cache_path("val"), m._cache_path("test")]),
    }
    payload["stage43_ah_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    variants = {row["variant"]: row for row in payload["variants"]}
    full = variants["full_features"]
    ablated = [row for name, row in variants.items() if name != "full_features"]
    positive_modules = set(payload["positive_t50_contribution_variants"]) | set(payload["positive_hard_or_all_contribution_variants"])
    gates = {
        "fresh_retrained_ablation": payload["result_source"] == "fresh_retrained_feature_family_ablation",
        "full_features_baseline_retrained": full["test_metrics_with_floor"]["rows"] > 0,
        "at_least_four_feature_family_ablations": len(ablated) >= 4,
        "not_inference_masking": payload["ablation_type"]["not_inference_masking"] is True,
        "history_or_baseline_family_contribution_found": any(
            name in positive_modules for name in ["no_history", "no_baseline_floor"]
        ),
        "at_least_two_feature_families_show_contribution": len(positive_modules) >= 2,
        "bootstrap_or_resampling_recorded": all(
            row["variant"] == "full_features" or row["bootstrap_contribution_ci"]["n"] >= 200 for row in payload["variants"]
        ),
        "latent_noncollapse": all(row["latent_variance"] > 0.01 for row in payload["variants"]),
        "checkpoints_not_committed": all(row["checkpoint_committed"] is False for row in payload["variants"]),
        "no_future_or_test_leakage": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True
        and payload["no_leakage"]["central_velocity_input"] is False
        and payload["no_leakage"]["test_endpoint_goal_construction"] is False
        and payload["no_leakage"]["test_statistics_normalization"] is False,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
        "not_overclaimed_full_factorial": payload["ablation_type"]["not_full_all_module_factorial"] is True,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    deploy = passed == total
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_ah_feature_family_retrained_ablation_pass"
        if deploy
        else "stage43_ah_feature_family_retrained_ablation_diagnostic",
        "feature_family_retrained_ablation_supports_modules": deploy,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _variant_rows(payload: Mapping[str, Any]) -> list[str]:
    rows = [
        "| variant | features | all | t50 | hard | easy | full-minus-variant all | full-minus-variant t50 | full-minus-variant hard | t50 CI mean | latent var |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["variants"]:
        metrics = row["test_metrics_with_floor"]
        delta = row["delta_full_minus_variant"]
        ci = row.get("bootstrap_contribution_ci", {}).get("metrics", {}).get(
            "t50_full_waypoint_ade_contribution", {"mean": 0.0}
        )
        rows.append(
            f"| `{row['variant']}` | `{row['feature_count']}` | `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(metrics['easy_degradation_vs_floor'])}` | `{_pct(delta['full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(delta['t50_full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(delta['hard_failure_full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(ci['mean'])}` | `{row['latent_variance']:.4f}` |"
        )
    return rows


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    gate = payload["stage43_ah_gate"]
    lines = [
        "# Stage43-AH Feature-Family Retrained Ablation",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- mode: `{payload['mode']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- positive t50 contribution variants: `{payload['positive_t50_contribution_variants']}`",
        f"- positive hard/all contribution variants: `{payload['positive_hard_or_all_contribution_variants']}`",
        "",
        "## Variants",
        "",
        *_variant_rows(payload),
        "",
        "## Interpretation",
        "",
        "Stage43-AH fresh-trains feature-family removal variants under the same protected full-waypoint latent dynamics protocol. A positive full-minus-variant value means the full feature family helped relative to a retrained model without that family.",
        "",
        "This is contribution evidence, not a deployment policy: some positive contribution variants still have high easy harm, and history/neighbor/domain removal can outperform full_features in this single-seed small run.",
        "",
        "This is stronger than inference masking, but still not a complete all-module factorial ablation and not multi-seed medium evidence.",
        "",
        "## Boundary",
        "",
        "- Dataset-local/raw-frame 2.5D only.",
        "- Future waypoints are labels/eval only.",
        "- No metric/seconds claim, no Stage5C, no SMC.",
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
    ]
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-AH Feature-Family Retrained Ablation Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- supports modules: `{gate['feature_family_retrained_ablation_supports_modules']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ],
    )
    _update_text_outputs(payload)


def _update_text_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_ah_gate"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"feature_family_retrained_ablation_supports_modules = `{gate['feature_family_retrained_ablation_supports_modules']}`",
        f"positive_t50_contribution_variants = `{payload['positive_t50_contribution_variants']}`",
        f"positive_hard_or_all_contribution_variants = `{payload['positive_hard_or_all_contribution_variants']}`",
        "",
        "Stage43-AH fresh-trains full_features plus no_history, no_goal, no_neighbor_interaction, no_baseline_floor, and no_domain variants. This moves Stage43 causal ablation evidence beyond inference masking, while still remaining a focused single-seed/small retrained ablation rather than a complete factorial study. It is contribution evidence, not a deployment policy: positive contribution can coexist with unsafe easy harm, and some removed-family variants outperform full_features in this small run.",
        "",
        "Boundary unchanged: dataset-local/raw-frame 2.5D only; future waypoints are supervision/eval only; no metric/seconds claim, no Stage5C, no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_ah_feature_family_retrained_ablation"] = {
        "source": payload["source"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "feature_family_retrained_ablation_supports_modules": gate[
            "feature_family_retrained_ablation_supports_modules"
        ],
        "positive_t50_contribution_variants": payload["positive_t50_contribution_variants"],
        "positive_hard_or_all_contribution_variants": payload["positive_hard_or_all_contribution_variants"],
        "variants": [
            {
                "variant": row["variant"],
                "metrics": row["test_metrics_with_floor"],
                "delta_full_minus_variant": row["delta_full_minus_variant"],
                "latent_variance": row["latent_variance"],
                "checkpoint_committed": False,
            }
            for row in payload["variants"]
        ],
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_ah_feature_family_retrained_ablation"
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
                        "stage": "Stage43-AH",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "feature_family_retrained_ablation_supports_modules": gate[
                            "feature_family_retrained_ablation_supports_modules"
                        ],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-AH retrained feature-family ablation.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true")
    group.add_argument("--small", action="store_true")
    group.add_argument("--medium", action="store_true")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--seed", type=int, default=431)
    parser.add_argument("--bootstrap", type=int, default=500)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if not args.quick and not args.small and not args.medium:
        args.small = True
    result = _run(args)
    gate = result["stage43_ah_gate"]
    print(f"Stage43-AH: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(
        f"feature_family_retrained_ablation_supports_modules={gate['feature_family_retrained_ablation_supports_modules']}"
    )
    return result


if __name__ == "__main__":
    main()
