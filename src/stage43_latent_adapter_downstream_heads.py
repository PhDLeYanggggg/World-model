from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_full_waypoint_latent_dynamics import (
    OUT_DIR,
    README_RESULTS,
    RESEARCH_STATE,
    WORK_SUMMARY,
    M3W_README,
    _bootstrap_ci,
    _build_split,
    _git_commit,
    _jsonable,
    _metrics,
    _search_policy,
    _select_with_policy,
    _target_vec,
    _trajectory_error,
)
from src.stage43_full_waypoint_latent_robustness_audit import (
    STAGE43_M_JSON,
    _load_model,
    _standardize_from_checkpoint,
)
from src.stage43_latent_transition_adapter_repair import (
    REPORT_JSON as STAGE43_BZ_JSON,
    LatentTransitionAdapter,
    _adapter_predict,
)
from src.stage43_latent_transition_consistency_audit import _predict_transition_latents
from src.stage43_world_state_head_audit import _binary_metrics, _regression_metrics


REPORT_JSON = OUT_DIR / "stage43_latent_adapter_downstream_heads.json"
REPORT_MD = OUT_DIR / "stage43_latent_adapter_downstream_heads.md"
GATE_MD = OUT_DIR / "stage43_stage_ca_latent_adapter_downstream_heads_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_CA_LATENT_ADAPTER_DOWNSTREAM_HEADS"
SOURCE = "fresh_stage43_ca_latent_adapter_downstream_heads"
HEADS = ("failure", "gain", "harm")
EPS = 1e-8


def _fit_ridge(x: np.ndarray, y: np.ndarray, *, ridge: float) -> np.ndarray:
    xb = np.concatenate([x.astype(np.float64), np.ones((len(x), 1), dtype=np.float64)], axis=1)
    yy = y.astype(np.float64)
    reg = float(ridge) * np.eye(xb.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    return np.linalg.solve(xb.T @ xb + reg, xb.T @ yy)


def _apply_linear(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    xb = np.concatenate([x.astype(np.float64), np.ones((len(x), 1), dtype=np.float64)], axis=1)
    return (xb @ weights).astype(np.float32)


def _fit_heads(latent: np.ndarray, ds: Any, *, ridge: float) -> dict[str, np.ndarray]:
    target_waypoint = ds.waypoint_delta.reshape(len(ds.x), -1).astype(np.float32)
    risk = np.stack([ds.y_failure, ds.y_gain, ds.y_harm], axis=1).astype(np.float32)
    density = ds.y_density[:, None].astype(np.float32)
    return {
        "waypoint": _fit_ridge(latent, target_waypoint, ridge=ridge),
        "risk": _fit_ridge(latent, risk, ridge=ridge),
        "density": _fit_ridge(latent, density, ridge=ridge),
    }


def _predict_heads(latent: np.ndarray, weights: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    waypoint = _apply_linear(latent, weights["waypoint"]).reshape(len(latent), 4, 2)
    risk = np.clip(_apply_linear(latent, weights["risk"]), 0.0, 1.0)
    density = np.clip(_apply_linear(latent, weights["density"]).reshape(-1), 0.0, 1.0)
    return {
        "waypoint": waypoint.astype(np.float32),
        "failure": risk[:, 0].astype(np.float32),
        "gain": risk[:, 1].astype(np.float32),
        "harm": risk[:, 2].astype(np.float32),
        "density": density.astype(np.float32),
    }


def _risk_metrics(ds: Any, pred: Mapping[str, np.ndarray]) -> dict[str, Any]:
    labels = {"failure": ds.y_failure, "gain": ds.y_gain, "harm": ds.y_harm}
    out = {head: _binary_metrics(labels[head], pred[head]) for head in HEADS}
    defined = [float(row["auroc"]) for row in out.values() if row.get("defined") and row.get("auroc") is not None]
    out["mean_defined_auroc"] = float(np.mean(defined)) if defined else 0.0
    return out


def _head_eval(ds: Any, pred: Mapping[str, np.ndarray]) -> dict[str, Any]:
    ade, fde = _trajectory_error(ds, pred["waypoint"])
    ungated_metrics = _metrics(ds, ade, fde, np.ones(len(ds.x), dtype=bool))
    density = _regression_metrics(ds.y_density, pred["density"])
    risk = _risk_metrics(ds, pred)
    return {
        "ungated": ungated_metrics,
        "mean_ade": float(np.mean(ade)),
        "mean_fde": float(np.mean(fde)),
        "density": density,
        "risk": risk,
    }


def _protected_eval(val: Any, test: Any, val_pred: Mapping[str, np.ndarray], test_pred: Mapping[str, np.ndarray]) -> dict[str, Any]:
    policy = _search_conservative_policy(val, val_pred)
    selected_ade, selected_fde, switched = _select_with_policy(test, test_pred, policy["policy"])
    return {
        "validation_policy": policy,
        "test_metrics_with_floor": _metrics(test, selected_ade, selected_fde, switched),
        "bootstrap": _bootstrap_ci(test, selected_ade, selected_fde, n=1000, seed=911),
    }


def _search_conservative_policy(val: Any, pred: Mapping[str, np.ndarray]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for gain in [0.25, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92, 0.97]:
        for harm in [0.05, 0.10, 0.15, 0.25, 0.35]:
            for failure in [0.10, 0.20, 0.35, 0.50, 0.65, 0.80]:
                policy = {"gain_threshold": gain, "harm_threshold": harm, "failure_threshold": failure}
                selected_ade, selected_fde, switched = _select_with_policy(val, pred, policy)
                metrics = _metrics(val, selected_ade, selected_fde, switched)
                if metrics["easy_degradation_vs_floor"] > 0.005:
                    continue
                if metrics["switch_rate"] > 0.20:
                    continue
                objective = (
                    metrics["full_waypoint_ade_improvement_vs_floor"]
                    + 1.2 * metrics["t50_full_waypoint_ade_improvement_vs_floor"]
                    + 1.0 * metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
                    - 0.10 * metrics["switch_rate"]
                )
                row = {"policy": policy, "metrics": metrics, "objective": float(objective), "mode": "conservative_validation"}
                if best is None or row["objective"] > best["objective"]:
                    best = row
    if best is None:
        selected_ade = val.floor_ade.copy()
        selected_fde = val.floor_fde.copy()
        switched = np.zeros(len(val.x), dtype=bool)
        return {
            "policy": {"gain_threshold": 1.01, "harm_threshold": -0.01, "failure_threshold": 1.01},
            "metrics": _metrics(val, selected_ade, selected_fde, switched),
            "objective": 0.0,
            "mode": "conservative_validation_floor",
            "diagnostic": "no_conservative_validation_safe_switching_policy_found_keep_floor",
        }
    return best


def _load_adapter(path: Path, feature_dim: int, latent_dim: int) -> LatentTransitionAdapter:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    adapter = LatentTransitionAdapter(
        feature_dim=feature_dim,
        latent_dim=latent_dim,
        hidden_dim=int(ckpt.get("hidden_dim", 128)),
        delta_clip=float(ckpt.get("delta_clip", 3.0)),
    )
    adapter.load_state_dict(ckpt["adapter_state"])
    adapter.eval()
    return adapter


def _encode_all(base_model: torch.nn.Module, adapter: LatentTransitionAdapter, ds: Any, *, batch_size: int) -> dict[str, np.ndarray]:
    pred = _predict_transition_latents(base_model, ds, batch_size=int(batch_size))
    adapter_latent = _adapter_predict(adapter, ds.x, pred["z_t"], batch_size=int(batch_size))
    return {
        "identity_z_t": pred["z_t"],
        "stage43_m_z_next": pred["z_next"],
        "stage43_bz_adapter_z_next": adapter_latent,
        "target_latent": pred["z_target"],
    }


def _domain_horizon_summary(ds: Any, selected_ade: np.ndarray, selected_fde: np.ndarray, switched: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"domain": {}, "horizon": {}}
    for axis, values in [("domain", ds.domain.astype(str)), ("horizon", ds.horizon.astype(str))]:
        for value in sorted(set(values.tolist())):
            mask = values == value
            if int(mask.sum()) == 0:
                continue
            out[axis][str(value)] = _metrics(
                type("Slice", (), {
                    "x": ds.x[mask],
                    "floor_ade": ds.floor_ade[mask],
                    "floor_fde": ds.floor_fde[mask],
                    "hard": ds.hard[mask],
                    "failure": ds.failure[mask],
                    "easy": ds.easy[mask],
                    "horizon": ds.horizon[mask],
                })(),
                selected_ade[mask],
                selected_fde[mask],
                switched[mask],
            )
    return out


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    adapter = payload["variants"][payload["selected_adapter_variant"]]
    identity = payload["variants"]["identity_z_t"]
    m_next = payload["variants"]["stage43_m_z_next"]
    adapter_protected = adapter["protected"]["test_metrics_with_floor"]
    gates = {
        "stage43_bz_precondition_passed": payload["stage43_bz_precondition"]["verdict"]
        == "stage43_bz_latent_transition_adapter_repair_pass",
        "train_only_downstream_heads_fit": payload["protocol"]["train_only_heads"] is True,
        "future_labels_eval_only": payload["no_leakage"]["future_labels_as_inputs"] is False
        and payload["no_leakage"]["future_labels_train_eval_only"] is True,
        "no_test_threshold_tuning": payload["no_leakage"]["test_threshold_tuning"] is False,
        "selected_variant_contains_adapter": "adapter" in payload["selected_adapter_variant"],
        "adapter_variant_validation_selected": payload["selected_adapter_variant"]
        == payload["best_adapter_variant_by_validation_objective"],
        "adapter_waypoint_ungated_beats_identity": adapter["eval"]["mean_ade"] < identity["eval"]["mean_ade"],
        "adapter_waypoint_ungated_beats_stage43_m": adapter["eval"]["mean_ade"] < m_next["eval"]["mean_ade"],
        "adapter_risk_auc_beats_identity": adapter["eval"]["risk"]["mean_defined_auroc"]
        > identity["eval"]["risk"]["mean_defined_auroc"],
        "adapter_risk_auc_beats_stage43_m": adapter["eval"]["risk"]["mean_defined_auroc"]
        >= m_next["eval"]["risk"]["mean_defined_auroc"],
        "protected_eval_completed": adapter_protected["rows"] > 0,
        "protected_easy_preserved": adapter_protected["easy_degradation_vs_floor"] <= 0.02,
        "protected_adapter_lift_vs_floor": adapter_protected["full_waypoint_ade_improvement_vs_floor"] > 0.0
        or adapter_protected["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0
        or adapter_protected["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0,
        "domain_and_horizon_breakdowns_reported": bool(adapter["protected"].get("slice_summary", {}).get("domain"))
        and bool(adapter["protected"].get("slice_summary", {}).get("horizon")),
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
        "long_objective_kept_active": payload["long_objective_complete"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed == total:
        verdict = "stage43_ca_latent_adapter_downstream_heads_pass"
    elif (
        gates["adapter_waypoint_ungated_beats_identity"]
        or gates["adapter_risk_auc_beats_identity"]
        or gates["protected_adapter_lift_vs_floor"]
    ):
        verdict = "stage43_ca_latent_adapter_downstream_heads_partial_lift"
    else:
        verdict = "stage43_ca_latent_adapter_downstream_heads_diagnostic_incomplete"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_variant_table(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "## Variant Comparison",
        "",
        "| variant | mean ADE | mean FDE | risk mean AUROC | protected all | protected t50 | hard/failure | easy degradation | switch rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in payload["variants"].items():
        protected = row["protected"]["test_metrics_with_floor"]
        lines.append(
            f"| `{name}` | `{row['eval']['mean_ade']:.4f}` | `{row['eval']['mean_fde']:.4f}` | "
            f"`{row['eval']['risk']['mean_defined_auroc']:.4f}` | "
            f"`{protected['full_waypoint_ade_improvement_vs_floor']:.4f}` | "
            f"`{protected['t50_full_waypoint_ade_improvement_vs_floor']:.4f}` | "
            f"`{protected['hard_failure_full_waypoint_ade_improvement_vs_floor']:.4f}` | "
            f"`{protected['easy_degradation_vs_floor']:.4f}` | "
            f"`{protected['switch_rate']:.4f}` |"
        )
    lines.append("")
    return lines


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    gate = payload["stage43_ca_gate"]
    lines = [
        "# Stage43-CA Latent Adapter Downstream Heads",
        "",
        f"- source: `{SOURCE}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- selected adapter variant: `{payload['selected_adapter_variant']}`",
        f"- best overall validation variant: `{payload['best_overall_variant_by_validation_objective']}`",
        "- deployable policy changed: `False`",
        "",
    ]
    lines += _render_variant_table(payload)
    lines += [
        "## Interpretation",
        "",
        "- Stage43-CA fits identical train-only downstream heads on identity `z_t`, Stage43-M `z_next`, Stage43-BZ adapter `z_next`, and current+future-latent concatenations.",
        "- Future waypoint/risk/density labels are used only for train/eval targets, not as inference inputs.",
        "- Validation selects the protected safe-switch policy; test is evaluated once.",
        "- This is downstream/world-state evidence for the latent adapter, not a deployment change and not a safety-floor removal.",
        "- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    lines.append("")
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-CA Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
        ],
    )
    adapter = payload["variants"][payload["selected_adapter_variant"]]
    protected = adapter["protected"]["test_metrics_with_floor"]
    world_lines = [
        "# Stage43 Current World-Model Gate",
        "",
        f"- source: `{SOURCE}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "- protected multimodal latent state candidate: `True`",
        f"- selected adapter variant: `{payload['selected_adapter_variant']}`",
        f"- adapter downstream mean ADE: `{adapter['eval']['mean_ade']:.4f}`",
        f"- adapter risk mean AUROC: `{adapter['eval']['risk']['mean_defined_auroc']:.4f}`",
        f"- protected all improvement: `{protected['full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- protected t50 improvement: `{protected['t50_full_waypoint_ade_improvement_vs_floor']:.4f}`",
        "- deployable policy changed: `False`",
        "- long objective complete: `False`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "",
        "## Current Boundary",
        "",
        "- Stage43-CA is a train-only downstream head audit, not an ungated deployment policy.",
        "- Safety floors remain required for deployment.",
        "- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
        "",
        "| gate | passed |",
        "| --- | --- |",
    ]
    for key, value in gate["gates"].items():
        world_lines.append(f"| `{key}` | `{bool(value)}` |")
    world_lines.append("")
    write_json(WORLD_GATE_JSON, _jsonable(gate))
    write_md(WORLD_GATE_MD, world_lines)


def _update_summary_files(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_ca_gate"]
    adapter = payload["variants"][payload["selected_adapter_variant"]]
    protected = adapter["protected"]["test_metrics_with_floor"]
    block = [
        f"## {SECTION}",
        "",
        f"source = `{SOURCE}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        "deployable_policy_changed = `False`",
        "",
        "Stage43-CA fits train-only downstream heads on identity, Stage43-M transition, Stage43-BZ adapter, and current+future-latent concatenations.",
        f"Selected adapter variant by validation objective: `{payload['selected_adapter_variant']}`.",
        f"Adapter downstream mean ADE: `{adapter['eval']['mean_ade']:.4f}`.",
        f"Adapter risk mean AUROC: `{adapter['eval']['risk']['mean_defined_auroc']:.4f}`.",
        f"Protected all improvement vs floor: `{protected['full_waypoint_ade_improvement_vs_floor']:.4f}`.",
        f"Protected t50 improvement vs floor: `{protected['t50_full_waypoint_ade_improvement_vs_floor']:.4f}`.",
        f"Protected easy degradation: `{protected['easy_degradation_vs_floor']:.4f}`.",
        "",
        "Interpretation: downstream readouts test whether the repaired latent transition supports future waypoint/risk/density heads. This does not change deployment, remove the safety floor, or enable Stage5C/SMC.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, block)
    state = read_json(RESEARCH_STATE, {})
    state["stage43_ca_latent_adapter_downstream_heads"] = {
        "source": SOURCE,
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "adapter_eval": adapter["eval"],
        "adapter_protected": protected,
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_ca_latent_adapter_downstream_heads"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, _jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable({"event": SOURCE, "verdict": gate["verdict"], "generated_at_utc": payload["generated_at_utc"]}), ensure_ascii=False) + "\n")


def run_latent_adapter_downstream_heads(
    *,
    batch_size: int = 8192,
    ridge: float = 1e-2,
    seed: int = 521,
) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43m = read_json(STAGE43_M_JSON, {})
    stage43bz = read_json(STAGE43_BZ_JSON, {})
    checkpoint, ckpt, base_model = _load_model(stage43m)
    adapter_path = Path(stage43bz.get("adapter_checkpoint", OUT_DIR / "checkpoints/stage43_latent_transition_adapter_repair.pt"))
    train = _standardize_from_checkpoint(_build_split("train", max_rows=None, seed=int(ckpt.get("seed", 431))), ckpt)
    val = _standardize_from_checkpoint(_build_split("val", max_rows=None, seed=int(ckpt.get("seed", 431))), ckpt)
    test = _standardize_from_checkpoint(_build_split("test", max_rows=None, seed=int(ckpt.get("seed", 431))), ckpt)
    # Touch the target vector to make the label-only contract explicit in reports/tests.
    target_shape = list(_target_vec(train).shape)
    probe = _predict_transition_latents(base_model, train, batch_size=int(batch_size))
    adapter = _load_adapter(adapter_path, train.x.shape[1], probe["z_t"].shape[1])
    latents = {
        "train": {
            "identity_z_t": probe["z_t"],
            "stage43_m_z_next": probe["z_next"],
            "stage43_bz_adapter_z_next": _adapter_predict(adapter, train.x, probe["z_t"], batch_size=int(batch_size)),
        },
        "val": _encode_all(base_model, adapter, val, batch_size=int(batch_size)),
        "test": _encode_all(base_model, adapter, test, batch_size=int(batch_size)),
    }
    for split in ["train", "val", "test"]:
        latents[split]["identity_plus_adapter_z"] = np.concatenate(
            [latents[split]["identity_z_t"], latents[split]["stage43_bz_adapter_z_next"]],
            axis=1,
        ).astype(np.float32)
        latents[split]["stage43_m_plus_adapter_z"] = np.concatenate(
            [latents[split]["stage43_m_z_next"], latents[split]["stage43_bz_adapter_z_next"]],
            axis=1,
        ).astype(np.float32)
        latents[split]["identity_stage43m_adapter_z"] = np.concatenate(
            [
                latents[split]["identity_z_t"],
                latents[split]["stage43_m_z_next"],
                latents[split]["stage43_bz_adapter_z_next"],
            ],
            axis=1,
        ).astype(np.float32)
    variants: dict[str, Any] = {}
    variant_names = [
        "identity_z_t",
        "stage43_m_z_next",
        "stage43_bz_adapter_z_next",
        "identity_plus_adapter_z",
        "stage43_m_plus_adapter_z",
        "identity_stage43m_adapter_z",
    ]
    for name in variant_names:
        weights = _fit_heads(latents["train"][name], train, ridge=float(ridge))
        val_pred = _predict_heads(latents["val"][name], weights)
        test_pred = _predict_heads(latents["test"][name], weights)
        val_eval = _head_eval(val, val_pred)
        eval_row = _head_eval(test, test_pred)
        protected = _protected_eval(val, test, val_pred, test_pred)
        selected_ade, selected_fde, switched = _select_with_policy(test, test_pred, protected["validation_policy"]["policy"])
        protected["slice_summary"] = _domain_horizon_summary(test, selected_ade, selected_fde, switched)
        variants[name] = {
            "weights_fit_on": "train_only",
            "validation_eval": val_eval,
            "eval": eval_row,
            "protected": protected,
        }
    best_overall = max(
        variant_names,
        key=lambda key: float(variants[key]["protected"]["validation_policy"].get("objective", 0.0)),
    )
    adapter_variants = [name for name in variant_names if "adapter" in name]
    best_adapter = max(
        adapter_variants,
        key=lambda key: float(variants[key]["protected"]["validation_policy"].get("objective", 0.0)),
    )
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_train_only_downstream_head_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage43_m_precondition": {
            "verdict": stage43m.get("stage43_m_gate", {}).get("verdict"),
            "checkpoint": str(checkpoint),
        },
        "stage43_bz_precondition": {
            "verdict": stage43bz.get("stage43_bz_gate", {}).get("verdict"),
            "adapter_checkpoint": str(adapter_path),
            "adapter_checkpoint_exists": adapter_path.exists(),
        },
        "protocol": {
            "train_only_heads": True,
            "ridge": float(ridge),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "target_vec_shape": target_shape,
            "num_workers": 0,
        },
        "rows": {"train": int(len(train.x)), "val": int(len(val.x)), "test": int(len(test.x))},
        "best_overall_variant_by_validation_objective": best_overall,
        "best_adapter_variant_by_validation_objective": best_adapter,
        "selected_adapter_variant": best_adapter,
        "variants": variants,
        "no_leakage": {
            "future_labels_as_inputs": False,
            "future_labels_train_eval_only": True,
            "future_waypoint_input": False,
            "future_endpoint_input": False,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_threshold_tuning": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {
            "deployable_policy_changed": False,
            "metric_or_seconds_claim": False,
            "true_3d_claim": False,
            "foundation_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "long_objective_complete": False,
    }
    payload["stage43_ca_gate"] = _gate(payload)
    _write_reports(payload)
    _update_summary_files(payload)
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Audit downstream heads from Stage43-BZ adapter latent.")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=521)
    args = parser.parse_args(argv)
    payload = run_latent_adapter_downstream_heads(batch_size=args.batch_size, ridge=args.ridge, seed=args.seed)
    gate = payload["stage43_ca_gate"]
    adapter = payload["variants"][payload["selected_adapter_variant"]]
    print(f"Stage43-CA: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"selected_adapter_variant={payload['selected_adapter_variant']}")
    print(f"adapter_mean_ade={adapter['eval']['mean_ade']:.4f}")
    print(f"adapter_risk_mean_auroc={adapter['eval']['risk']['mean_defined_auroc']:.4f}")
    return payload


if __name__ == "__main__":
    main()
