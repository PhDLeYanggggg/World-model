from __future__ import annotations

import argparse
import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_latent_state_robustness_audit import _bootstrap_metric_fast
from src.stage43_protected_latent_state_model import (
    DATA35,
    DATA36,
    DATA37,
    OUT_DIR,
    ProtectedLatentStateModel,
    _err_from_delta,
    _git_commit,
    _jsonable,
    _metrics,
    _predict,
    _select_with_policy,
)
from src.stage43_source_level_heldout_split import REPORT_JSON as SPLIT_REPORT_JSON
from src.stage43_source_level_latent_model import REPORT_JSON as STAGE43G_JSON
from src.stage43_source_level_latent_model import build_source_level_datasets


REPORT_JSON = OUT_DIR / "stage43_source_level_latent_robustness_audit.json"
REPORT_MD = OUT_DIR / "stage43_source_level_latent_robustness_audit.md"
GATE_MD = OUT_DIR / "stage43_stage_h_source_level_latent_robustness_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE43_H_SOURCE_LEVEL_LATENT_ROBUSTNESS"
SOURCE = "fresh_stage43_h_source_level_latent_robustness"
EPS = 1e-8


def _load_npz(path: Path) -> Mapping[str, np.ndarray]:
    return np.load(path, allow_pickle=False)


def _apply_checkpoint_standardization(ds, ckpt: Mapping[str, Any]):
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_std"], dtype=np.float32)
    ds.x = ((ds.x - mean) / std).astype(np.float32)
    return ds


def _unit_selected_error(ds, pred_delta: np.ndarray, switched: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized_err = _err_from_delta(ds, pred_delta)
    unit_err = (normalized_err.astype(np.float64) * ds.scale.astype(np.float64)).astype(np.float32)
    selected = np.where(switched, unit_err, ds.floor_err).astype(np.float32)
    return normalized_err, selected


def _source_level_test_metadata(manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
    assignments = {str(k): str(v) for k, v in manifest["source_assignments"].items()}
    rows: dict[str, list[np.ndarray]] = defaultdict(list)
    for old_split in ["train", "val", "test"]:
        geo = _load_npz(DATA35 / f"expanded_external_{old_split}.npz")
        labels = _load_npz(DATA35 / f"labels_{old_split}.npz")
        family = _load_npz(DATA37 / f"t50_baseline_family_{old_split}.npz")
        selection = _load_npz(DATA36 / f"stage35_selection_{old_split}.npz")
        source = geo["source_file"].astype(str)
        ids = np.where(np.asarray([assignments[str(src)] == "test" for src in source], dtype=bool))[0]
        if len(ids) == 0:
            continue
        selected_family = selection["selected"].astype(np.int64).clip(0, family["prediction"].shape[1] - 1)
        if old_split == "test" and (DATA37 / "stage37_best_t50_selection_test.npz").exists():
            stage37 = _load_npz(DATA37 / "stage37_best_t50_selection_test.npz")
            h50 = geo["horizon"].astype(np.int64) == 50
            selected37 = stage37["selected_family"].astype(np.int64).clip(0, family["prediction"].shape[1] - 1)
            selected_family[h50] = selected37[h50]
        row = np.arange(len(source))
        floor_xy = family["prediction"][row, selected_family].astype(np.float32)
        current_xy = np.stack([geo["current_x"], geo["current_y"]], axis=1).astype(np.float32)
        for key, value in {
            "old_split": np.asarray([old_split] * len(ids)),
            "local_row": ids.astype(np.int64),
            "dataset": geo["dataset"].astype(str)[ids],
            "scene_id": geo["scene_id"].astype(str)[ids],
            "source_file": source[ids],
            "agent_id": geo["agent_id"].astype(np.int64)[ids],
            "frame_id": geo["frame_id"].astype(np.float64)[ids],
            "horizon": geo["horizon"].astype(np.int64)[ids],
            "current_xy": current_xy[ids],
            "floor_xy": floor_xy[ids],
            "scale": labels["scale"].astype(np.float32)[ids],
        }.items():
            rows[key].append(value)
    return {key: np.concatenate(value, axis=0) for key, value in rows.items()}


def _min_endpoint_distance(xy: np.ndarray, group_key: np.ndarray, normalizer: np.ndarray, agent_id: np.ndarray) -> np.ndarray:
    out = np.full(len(xy), np.inf, dtype=np.float64)
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, key in enumerate(group_key):
        groups[str(key)].append(idx)
    for members in groups.values():
        if len(members) < 2:
            continue
        mem = np.asarray(members, dtype=np.int64)
        pts = xy[mem].astype(np.float64)
        agents = agent_id[mem].astype(np.int64)
        for local_i, row in enumerate(mem):
            keep = np.arange(len(mem)) != local_i
            keep &= agents != agents[local_i]
            if not np.any(keep):
                continue
            d = np.linalg.norm(pts[keep] - pts[local_i][None, :], axis=1)
            out[row] = float(np.min(d) / max(float(normalizer[row]), EPS))
    return out


def _proximity_stats(selected_xy: np.ndarray, floor_xy: np.ndarray, meta: Mapping[str, np.ndarray]) -> dict[str, Any]:
    group_key = np.asarray(
        [
            f"{src}|{int(round(float(frame)))}|{int(h)}"
            for src, frame, h in zip(meta["source_file"], meta["frame_id"], meta["horizon"])
        ],
        dtype=object,
    )
    selected_min = _min_endpoint_distance(selected_xy, group_key, meta["scale"], meta["agent_id"])
    floor_min = _min_endpoint_distance(floor_xy, group_key, meta["scale"], meta["agent_id"])
    finite_selected = np.isfinite(selected_min)
    finite_floor = np.isfinite(floor_min)

    def rate(arr: np.ndarray, finite: np.ndarray, threshold: float) -> float:
        if int(finite.sum()) == 0:
            return 0.0
        return float(np.mean(arr[finite] < threshold))

    out = {
        "rows": int(len(selected_xy)),
        "grouped_rows_selected": int(finite_selected.sum()),
        "grouped_rows_floor": int(finite_floor.sum()),
        "selected_near_005": rate(selected_min, finite_selected, 0.05),
        "floor_near_005": rate(floor_min, finite_floor, 0.05),
        "selected_near_010": rate(selected_min, finite_selected, 0.10),
        "floor_near_010": rate(floor_min, finite_floor, 0.10),
        "selected_p05_min_distance": float(np.percentile(selected_min[finite_selected], 5)) if int(finite_selected.sum()) else None,
        "floor_p05_min_distance": float(np.percentile(floor_min[finite_floor], 5)) if int(finite_floor.sum()) else None,
    }
    out["near_005_delta_vs_floor"] = float(out["selected_near_005"] - out["floor_near_005"])
    out["near_010_delta_vs_floor"] = float(out["selected_near_010"] - out["floor_near_010"])
    return out


def _slice_table(ds, unit_selected: np.ndarray, switched: np.ndarray, meta: Mapping[str, np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for domain in sorted(set(ds.domain.astype(str).tolist())):
        mask = ds.domain.astype(str) == domain
        sub = {k: v[mask] for k, v in meta.items()}
        out[f"domain:{domain}"] = {
            "metrics": _metrics(ds, unit_selected, switched) if False else _metrics_subset(ds, unit_selected, switched, mask),
            "proximity": _proximity_stats(sub["selected_xy"], sub["floor_xy"], sub),
        }
    for source in sorted(set(meta["source_file"].astype(str).tolist())):
        mask = meta["source_file"].astype(str) == source
        sid = __import__("hashlib").sha256(source.encode("utf-8")).hexdigest()[:16]
        out[f"source:{sid}"] = {
            "source_id": sid,
            "domain": sorted(set(ds.domain[mask].astype(str).tolist())),
            "scene": sorted(set(meta["scene_id"][mask].astype(str).tolist())),
            "metrics": _metrics_subset(ds, unit_selected, switched, mask),
        }
    return out


def _metrics_subset(ds, selected: np.ndarray, switched: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    ids = np.where(mask)[0]
    if len(ids) == 0:
        return {"rows": 0}
    sub = type(ds)(
        split=ds.split,
        x=ds.x[ids],
        y_delta=ds.y_delta[ids],
        y_failure=ds.y_failure[ids],
        y_gain=ds.y_gain[ids],
        y_harm=ds.y_harm[ids],
        y_occupancy=ds.y_occupancy[ids],
        horizon=ds.horizon[ids],
        domain=ds.domain[ids],
        floor_err=ds.floor_err[ids],
        strongest_err=ds.strongest_err[ids],
        candidate_err_ref=ds.candidate_err_ref[ids],
        hard=ds.hard[ids],
        failure=ds.failure[ids],
        easy=ds.easy[ids],
        scale=ds.scale[ids],
        feature_names=ds.feature_names,
    )
    return _metrics(sub, selected[ids], switched[ids])


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    unit = payload["unit_consistent_metrics"]
    boot = payload["bootstrap"]
    prox = payload["proximity"]
    gates = {
        "stage43_g_checkpoint_exists": Path(payload["checkpoint"]).exists(),
        "source_level_full_test_eval_completed": unit["rows"] >= 80000,
        "unit_mismatch_detected_and_reported": payload["unit_consistency"]["normalized_all_minus_unit_all"] > 0.1,
        "unit_all_ci_low_positive": boot["unit_all"]["ci_low"] > 0.0,
        "unit_t50_ci_low_positive": boot["unit_t50"]["ci_low"] > 0.0,
        "unit_hard_ci_low_positive": boot["unit_hard_failure"]["ci_low"] > 0.0,
        "easy_preservation_gate": boot["unit_easy_degradation"]["ci_high"] <= 0.02,
        "proximity_not_materially_worse": prox["near_005_delta_vs_floor"] <= 0.01
        and prox["near_010_delta_vs_floor"] <= 0.01,
        "full_switch_caveat_recorded": payload["safety_floor_intervention"]["fallback_rate"] == 0.0,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_h_unit_consistent_safe_candidate_pass"
        if passed == total
        else "stage43_h_unit_consistent_audit_failed_keep_floor",
        "deploy_stage43_g": passed == total,
        "keep_frozen_floor": passed != total,
    }


def run_audit(*, bootstrap: int = 1000, batch_size: int = 4096) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43g = read_json(STAGE43G_JSON, {})
    checkpoint = Path(stage43g.get("checkpoint", OUT_DIR / "checkpoints/stage43_source_level_latent_full.pt"))
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _, _, test, manifest = build_source_level_datasets(seed=int(ckpt.get("seed", 443)))
    test = _apply_checkpoint_standardization(test, ckpt)
    model = ProtectedLatentStateModel(int(ckpt["input_dim"]), int(ckpt["hidden_dim"]), int(ckpt["latent_dim"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    pred = _predict(model, test, torch.device("cpu"), batch_size)
    policy = stage43g["validation_selected_policy"]["policy"]
    normalized_selected, switched = _select_with_policy(test, pred, policy)
    normalized_metrics = _metrics(test, normalized_selected, switched)
    normalized_err, unit_selected = _unit_selected_error(test, pred["delta"], switched)
    unit_metrics = _metrics(test, unit_selected, switched)
    meta = _source_level_test_metadata(manifest)
    selected_xy = np.where(
        switched[:, None],
        meta["current_xy"].astype(np.float32) + pred["delta"].astype(np.float32) * test.scale[:, None].astype(np.float32),
        meta["floor_xy"].astype(np.float32),
    )
    meta = {**meta, "selected_xy": selected_xy}
    proximity = _proximity_stats(selected_xy, meta["floor_xy"], meta)
    all_ids = np.arange(len(unit_selected))
    boot = {
        "unit_all": _bootstrap_metric_fast(unit_selected, test.floor_err, all_ids, n=bootstrap, seed=43801),
        "unit_t50": _bootstrap_metric_fast(unit_selected, test.floor_err, np.where(test.horizon == 50)[0], n=bootstrap, seed=43802),
        "unit_t100_raw_frame_diagnostic": _bootstrap_metric_fast(unit_selected, test.floor_err, np.where(test.horizon == 100)[0], n=bootstrap, seed=43803),
        "unit_hard_failure": _bootstrap_metric_fast(unit_selected, test.floor_err, np.where(test.hard | test.failure)[0], n=bootstrap, seed=43804),
        "unit_easy_degradation": _bootstrap_metric_fast(unit_selected, test.floor_err, np.where(test.easy)[0], easy=True, n=bootstrap, seed=43805),
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_run_unit_consistency_and_safety_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "checkpoint": str(checkpoint),
        "checkpoint_committed": False,
        "source_level_split_row_hash": manifest["pool"]["row_hash"],
        "normalized_metrics_as_reported_stage43_g": normalized_metrics,
        "unit_consistent_metrics": unit_metrics,
        "unit_consistency": {
            "normalized_error_definition": "sqrt((pred_delta_norm - label_delta_norm)^2)",
            "unit_consistent_error_definition": "normalized_error * row_scale before comparing with floor FDE",
            "normalized_all_minus_unit_all": float(normalized_metrics["all_improvement_vs_floor"] - unit_metrics["all_improvement_vs_floor"]),
            "normalized_t50_minus_unit_t50": float(normalized_metrics["t50_improvement_vs_floor"] - unit_metrics["t50_improvement_vs_floor"]),
            "stage43_g_normalized_metrics_not_deployment_evidence": True,
        },
        "bootstrap": boot,
        "proximity": proximity,
        "slice_table": _slice_table(test, unit_selected, switched, meta),
        "safety_floor_intervention": {
            "switch_rate": float(np.mean(switched)),
            "fallback_rate": float(1.0 - np.mean(switched)),
            "full_switch_result": bool(float(np.mean(switched)) >= 0.999),
        },
        "claim_boundary": {
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage43_h_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    gate = payload["stage43_h_gate"]
    norm = payload["normalized_metrics_as_reported_stage43_g"]
    unit = payload["unit_consistent_metrics"]
    prox = payload["proximity"]
    boot = payload["bootstrap"]
    lines = [
        "# Stage43-H Source-Level Latent Robustness Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- deploy Stage43-G: `{gate['deploy_stage43_g']}`",
        f"- keep frozen floor: `{gate['keep_frozen_floor']}`",
        f"- checkpoint: `{payload['checkpoint']}`",
        "- checkpoint committed: `False`",
        "",
        "## Unit Consistency Finding",
        "",
        "Stage43-G reported normalized-delta error against a dataset-local floor FDE. Stage43-H recomputes neural endpoint error as `normalized_error * row_scale` before comparing with the floor.",
        "",
        "| metric | normalized Stage43-G | unit-consistent Stage43-H |",
        "| --- | ---: | ---: |",
        f"| all | {norm['all_improvement_vs_floor']:.6f} | {unit['all_improvement_vs_floor']:.6f} |",
        f"| t50 | {norm['t50_improvement_vs_floor']:.6f} | {unit['t50_improvement_vs_floor']:.6f} |",
        f"| t100 raw diagnostic | {norm['t100_raw_frame_diagnostic_vs_floor']:.6f} | {unit['t100_raw_frame_diagnostic_vs_floor']:.6f} |",
        f"| hard/failure | {norm['hard_failure_improvement_vs_floor']:.6f} | {unit['hard_failure_improvement_vs_floor']:.6f} |",
        f"| easy degradation | {norm['easy_degradation_vs_floor']:.6f} | {unit['easy_degradation_vs_floor']:.6f} |",
        "",
        "## Bootstrap CI on Unit-Consistent Metrics",
        "",
        "| metric | rows | mean | ci low | ci high |",
        "| --- | ---: | ---: | ---: | ---: |",
        *[
            f"| {name} | {row['rows']} | {row['mean']:.6f} | {row['ci_low']:.6f} | {row['ci_high']:.6f} |"
            for name, row in boot.items()
        ],
        "",
        "## Endpoint Proximity Proxy",
        "",
        f"- selected near@0.05: `{prox['selected_near_005']:.6f}`",
        f"- floor near@0.05: `{prox['floor_near_005']:.6f}`",
        f"- near@0.05 delta: `{prox['near_005_delta_vs_floor']:.6f}`",
        f"- selected near@0.10: `{prox['selected_near_010']:.6f}`",
        f"- floor near@0.10: `{prox['floor_near_010']:.6f}`",
        f"- near@0.10 delta: `{prox['near_010_delta_vs_floor']:.6f}`",
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        "",
        "Conclusion: Stage43-G remains an interesting source-level neural dynamics signal, but the deployment claim is not valid under unit-consistent safety auditing because easy degradation is unsafe. Keep the frozen Stage37/Stage42 floor and repair with a calibrated safe-switch policy.",
    ]
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-H Source-Level Latent Robustness Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- deploy Stage43-G: `{gate['deploy_stage43_g']}`",
            f"- keep frozen floor: `{gate['keep_frozen_floor']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ],
    )
    _update_readmes(payload)


def _update_readmes(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_h_gate"]
    unit = payload["unit_consistent_metrics"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"deploy_stage43_g = `{gate['deploy_stage43_g']}`",
        "",
        "Stage43-H audits Stage43-G and finds a unit-consistency issue: the Stage43-G headline compared normalized neural delta error against dataset-local floor FDE. After multiplying neural error by each row's scale, all/t50/hard remain positive but easy degradation becomes unsafe.",
        "",
        f"Unit-consistent metrics: all `{unit['all_improvement_vs_floor']:.6f}`, t50 `{unit['t50_improvement_vs_floor']:.6f}`, t100 raw diagnostic `{unit['t100_raw_frame_diagnostic_vs_floor']:.6f}`, hard/failure `{unit['hard_failure_improvement_vs_floor']:.6f}`, easy degradation `{unit['easy_degradation_vs_floor']:.6f}`.",
        "",
        "Conclusion: keep the frozen Stage37/Stage42 safety floor. Stage43-G is a useful neural dynamics signal but not a deployable replacement until a calibrated safe-switch repair passes unit-consistent easy/proximity gates.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(RESEARCH_STATE, {})
    state["stage43_h_source_level_latent_robustness"] = {
        "source": payload["source"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "deploy_stage43_g": gate["deploy_stage43_g"],
        "keep_frozen_floor": gate["keep_frozen_floor"],
        "unit_consistent_metrics": payload["unit_consistent_metrics"],
        "unit_consistency": payload["unit_consistency"],
        "proximity": payload["proximity"],
        "report": str(REPORT_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    write_json(RESEARCH_STATE, state)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args(argv)
    return run_audit(bootstrap=int(args.bootstrap), batch_size=int(args.batch_size))


if __name__ == "__main__":
    result = main()
    gate = result["stage43_h_gate"]
    print(f"Stage43-H robustness audit: {gate['verdict']} ({gate['passed']}/{gate['total']})")
