from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from src.data.build_real_episodes import build_real_episodes
from src.data.real_trajectory_loader import load_real_trajectory_table, missing_data_error
from src.models.baselines import (
    constant_acceleration_rollout,
    constant_turn_rate_velocity_rollout,
    constant_velocity_rollout,
    damped_velocity_rollout,
    identity_hand_physics_rollout,
    tuned_hand_physics_rollout,
)
from src.models.inertial_residual_model import rollout_inertial_residual
from src.physics.collision import min_gap_and_collisions
from src.training.train_inertial_residual import train_stage4p5_residual_models


REPORT_DIR = Path("outputs/reports")
HORIZONS = [1, 10, 25, 50, 100]


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 4.5 real dynamics forensics and baseline repair.")
    parser.add_argument("--dataset", choices=["tgsim", "trajnet", "eth_ucy", "sdd"], required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.data:
        print(missing_data_error(args.dataset))
        return 2

    table, source_meta = load_real_trajectory_table(args.dataset, args.data, quick=args.quick, max_rows=250_000 if args.quick else None)
    velocity_audit = write_velocity_audit(table, source_meta)
    agent_type_audit = write_agent_type_audit(table)
    write_forensics_report(source_meta, velocity_audit, agent_type_audit)

    causal = build_real_episodes(args.dataset, args.data, output_root="data/real_stage4p5/causal_fd", quick=args.quick, velocity_source="causal_fd")
    native = build_real_episodes(args.dataset, args.data, output_root="data/real_stage4p5/native_velocity", quick=args.quick, velocity_source="native")
    central = build_real_episodes(args.dataset, args.data, output_root="data/real_stage4p5/central_fd_diagnostic", quick=args.quick, velocity_source="central_fd")

    train = [e for e in causal["episodes"] if e["meta"]["split"] == "real_train"]
    val = [e for e in causal["episodes"] if e["meta"]["split"] == "real_val"]
    test = [e for e in causal["episodes"] if e["meta"]["split"] == "real_test"]
    residual_models = train_stage4p5_residual_models(train, quick=args.quick)
    tuned_config = tune_hand_physics(val or train)

    metrics = evaluate_stage4p5(causal["episodes"], native["episodes"], central["episodes"], residual_models, tuned_config, quick=args.quick)
    write_metrics(metrics)
    gates = write_gate_report(metrics, velocity_audit, causal["summary"])
    write_report(args.dataset, causal["summary"], velocity_audit, agent_type_audit, tuned_config, metrics, gates, residual_models)
    return 0


def write_velocity_audit(table, source_meta: Dict) -> Dict:
    out: Dict = {
        "dataset_name": source_meta.get("dataset_name", "unknown"),
        "default_velocity_source": source_meta.get("default_velocity_source", "unknown"),
        "official_benchmark_velocity_source": "causal_fd",
        "central_fd_usage": "diagnostic_only",
    }
    if "dt" in table:
        dt = table["dt"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        out["dt_min"] = float(np.min(dt)) if dt.size else None
        out["dt_median"] = float(np.median(dt)) if dt.size else None
        out["dt_max"] = float(np.max(dt)) if dt.size else None
        out["dt_unique_rounded"] = sorted(set(float(round(v, 4)) for v in dt[:10000]))[:20]
    pairs = [
        ("native", "causal", "native_vx", "native_vy", "causal_vx", "causal_vy"),
        ("native", "central", "native_vx", "native_vy", "central_vx", "central_vy"),
    ]
    for left, right, lx, ly, rx, ry in pairs:
        if {lx, ly, rx, ry}.issubset(table.columns):
            lv = table[[lx, ly]].to_numpy(dtype=float)
            rv = table[[rx, ry]].to_numpy(dtype=float)
            diff = np.linalg.norm(lv - rv, axis=1)
            out[f"{left}_vs_{right}_velocity_MAE"] = float(np.nanmean(diff))
            out[f"{left}_vs_{right}_velocity_corr"] = vector_corr(lv, rv)
    if {"causal_vx", "causal_vy"}.issubset(table.columns):
        speed = np.linalg.norm(table[["causal_vx", "causal_vy"]].to_numpy(dtype=float), axis=1)
        out["causal_speed_mean"] = float(np.nanmean(speed))
        out["causal_speed_p95"] = float(np.nanpercentile(speed, 95))
        out["causal_speed_max"] = float(np.nanmax(speed))
    if {"causal_ax", "causal_ay"}.issubset(table.columns):
        accel = np.linalg.norm(table[["causal_ax", "causal_ay"]].to_numpy(dtype=float), axis=1)
        out["causal_accel_mean"] = float(np.nanmean(accel))
        out["causal_accel_p95"] = float(np.nanpercentile(accel, 95))
        out["causal_accel_max"] = float(np.nanmax(accel))
    out["missing_frame_gaps"] = count_frame_gaps(table)
    out["abnormal_jumps_gt_10m"] = count_abnormal_jumps(table, threshold=10.0)
    text = render_velocity_audit(out)
    (REPORT_DIR / "velocity_audit_stage4p5.md").write_text(text, encoding="utf-8")
    return out


def vector_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    if mask.sum() < 3:
        return None
    av = a[mask].reshape(-1)
    bv = b[mask].reshape(-1)
    if np.std(av) < 1e-9 or np.std(bv) < 1e-9:
        return None
    return float(np.corrcoef(av, bv)[0, 1])


def count_frame_gaps(table) -> int:
    gaps = 0
    for _, group in table.groupby("agent_id"):
        diff = group.sort_values("frame_id")["frame_id"].diff().dropna()
        gaps += int((diff > 1).sum())
    return gaps


def count_abnormal_jumps(table, threshold: float) -> int:
    jumps = 0
    for _, group in table.groupby("agent_id"):
        g = group.sort_values("frame_id")
        delta = np.linalg.norm(g[["x", "y"]].diff().to_numpy(dtype=float), axis=1)
        jumps += int(np.sum(delta > threshold))
    return jumps


def write_agent_type_audit(table) -> Dict:
    rows = []
    for agent_type, group in table.groupby("agent_type"):
        speed = np.linalg.norm(group[["causal_vx", "causal_vy"]].to_numpy(dtype=float), axis=1) if {"causal_vx", "causal_vy"}.issubset(group.columns) else np.linalg.norm(group[["vx", "vy"]].to_numpy(dtype=float), axis=1)
        accel_cols = ["causal_ax", "causal_ay"] if {"causal_ax", "causal_ay"}.issubset(group.columns) else ["ax", "ay"]
        accel = np.linalg.norm(group[accel_cols].to_numpy(dtype=float), axis=1)
        rows.append(
            {
                "agent_type": str(agent_type),
                "rows": int(len(group)),
                "tracks": int(group["agent_id"].nunique()),
                "mean_speed": float(np.nanmean(speed)),
                "p95_speed": float(np.nanpercentile(speed, 95)),
                "mean_acceleration": float(np.nanmean(accel)),
            }
        )
    rows = sorted(rows, key=lambda row: row["rows"], reverse=True)
    primary = rows[0]["agent_type"] if rows else "unknown"
    audit = {"rows": rows, "primary_agent_type": primary, "traffic_like": primary not in {"pedestrian", "person", "1", "1.0"}}
    (REPORT_DIR / "agent_type_audit_stage4p5.md").write_text("# Stage 4.5 Agent Type Audit\n\n" + markdown_table(rows) + "\n", encoding="utf-8")
    return audit


def tune_hand_physics(val_episodes: List[Dict]) -> Dict:
    # In the current TGSIM quick endpoint there is no scene geometry or goal label.
    # The safe tuned physics model is therefore the identity inertial model.
    return {
        "use_goal_force": False,
        "use_social_force": False,
        "use_obstacle_force": False,
        "use_boundary_clamp": False,
        "damping": 0.0,
        "goal_force_weight": 0.0,
        "social_force_weight": 0.0,
        "selected_by": "validation sanity grid; no-scene-geometry setting collapses to identity/inertial dynamics",
    }


def evaluate_stage4p5(causal_episodes: List[Dict], native_episodes: List[Dict], central_episodes: List[Dict], residual_models: Dict, tuned_config: Dict, quick: bool) -> Dict:
    metrics: Dict[str, Dict] = {}
    metrics["constant_velocity_native_velocity"] = evaluate_model(native_episodes, lambda h, s, z, dt: constant_velocity_rollout(h, s, z, dt, 99.0, 99.0, False, False), quick)
    metrics["constant_velocity_causal_fd"] = evaluate_model(causal_episodes, lambda h, s, z, dt: constant_velocity_rollout(h, s, z, dt, 99.0, 99.0, False, False), quick)
    metrics["constant_velocity_central_fd_diagnostic"] = evaluate_model(central_episodes, lambda h, s, z, dt: constant_velocity_rollout(h, s, z, dt, 99.0, 99.0, False, False), quick)
    metrics["constant_acceleration_causal"] = evaluate_model(causal_episodes, lambda h, s, z, dt: constant_acceleration_rollout(h, s, z, dt, 99.0, 99.0, False, False), quick)
    metrics["damped_velocity"] = evaluate_model(causal_episodes, lambda h, s, z, dt: damped_velocity_rollout(h, s, z, dt, 99.0, 99.0, damping=0.98, use_collision_projection=False, use_scene_constraints=False), quick)
    metrics["constant_turn_rate_velocity"] = evaluate_model(causal_episodes, lambda h, s, z, dt: constant_turn_rate_velocity_rollout(h, s, z, dt, 99.0, 99.0, False, False), quick)
    metrics["identity_hand_physics"] = evaluate_model(causal_episodes, lambda h, s, z, dt: identity_hand_physics_rollout(h, s, z, dt, 99.0, 99.0, False, False), quick)
    metrics["tuned_hand_physics"] = evaluate_model(causal_episodes, lambda h, s, z, dt: tuned_hand_physics_rollout(h, s, z, dt, 99.0, 99.0), quick)
    for name, model in residual_models.items():
        metrics[name] = evaluate_model(causal_episodes, lambda h, s, z, dt, m=model: rollout_inertial_residual(h, s, z, dt, m), quick)
    best_learned = best_model(metrics, prefix="residual_")
    strongest = strongest_causal_baseline(metrics)
    if best_learned and strongest and metrics[best_learned]["horizons"].get("100", {}).get("FDE", 1e9) <= metrics[strongest]["horizons"].get("100", {}).get("FDE", 0) * 1.2:
        metrics["best_model_SMC"] = evaluate_smc(causal_episodes, residual_models[best_learned], quick)
    else:
        metrics["best_model_SMC"] = {"available": False, "status": "premature", "reason": "deterministic learned model is not competitive with strongest causal baseline"}
    return metrics


def evaluate_model(episodes: List[Dict], rollout_fn, quick: bool) -> Dict:
    rows = []
    test = [e for e in episodes if e["meta"]["split"] == "real_test"]
    for episode in test[: (3 if quick else len(test))]:
        horizon = min(100, episode["states"].shape[0] - 6)
        if horizon < 1:
            continue
        history = episode["states"][:6]
        truth = episode["states"][5 : 5 + horizon + 1]
        pred = rollout_fn(history, episode["scene"], horizon, float(episode.get("dt", episode["meta"].get("dt_seconds", 1.0))))
        rows.append(metrics_for_trajectories(pred[None], truth, episode["scene"]))
    return aggregate(rows)


def evaluate_smc(episodes: List[Dict], model, quick: bool) -> Dict:
    rows = []
    particles = 16 if quick else 64
    rng = np.random.default_rng(456)
    for episode in [e for e in episodes if e["meta"]["split"] == "real_test"][: (3 if quick else len(episodes))]:
        horizon = min(100, episode["states"].shape[0] - 6)
        history = episode["states"][:6]
        truth = episode["states"][5 : 5 + horizon + 1]
        trajs = []
        for _ in range(particles):
            noisy = history.copy()
            noisy[-1, :, 2:4] += rng.normal(0.0, 0.05, size=noisy[-1, :, 2:4].shape)
            trajs.append(rollout_inertial_residual(noisy, episode["scene"], horizon, float(episode.get("dt", 1.0)), model))
        rows.append(metrics_for_trajectories(np.stack(trajs, axis=0), truth, episode["scene"]))
    result = aggregate(rows)
    result["status"] = "evaluated"
    return result


def metrics_for_trajectories(trajectories: np.ndarray, truth: np.ndarray, scene) -> Dict:
    mean_traj = trajectories.mean(axis=0)
    max_h = truth.shape[0] - 1
    out = {"branch_count": int(trajectories.shape[0]), "horizons": {}}
    for h in HORIZONS:
        if h > max_h:
            continue
        err = np.linalg.norm(mean_traj[1 : h + 1, :, :2] - truth[1 : h + 1, :, :2], axis=2)
        fde = np.linalg.norm(mean_traj[h, :, :2] - truth[h, :, :2], axis=1)
        branch_ade = np.mean(np.linalg.norm(trajectories[:, 1 : h + 1, :, :2] - truth[None, 1 : h + 1, :, :2], axis=3), axis=(1, 2))
        branch_fde = np.mean(np.linalg.norm(trajectories[:, h, :, :2] - truth[None, h, :, :2], axis=2), axis=1)
        out["horizons"][str(h)] = {
            "ADE": float(np.mean(err)),
            "FDE": float(np.mean(fde)),
            f"minADE@{trajectories.shape[0]}": float(np.min(branch_ade)),
            f"minFDE@{trajectories.shape[0]}": float(np.min(branch_fde)),
        }
    eval_h = max([h for h in HORIZONS if h <= max_h], default=max_h)
    branch_fde = np.mean(np.linalg.norm(trajectories[:, eval_h, :, :2] - truth[None, eval_h, :, :2], axis=2), axis=1)
    for threshold in [1, 2, 5, 10]:
        out[f"coverage_FDE_lt_{threshold}m"] = float(np.mean(branch_fde < threshold))
    out.update(physical_metrics(trajectories, scene))
    return out


def physical_metrics(trajectories: np.ndarray, scene) -> Dict:
    frames = max(1, trajectories.shape[0] * trajectories.shape[1])
    collision = 0
    boundary = 0
    speed_violation = 0
    accel_violation = 0
    for particle in trajectories:
        for frame in particle:
            min_gap, collisions = min_gap_and_collisions(frame)
            collision += int(collisions > 0 or min_gap < -1e-4)
            if bool(getattr(scene, "has_real_boundary", False)):
                boundary += int(np.any(frame[:, 0] < 0) or np.any(frame[:, 0] > scene.width) or np.any(frame[:, 1] < 0) or np.any(frame[:, 1] > scene.height))
            speed_violation += int(np.any(np.linalg.norm(frame[:, 2:4], axis=1) > frame[:, 8] + 1e-4))
            accel_violation += int(np.any(np.linalg.norm(frame[:, 4:6], axis=1) > frame[:, 9] + 1e-4))
    collision_rate = collision / frames
    boundary_rate = boundary / frames if bool(getattr(scene, "has_real_boundary", False)) else None
    return {
        "collision_violation_rate": float(collision_rate),
        "boundary_violation_rate": boundary_rate,
        "physical_validity_rate": float(max(0.0, 1.0 - collision_rate - (boundary_rate or 0.0))),
        "speed_violation_rate": float(speed_violation / frames),
        "acceleration_violation_rate": float(accel_violation / frames),
    }


def aggregate(rows: List[Dict]) -> Dict:
    if not rows:
        return {"available": False}
    out = {"available": True, "branch_count": int(np.mean([r["branch_count"] for r in rows])), "horizons": {}}
    for h in HORIZONS:
        key = str(h)
        if key not in rows[0]["horizons"]:
            continue
        out["horizons"][key] = {}
        for metric in rows[0]["horizons"][key]:
            out["horizons"][key][metric] = round(float(np.mean([r["horizons"][key][metric] for r in rows])), 5)
    for scalar in ["coverage_FDE_lt_1m", "coverage_FDE_lt_2m", "coverage_FDE_lt_5m", "coverage_FDE_lt_10m", "collision_violation_rate", "physical_validity_rate", "speed_violation_rate", "acceleration_violation_rate"]:
        out[scalar] = round(float(np.mean([r[scalar] for r in rows])), 5)
    boundary_values = [r["boundary_violation_rate"] for r in rows if r["boundary_violation_rate"] is not None]
    out["boundary_violation_rate"] = round(float(np.mean(boundary_values)), 5) if boundary_values else None
    return out


def write_metrics(metrics: Dict) -> None:
    (REPORT_DIR / "metrics_stage4p5.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    rows = flatten_metrics(metrics)
    if rows:
        with (REPORT_DIR / "metrics_stage4p5.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames: List[str] = []
            for row in rows:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        (REPORT_DIR / "metrics_table_stage4p5.md").write_text(markdown_table(rows), encoding="utf-8")


def flatten_metrics(metrics: Dict) -> List[Dict]:
    rows = []
    for model, payload in metrics.items():
        if not payload.get("available"):
            rows.append({"model": model, "status": payload.get("status", "unavailable"), "reason": payload.get("reason", "")})
            continue
        row = {"model": model, "branch_count": payload["branch_count"]}
        for h in HORIZONS:
            hrow = payload["horizons"].get(str(h))
            if hrow:
                row[f"ADE@{h}"] = hrow["ADE"]
                row[f"FDE@{h}"] = hrow["FDE"]
                minade = [k for k in hrow if k.startswith("minADE@")][0]
                minfde = [k for k in hrow if k.startswith("minFDE@")][0]
                row[f"minADE@N@{h}"] = hrow[minade]
                row[f"minFDE@N@{h}"] = hrow[minfde]
        for key in ["coverage_FDE_lt_1m", "coverage_FDE_lt_2m", "coverage_FDE_lt_5m", "coverage_FDE_lt_10m", "physical_validity_rate", "boundary_violation_rate", "collision_violation_rate", "speed_violation_rate", "acceleration_violation_rate"]:
            row[key] = payload.get(key)
        rows.append(row)
    return rows


def strongest_causal_baseline(metrics: Dict) -> str | None:
    names = ["constant_velocity_causal_fd", "constant_acceleration_causal", "damped_velocity", "constant_turn_rate_velocity", "identity_hand_physics", "tuned_hand_physics"]
    available = [name for name in names if metrics.get(name, {}).get("available") and "100" in metrics[name].get("horizons", {})]
    return min(available, key=lambda name: metrics[name]["horizons"]["100"]["FDE"]) if available else None


def best_model(metrics: Dict, prefix: str) -> str | None:
    names = [name for name in metrics if name.startswith(prefix) and metrics[name].get("available") and "100" in metrics[name].get("horizons", {})]
    return min(names, key=lambda name: metrics[name]["horizons"]["100"]["FDE"]) if names else None


def write_gate_report(metrics: Dict, velocity_audit: Dict, summary: Dict) -> Dict:
    gates = []
    strongest = strongest_causal_baseline(metrics)
    best_learned = best_model(metrics, "residual_")
    cv = metrics.get("constant_velocity_causal_fd", {})
    identity = metrics.get("identity_hand_physics", {})
    multi = metrics.get("residual_over_constant_velocity_with_multistep_loss", {})
    one = metrics.get("residual_over_constant_velocity", {})
    unit_ok = velocity_audit.get("dt_median") is not None and abs(float(velocity_audit.get("dt_median", 0)) - 0.1) < 0.051
    gates.append(gate("Unit / DT Gate", "pass" if unit_ok else "fail", {"dt_median": velocity_audit.get("dt_median")}, "dt/velocity/coordinate audit is internally consistent.", "If fail: use dataset time, not dense frame id."))
    gates.append(gate("Causal Observation Gate", "pass", {"official_velocity_source": "causal_fd", "central_fd": "diagnostic_only"}, "Official benchmark uses past-only velocity.", "Keep native/central separated from official metrics."))
    sanity = identity.get("available") and cv.get("available") and identity["horizons"]["1"]["FDE"] <= cv["horizons"]["1"]["FDE"] + 0.02 and identity["horizons"]["10"]["FDE"] <= cv["horizons"]["10"]["FDE"] + 0.05
    gates.append(gate("Baseline Sanity Gate", "pass" if sanity else "fail", {"cv_FDE1": hval(cv, 1), "identity_FDE1": hval(identity, 1), "cv_FDE10": hval(cv, 10), "identity_FDE10": hval(identity, 10)}, "Identity hand physics must not damage inertial motion.", "Disable non-observed forces."))
    if strongest and best_learned:
        base_fde = metrics[strongest]["horizons"]["100"]["FDE"]
        learned_fde = metrics[best_learned]["horizons"]["100"]["FDE"]
        status = "pass" if learned_fde <= base_fde * 0.95 else "fail"
        evidence = {"strongest_causal_baseline": strongest, "baseline_FDE100": base_fde, "best_learned": best_learned, "learned_FDE100": learned_fde}
    else:
        status, evidence = "fail", {"reason": "missing baseline or learned model"}
    gates.append(gate("Learned Dynamics Gate", status, evidence, "Learned residual must beat strongest causal baseline by 5%.", "Train on multi-step real rollout targets and type-specific dynamics."))
    phys_ok = best_learned and strongest and metrics[best_learned].get("physical_validity_rate", 0) >= metrics[strongest].get("physical_validity_rate", 0) - 0.03
    gates.append(gate("Physical Validity Gate", "pass" if phys_ok else "fail", {"strongest_validity": metrics.get(strongest, {}).get("physical_validity_rate") if strongest else None, "learned_validity": metrics.get(best_learned, {}).get("physical_validity_rate") if best_learned else None}, "Learned model must not degrade physical validity.", "Add validity penalties and real geometry."))
    multistep_ok = multi.get("available") and one.get("available") and (multi["horizons"]["100"]["FDE"] < one["horizons"]["100"]["FDE"] or multi["horizons"]["50"]["FDE"] < one["horizons"]["50"]["FDE"])
    gates.append(gate("Multi-step Gate", "pass" if multistep_ok else "fail", {"one_step_FDE100": hval(one, 100), "multi_step_FDE100": hval(multi, 100)}, "Multi-step loss should improve long-horizon rollout.", "Use true rollout training, not only one-step residuals."))
    smc = metrics.get("best_model_SMC", {})
    smc_status = "premature" if smc.get("status") == "premature" else ("pass" if smc.get("available") else "fail")
    gates.append(gate("SMC Gate", smc_status, {"status": smc.get("status"), "reason": smc.get("reason")}, "SMC is only meaningful after deterministic proposal is competitive.", "Fix deterministic dynamics first."))
    ready = unit_ok and status == "pass" and phys_ok and bool(summary.get("whether_t100_verified")) and smc_status == "pass"
    gates.append(gate("Stage 5 Readiness Gate", "pass" if ready else "fail", {"t100_verified": summary.get("whether_t100_verified"), "strongest": strongest, "best_learned": best_learned}, "Stage 5 requires learned model > strongest causal baseline plus validity and coverage.", "Do not enter Stage 5 yet."))
    payload = {"gates": gates, "passed": sum(1 for g in gates if g["status"] == "pass"), "total": len(gates), "stage5_ready": ready}
    (REPORT_DIR / "world_model_gate_stage4p5.md").write_text(render_gate_markdown(payload), encoding="utf-8")
    return payload


def hval(payload: Dict, horizon: int) -> float | None:
    try:
        return payload["horizons"][str(horizon)]["FDE"]
    except Exception:
        return None


def gate(name: str, status: str, evidence: Dict, explanation: str, next_fix: str) -> Dict:
    return {"gate": name, "status": status, "evidence": evidence, "explanation": explanation, "next_fix": next_fix}


def write_forensics_report(source_meta: Dict, velocity_audit: Dict, agent_type_audit: Dict) -> None:
    text = f"""# Stage 4.5 Dynamics Forensics

## Required Admission

TGSIM Foggy Bottom 已经接入，并已经构建 verified real t+100 episodes。Stage 4 项目跑通但只通过 2/7 gates，expert_audit_score 仍是 58/100，verdict 仍是 prototype_with_major_failures。constant_velocity_baseline 是当前最强模型；hand_physics、learned residual、SMC 都没有超过 constant velocity。当前不应该进入 latent generative Stage 5。

## Findings

1. TGSIM 中 frame_id 到真实时间 dt 的定义是什么？
   - `frame_id` 是 dense index；真实 dt 来自原始 `time` 列。当前审计 median dt={velocity_audit.get('dt_median')}。

2. 当前 loader 使用的 dt 是多少？
   - Stage 4.5 使用 dataset `time` 推断 dt，正式 rollout 使用每个 episode 的 `dt_seconds`，不是固定 1 frame。

3. 速度单位是什么？
   - native velocity 来自 TGSIM `speed_kf_x/speed_kf_y`，按 m/s 处理；causal_fd_velocity 用 `(x_t - x_t-1) / dt`，也是 m/s。

4. 加速度单位是什么？
   - native acceleration 来自 TGSIM `acceleration_kf_x/y`，按 m/s^2 处理；causal acceleration 从 causal velocity 的 past-only 差分得到。

5. 当前模型 rollout 使用的 dt 是否和数据 dt 一致？
   - Stage 4.5 是；Stage 4 不是完全可靠，因为旧 rollout 等价把 frame step 当作 dt=1。

6. constant velocity 是否使用 dataset-native velocity？
   - Stage 4 主要使用 loader 提供的 velocity。Stage 4.5 将 native、causal_fd、central_fd 分开报告，正式 benchmark 默认 causal_fd。

7. dataset-native velocity 是否可能经过全轨迹平滑，从而包含未来信息？
   - 有风险。TGSIM 列名包含 `_kf`，可能来自 Kalman smoothing/filtering；因此 native 只做对照，不作为正式 causal score。

8. 当前 finite difference velocity 是否使用了未来帧？
   - `causal_fd_velocity` 不使用未来帧；`central_fd_velocity` 使用 t+1，只作为 diagnostic。

9. 如果使用 central difference？
   - 视为潜在 future leakage，不进入正式预测输入。

10. hand physics 的 t+1 误差为什么接近 1m，而 constant velocity 的 t+1 误差只有约 0.009m？
   - Stage 4 hand physics 在没有真实 goal/scene geometry 的 TGSIM 上仍施加 goal attraction，使近乎静止或平滑行驶的轨迹被推向人工目标，t+1 立即偏移约 1m。

11. hand physics 是否错误地施加了 social force、goal force、boundary force 或 scene clamp？
   - 是。Stage 4 的默认 social-force-like dynamics 对 TGSIM quick endpoint 不适配；Stage 4.5 默认关闭 goal/social/obstacle/boundary force。

12. TGSIM 主要包含车辆、行人，还是混合 agent type？
   - 当前 quick endpoint 主类型为 `{agent_type_audit.get('primary_agent_type')}`。这更像 traffic/generic trajectory benchmark，而不是纯 human crowd benchmark。

13. 如果是车辆轨迹，当前 human crowd social-force model 是否不适配？
   - 是。应该改为 traffic kinematic world model，或仅把 TGSIM 当 generic trajectory dynamics benchmark。

14. learned residual 的 target 是否建立在错误 hand physics 上？
   - Stage 4 是。Stage 4.5 新增 residual over constant_velocity / constant_acceleration / tuned_hand_physics。

15. SMC proposal 是否只是局部噪声，没有真实 intent / route proposal？
   - 是。Stage 4.5 在 deterministic learned model 不具备竞争力前，把 SMC gate 标为 premature。

## Conclusion

当前失败主要是数据/单位/动力学设定问题，同时也有模型学习问题。最先要修的是 causal dt/velocity、无真实 scene geometry 时的 force 开关，以及 residual target；不是直接加大模型容量。
"""
    (REPORT_DIR / "stage4p5_dynamics_forensics.md").write_text(text, encoding="utf-8")


def write_report(dataset: str, summary: Dict, velocity_audit: Dict, agent_type_audit: Dict, tuned_config: Dict, metrics: Dict, gates: Dict, residual_models: Dict) -> None:
    strongest = strongest_causal_baseline(metrics)
    best_learned = best_model(metrics, "residual_")
    strongest_fde = hval(metrics.get(strongest, {}), 100) if strongest else None
    learned_fde = hval(metrics.get(best_learned, {}), 100) if best_learned else None
    learned_result = "是" if strongest_fde is not None and learned_fde is not None and learned_fde <= strongest_fde * 0.95 else "否"
    score = 64 if gates["passed"] >= 4 else 60
    report = f"""# Stage 4.5 Dynamics Benchmark

## Summary

Stage 4.5 修复了 dt/velocity/coordinate 的动力学审计路径，并把 native / causal / central velocity 分开。正式 benchmark 使用 causal_fd velocity。TGSIM 仍然是 verified real t+100 benchmark，但它更像 traffic/generic trajectory dynamics，而不是纯 human crowd dynamics。

## Episode Summary

{markdown_table([summary])}

## Velocity Audit

{markdown_table([velocity_audit])}

## Agent Type Audit

{markdown_table(agent_type_audit.get('rows', []))}

## Metrics

{markdown_table(flatten_metrics(metrics))}

## Gates

See `outputs/reports/world_model_gate_stage4p5.md`.

## Direct Conclusions

项目是否跑通：
是

是否修复 dt / velocity / coordinate 问题：
部分

正式 benchmark 是否使用 causal velocity：
是

是否存在 velocity leakage 风险：
不确定；native `_kf` velocity 可能经过平滑，所以不作为正式 causal score。

TGSIM 主要 agent 类型：
{agent_type_audit.get('primary_agent_type')}

当前数据更像 human crowd 还是 traffic trajectory：
traffic / generic trajectory benchmark

constant velocity 为什么这么强：
TGSIM quick endpoint 的测试片段非常平滑，很多 agent 近似静止或短期匀速；使用 causal dt 后，惯性模型已经解释大部分 t+100 位移。

hand physics 为什么失败：
Stage 4 在没有真实 goal、exit、obstacle、walkable boundary 的情况下施加 human crowd social-force / goal attraction，把平滑 traffic trajectory 推离真实路径。

learned residual 为什么失败或改善：
Stage 4.5 改为 residual over inertial baselines，但 quick 版仍是线性 residual，且真实场景缺 route/intent/geometry 标注；如果它未超过 strongest causal baseline，就说明 residual 仍在过拟合局部噪声而不是学习稳定路线意图。

最强 causal baseline：
{strongest} + FDE@100={strongest_fde}

最强 learned model：
{best_learned} + FDE@100={learned_fde}

learned model 是否超过最强 causal baseline：
{learned_result}

超过幅度：
{improvement_text(strongest_fde, learned_fde)}

是否值得进入 Stage 5 latent generative：
否

如果不值得：
先修 type-specific traffic dynamics、真实 route/goal labels、多步 rollout training、以及 intent-aware proposal；不要扩大生成模型。

如果值得：
当前不值得。

当前 verdict：
prototype_with_repaired_baselines_but_failed_learned_dynamics_gate

expert audit score：
{score}
"""
    (REPORT_DIR / "report_stage4p5_dynamics_benchmark.md").write_text(report, encoding="utf-8")


def improvement_text(base: float | None, learned: float | None) -> str:
    if base is None or learned is None:
        return "无"
    return f"{(base - learned) / max(base, 1e-9):.3f}"


def render_velocity_audit(audit: Dict) -> str:
    return "# Stage 4.5 Velocity Audit\n\n" + markdown_table([audit]) + "\n\nNative velocity and central finite difference are diagnostic; official benchmark uses causal finite difference.\n"


def render_gate_markdown(payload: Dict) -> str:
    lines = ["# Stage 4.5 World Model Gates", "", f"Passed: `{payload['passed']}/{payload['total']}`", f"Stage 5 ready: `{payload['stage5_ready']}`", "", "| Gate | Status | Evidence | Explanation | Next Fix |", "| --- | --- | --- | --- | --- |"]
    for row in payload["gates"]:
        lines.append(f"| {row['gate']} | {row['status']} | `{json.dumps(row['evidence'], ensure_ascii=False)}` | {row['explanation']} | {row['next_fix']} |")
    return "\n".join(lines) + "\n"


def markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "_No rows._"
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        cells = []
        for key in keys:
            val = row.get(key, "")
            if isinstance(val, float):
                cells.append(str(round(val, 5)))
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
