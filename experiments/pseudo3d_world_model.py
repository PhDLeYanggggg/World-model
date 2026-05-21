#!/usr/bin/env python3
"""Minimum viable pseudo-3D human collision world model.

The goal of this experiment is not a stronger pixel transition head. It builds
a camera-aware 2.5D state-space model:

- image coordinates are observations;
- ground/world coordinates are the latent physical state;
- humans are vertical cylinders with uncertain body parameters;
- projection and observation likelihood connect world state back to pixels;
- transition, collision, density, boundaries, and scene constraints run in
  pseudo-metric world coordinates.

The AerialMPT release does not include camera intrinsics, extrinsics, or control
points. This script therefore uses a weak ground-plane homography based on the
dataset GSD range and reports the calibration uncertainty explicitly.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import load
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

from human_collision_world_model import (
    DATA_ROOT,
    FPS,
    HORIZON,
    PERSON_MASS_KG,
    SocialNonlinearModel,
    choose_holdout_scene,
    load_aerialmpt,
    smc_rollout,
    state_at_frame,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "outputs" / "pseudo3d_world_model"
LEGACY_PIXEL_MODEL = ROOT / "experiments" / "outputs" / "human_collision_world_model" / "nonlinear_social_model.joblib"

DT = 1.0 / FPS
MPP_MEAN = 0.105
MPP_LOW = 0.08
MPP_HIGH = 0.13
DEFAULT_BODY_RADIUS_M = 0.30
DEFAULT_BODY_HEIGHT_M = 1.70
DEFAULT_MASS_KG = 70.0
OBS_SIGMA_PX = 4.0
MAX_PEDESTRIAN_SPEED_MPS = 2.2


@dataclass
class Calibration:
    H: np.ndarray
    H_inv: np.ndarray
    meter_per_pixel: float
    scale_uncertainty: float
    image_width: int
    image_height: int
    quality: Dict


@dataclass
class SceneGeometry:
    walkable_polygon: List[Tuple[float, float]]
    obstacle_polygons: List[List[Tuple[float, float]]]
    goal_regions: List[Dict]
    soft_boundary_margin_m: float
    density_cell_m: float
    source: str


def main() -> None:
    seed_everything(31)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_tracks = load_aerialmpt(DATA_ROOT)
    scene_name = choose_holdout_scene(all_tracks[all_tracks["split"] == "test"])
    scene_tracks = all_tracks[(all_tracks["split"] == "test") & (all_tracks["scene"] == scene_name)].copy()
    image_path = first_scene_image_path(scene_tracks)
    with Image.open(image_path) as image:
        image_size = image.size

    calibration = setup_weak_homography(image_size)
    scene_geometry = build_scene_geometry(scene_name, calibration)
    start_frame = choose_pseudo3d_start_frame(scene_tracks, warmup_frames=4)
    max_observed_horizon = int(scene_tracks["frame"].max() - start_frame)
    requested_horizons = [1, 5, 10, 20, 50, 100]
    eval_horizons = [h for h in requested_horizons if h <= max_observed_horizon]
    if max_observed_horizon not in eval_horizons and max_observed_horizon > 0:
        eval_horizons.append(max_observed_horizon)
    eval_horizons = sorted(set(eval_horizons))

    initial_observation = observations_at_frame(scene_tracks, start_frame)
    active_ids = choose_active_ids(initial_observation, max_agents=18)
    initial_world_state = initialize_world_state_from_observation(
        scene_tracks=scene_tracks,
        frame=start_frame,
        ids=active_ids,
        calibration=calibration,
        scene_geometry=scene_geometry,
    )

    versions = {
        "Version A: homography world only": {
            "use_scene": False,
            "use_goal": False,
            "use_observation": False,
            "particles": 32,
            "proposal_count": 3,
            "seed": 101,
        },
        "Version B: world + scene geometry": {
            "use_scene": True,
            "use_goal": False,
            "use_observation": False,
            "particles": 32,
            "proposal_count": 3,
            "seed": 202,
        },
        "Version C: world + latent goal": {
            "use_scene": False,
            "use_goal": True,
            "use_observation": False,
            "particles": 40,
            "proposal_count": 3,
            "seed": 303,
        },
        "Version D: full pseudo-3D world model": {
            "use_scene": True,
            "use_goal": True,
            "use_observation": True,
            "particles": 40,
            "proposal_count": 3,
            "seed": 404,
        },
    }

    world_runs = {}
    for name, config in versions.items():
        print(f"running {name}", flush=True)
        t0 = time.time()
        world_runs[name] = run_world_smc(
            initial_state=initial_world_state,
            scene_tracks=scene_tracks,
            start_frame=start_frame,
            calibration=calibration,
            scene_geometry=scene_geometry,
            horizon=HORIZON,
            eval_horizons=eval_horizons,
            config={**config, "label": name},
        )
        world_runs[name]["runtime_s"] = round(time.time() - t0, 3)

    teacher_forced = run_teacher_forced_filter(
        initial_state=initial_world_state,
        scene_tracks=scene_tracks,
        start_frame=start_frame,
        calibration=calibration,
        scene_geometry=scene_geometry,
        horizon=max_observed_horizon,
        eval_horizons=eval_horizons,
        config={**versions["Version D: full pseudo-3D world model"], "particles": 40, "seed": 505, "label": "teacher-forced pseudo-3D filter"},
    )

    baselines = run_baselines(
        scene_tracks=scene_tracks,
        start_frame=start_frame,
        active_ids=active_ids,
        calibration=calibration,
        eval_horizons=eval_horizons,
        max_observed_horizon=max_observed_horizon,
    )

    full_run = world_runs["Version D: full pseudo-3D world model"]
    terminal_clusters = cluster_terminal_world_states(
        full_run["particles"],
        calibration=calibration,
        scene_geometry=scene_geometry,
        initial_centroid=centroid_world(initial_world_state),
    )
    horizon_diagnostics = build_horizon_diagnostics(
        free_run=full_run,
        teacher_forced=teacher_forced,
        scene_tracks=scene_tracks,
        start_frame=start_frame,
        active_ids=active_ids,
        calibration=calibration,
        eval_horizons=eval_horizons,
    )
    ablation_table = build_ablation_table(
        baselines=baselines,
        world_runs=world_runs,
        scene_tracks=scene_tracks,
        start_frame=start_frame,
        active_ids=active_ids,
        calibration=calibration,
        eval_horizon=max_observed_horizon,
    )

    artifacts = render_visualizations(
        image_path=image_path,
        initial_state=initial_world_state,
        scene_tracks=scene_tracks,
        start_frame=start_frame,
        active_ids=active_ids,
        calibration=calibration,
        scene_geometry=scene_geometry,
        free_run=full_run,
        teacher_forced=teacher_forced,
        terminal_clusters=terminal_clusters,
        eval_horizon=max_observed_horizon,
    )

    summary = {
        "dataset": {
            "source": "AerialMPT official DLR dataset",
            "scene": scene_name,
            "frames_available": int(scene_tracks["frame"].nunique()),
            "frame_min": int(scene_tracks["frame"].min()),
            "frame_max": int(scene_tracks["frame"].max()),
            "start_frame": int(start_frame),
            "max_observed_horizon": int(max_observed_horizon),
            "requested_free_run_horizon": HORIZON,
            "t100_ground_truth_available": bool(start_frame + HORIZON <= scene_tracks["frame"].max()),
            "note": "t+100 is free-run only for this scene because the sequence has fewer than 101 future frames.",
        },
        "calibration_quality": calibration.quality,
        "coordinate_systems": {
            "image": {"axes": "[u, v]", "unit": "pixel", "origin": "top-left"},
            "world": {
                "axes": "[X, Y, Z]",
                "unit": "meter",
                "ground_plane": "Z = ground_height(X,Y) = 0",
                "origin": "image center mapped through weak homography",
            },
            "body": {
                "geometry": "vertical cylinder/capsule approximation",
                "default_radius_m": DEFAULT_BODY_RADIUS_M,
                "default_height_m": DEFAULT_BODY_HEIGHT_M,
                "mass_kg": DEFAULT_MASS_KG,
                "latent_parameters": ["body_radius_m", "body_height_m", "mass_kg", "desired_speed_mps", "goal_world"],
            },
        },
        "scene_geometry": summarize_scene_geometry(scene_geometry),
        "models": {
            "free_run_versions": {name: summarize_world_run(run) for name, run in world_runs.items()},
            "teacher_forced_filtering": summarize_world_run(teacher_forced),
            "baselines": baselines,
        },
        "horizon_diagnostics": horizon_diagnostics,
        "ablation_table": ablation_table,
        "terminal_world_clusters": terminal_clusters,
        "observability": observability_report(),
        "artifacts": artifacts,
    }

    summary_path = OUT_DIR / "summary.json"
    report_path = OUT_DIR / "REPORT.md"
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
    report_path.write_text(build_report(summary), encoding="utf-8")

    print(json.dumps({"summary": str(summary_path), "report": str(report_path), **artifacts}, indent=2))
    print(json.dumps(key_console_summary(summary), indent=2))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def setup_weak_homography(image_size: Tuple[int, int]) -> Calibration:
    width, height = image_size
    cx, cy = width / 2.0, height / 2.0
    s = MPP_MEAN
    H = np.array([[1.0 / s, 0.0, cx], [0.0, 1.0 / s, cy], [0.0, 0.0, 1.0]], dtype=float)
    H_inv = np.linalg.inv(H)
    scale_uncertainty = (MPP_HIGH - MPP_LOW) / (2.0 * MPP_MEAN)
    quality = {
        "has_real_camera_calibration": False,
        "has_homography_control_points": False,
        "estimated_meter_per_pixel": MPP_MEAN,
        "meter_per_pixel_range_from_dataset_gsd": [MPP_LOW, MPP_HIGH],
        "scale_uncertainty": round(scale_uncertainty, 4),
        "projection_error_px": None,
        "algebraic_roundtrip_error_px": 0.0,
        "assumption": "Weak homography from AerialMPT GSD only; no true camera intrinsics/extrinsics or control points were provided.",
        "H_ground_to_image": H.tolist(),
        "H_inv_image_to_ground": H_inv.tolist(),
    }
    return Calibration(H=H, H_inv=H_inv, meter_per_pixel=s, scale_uncertainty=scale_uncertainty, image_width=width, image_height=height, quality=quality)


def project_world_to_image(P_world: Sequence[float], camera_params: Optional[Dict] = None) -> Tuple[float, float]:
    if camera_params and all(key in camera_params for key in ["K", "R", "t"]):
        K = np.asarray(camera_params["K"], dtype=float)
        R = np.asarray(camera_params["R"], dtype=float)
        t = np.asarray(camera_params["t"], dtype=float).reshape(3, 1)
        P = np.asarray(P_world[:3], dtype=float).reshape(3, 1)
        p = K @ (R @ P + t)
        return float(p[0, 0] / p[2, 0]), float(p[1, 0] / p[2, 0])
    if camera_params and "H" in camera_params:
        return ground_to_image_homography(P_world[0], P_world[1], np.asarray(camera_params["H"], dtype=float))
    raise ValueError("camera_params must contain K/R/t or H")


def backproject_image_to_ground(u: float, v: float, camera_params: Dict) -> Tuple[float, float, float]:
    if all(key in camera_params for key in ["K", "R", "t"]):
        K = np.asarray(camera_params["K"], dtype=float)
        R = np.asarray(camera_params["R"], dtype=float)
        t = np.asarray(camera_params["t"], dtype=float).reshape(3, 1)
        ray_camera = np.linalg.inv(K) @ np.array([[u], [v], [1.0]])
        ray_world = R.T @ ray_camera
        camera_center = -R.T @ t
        if abs(ray_world[2, 0]) < 1e-9:
            raise ValueError("Image ray is parallel to the ground plane")
        alpha = -camera_center[2, 0] / ray_world[2, 0]
        point = camera_center + alpha * ray_world
        return float(point[0, 0]), float(point[1, 0]), 0.0
    if "H_inv" in camera_params:
        x, y = image_to_ground_homography(u, v, np.asarray(camera_params["H_inv"], dtype=float))
        return x, y, 0.0
    raise ValueError("camera_params must contain K/R/t or H_inv")


def image_to_ground_homography(u: float, v: float, H_inv: np.ndarray) -> Tuple[float, float]:
    p = H_inv @ np.array([u, v, 1.0], dtype=float)
    return float(p[0] / p[2]), float(p[1] / p[2])


def ground_to_image_homography(X: float, Y: float, H: np.ndarray) -> Tuple[float, float]:
    p = H @ np.array([X, Y, 1.0], dtype=float)
    return float(p[0] / p[2]), float(p[1] / p[2])


def build_scene_geometry(scene_name: str, calibration: Calibration) -> SceneGeometry:
    width, height = calibration.image_width, calibration.image_height
    walkable_px = [(0, 0), (width, 0), (width, height), (0, height)]
    obstacles_px: List[List[Tuple[float, float]]] = []

    if scene_name.startswith("bauma"):
        obstacles_px = [
            [(0, 86), (158, 86), (158, 246), (0, 246)],
            [(164, 312), (323, 312), (323, 431), (164, 431)],
            [(436, 305), (612, 305), (612, 522), (436, 522)],
            [(104, 456), (208, 456), (208, 552), (104, 552)],
        ]

    goal_regions_px = [
        {"name": "west_exit", "center_px": (16, height * 0.52), "heading": math.pi},
        {"name": "east_exit", "center_px": (width - 16, height * 0.50), "heading": 0.0},
        {"name": "north_exit", "center_px": (width * 0.50, 16), "heading": -math.pi / 2},
        {"name": "south_exit", "center_px": (width * 0.50, height - 16), "heading": math.pi / 2},
    ]
    goal_regions = []
    for region in goal_regions_px:
        x, y = image_to_ground_homography(region["center_px"][0], region["center_px"][1], calibration.H_inv)
        goal_regions.append({"name": region["name"], "center": [x, y], "heading": region["heading"]})

    return SceneGeometry(
        walkable_polygon=[pixel_to_world_point(point, calibration) for point in walkable_px],
        obstacle_polygons=[[pixel_to_world_point(point, calibration) for point in poly] for poly in obstacles_px],
        goal_regions=goal_regions,
        soft_boundary_margin_m=1.2,
        density_cell_m=2.0,
        source="manual rough polygons in image space projected to weak ground plane",
    )


def pixel_to_world_point(point: Tuple[float, float], calibration: Calibration) -> Tuple[float, float]:
    return image_to_ground_homography(point[0], point[1], calibration.H_inv)


def summarize_scene_geometry(scene: SceneGeometry) -> Dict:
    return {
        "walkable_area_polygon": round_polygon(scene.walkable_polygon),
        "obstacle_polygon_count": len(scene.obstacle_polygons),
        "obstacle_polygons": [round_polygon(poly) for poly in scene.obstacle_polygons],
        "goal_regions": [
            {"name": region["name"], "center": [round(region["center"][0], 3), round(region["center"][1], 3)], "heading": round(region["heading"], 3)}
            for region in scene.goal_regions
        ],
        "soft_boundary_margin_m": scene.soft_boundary_margin_m,
        "density_cell_m": scene.density_cell_m,
        "source": scene.source,
    }


def round_polygon(poly: Sequence[Tuple[float, float]]) -> List[List[float]]:
    return [[round(float(x), 3), round(float(y), 3)] for x, y in poly]


def choose_pseudo3d_start_frame(scene_tracks: pd.DataFrame, warmup_frames: int) -> int:
    frames = sorted(int(frame) for frame in scene_tracks["frame"].unique())
    return frames[min(len(frames) - 1, warmup_frames - 1)]


def first_scene_image_path(scene_tracks: pd.DataFrame) -> Path:
    row = scene_tracks.sort_values("frame").iloc[0]
    scene_dir = DATA_ROOT / row["split"] / row["scene"]
    name = row.get("image_name")
    if isinstance(name, str):
        path = scene_dir / f"{name}.png"
        if path.exists():
            return path
    images = sorted(scene_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(scene_dir)
    return images[0]


def observations_at_frame(scene_tracks: pd.DataFrame, frame: int, ids: Optional[Iterable[int]] = None) -> List[Dict]:
    rows = scene_tracks[scene_tracks["frame"] == frame].copy()
    if ids is not None:
        ids_set = {int(item) for item in ids}
        rows = rows[rows["track_id"].isin(ids_set)]
    observations = []
    for _, row in rows.iterrows():
        observations.append(
            {
                "id": int(row["track_id"]),
                "u": float(row["x"] + row["w"] * 0.5),
                "v": float(row["y"] + row["h"] * 0.5),
                "bbox": [float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"])],
                "confidence": 1.0,
            }
        )
    return observations


def choose_active_ids(observations: List[Dict], max_agents: int) -> List[int]:
    if len(observations) <= max_agents:
        return [obs["id"] for obs in observations]
    points = np.array([[obs["u"], obs["v"]] for obs in observations], dtype=float)
    center = points.mean(axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    chosen = np.argsort(distances)[:max_agents]
    return [observations[int(index)]["id"] for index in chosen]


def initialize_world_state_from_observation(
    scene_tracks: pd.DataFrame,
    frame: int,
    ids: List[int],
    calibration: Calibration,
    scene_geometry: SceneGeometry,
) -> List[Dict]:
    current = {obs["id"]: obs for obs in observations_at_frame(scene_tracks, frame, ids)}
    previous = {obs["id"]: obs for obs in observations_at_frame(scene_tracks, frame - 1, ids)}
    group_ids = assign_initial_groups(current, calibration)
    state = []

    for agent_id in ids:
        if agent_id not in current:
            continue
        obs = current[agent_id]
        X, Y = image_to_ground_homography(obs["u"], obs["v"], calibration.H_inv)
        Vx, Vy = 0.0, 0.0
        if agent_id in previous:
            prev_x, prev_y = image_to_ground_homography(previous[agent_id]["u"], previous[agent_id]["v"], calibration.H_inv)
            Vx = (X - prev_x) / DT
            Vy = (Y - prev_y) / DT
        speed = math.hypot(Vx, Vy)
        desired_speed = float(np.clip(speed if speed > 0.2 else 1.0, 0.45, 1.65))
        goal = infer_initial_goal((X, Y), (Vx, Vy), scene_geometry)
        direction = unit_vector((goal[0] - X, goal[1] - Y))
        state.append(
            {
                "id": int(agent_id),
                "X": X,
                "Y": Y,
                "Z": 0.0,
                "Vx": Vx,
                "Vy": Vy,
                "Vz": 0.0,
                "Ax": 0.0,
                "Ay": 0.0,
                "body_radius_m": DEFAULT_BODY_RADIUS_M,
                "body_height_m": DEFAULT_BODY_HEIGHT_M,
                "mass_kg": DEFAULT_MASS_KG,
                "desired_speed_mps": desired_speed,
                "desired_direction_world": [direction[0], direction[1]],
                "goal_latent": [goal[0], goal[1]],
                "group_id_latent": int(group_ids.get(agent_id, agent_id)),
                "uncertainty_covariance": [[0.25, 0.0], [0.0, 0.25]],
            }
        )
    return state


def assign_initial_groups(observations: Dict[int, Dict], calibration: Calibration) -> Dict[int, int]:
    ids = list(observations.keys())
    points = {}
    for agent_id, obs in observations.items():
        points[agent_id] = image_to_ground_homography(obs["u"], obs["v"], calibration.H_inv)
    group = {agent_id: agent_id for agent_id in ids}

    def find(x: int) -> int:
        while group[x] != x:
            group[x] = group[group[x]]
            x = group[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            group[rb] = ra

    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if distance(points[a], points[b]) < 1.4:
                union(a, b)
    return {agent_id: find(agent_id) for agent_id in ids}


def infer_initial_goal(position: Tuple[float, float], velocity: Tuple[float, float], scene: SceneGeometry) -> Tuple[float, float]:
    speed = math.hypot(velocity[0], velocity[1])
    if speed > 0.15:
        heading = (velocity[0] / speed, velocity[1] / speed)
        projected = (position[0] + heading[0] * 12.0, position[1] + heading[1] * 12.0)
        return clamp_to_walkable(projected, scene.walkable_polygon)

    best = min(scene.goal_regions, key=lambda region: distance(position, region["center"]))
    return float(best["center"][0]), float(best["center"][1])


def run_world_smc(
    initial_state: List[Dict],
    scene_tracks: pd.DataFrame,
    start_frame: int,
    calibration: Calibration,
    scene_geometry: SceneGeometry,
    horizon: int,
    eval_horizons: List[int],
    config: Dict,
) -> Dict:
    rng = np.random.default_rng(config["seed"])
    particle_count = int(config["particles"])
    proposal_count = int(config["proposal_count"])
    max_eval_horizon = max(eval_horizons) if eval_horizons else 0
    keep_steps = {0, horizon, *eval_horizons, *range(0, max_eval_horizon + 1), *range(0, horizon + 1, 5)}
    particles = [
        {
            "world_state": jitter_initial_state(initial_state, rng),
            "latent_goals": {agent["id"]: list(agent["goal_latent"]) for agent in initial_state},
            "latent_body_params": {
                agent["id"]: {
                    "body_radius_m": agent["body_radius_m"],
                    "body_height_m": agent["body_height_m"],
                    "mass_kg": agent["mass_kg"],
                }
                for agent in initial_state
            },
            "log_weight": 0.0,
            "history": {0: clone_world_state(initial_state)},
            "diagnostics": [],
        }
        for _ in range(particle_count)
    ]
    ess_history = []

    for step in range(1, horizon + 1):
        proposed_particles = []
        observation = None
        if config.get("use_observation") and start_frame + step <= scene_tracks["frame"].max():
            observation = observations_at_frame(scene_tracks, start_frame + step, [agent["id"] for agent in initial_state])

        for particle in particles:
            for _ in range(proposal_count):
                proposal, transition_logp, log_q = sample_world_transition(
                    particle=particle,
                    scene_geometry=scene_geometry,
                    use_scene=bool(config.get("use_scene")),
                    use_goal=bool(config.get("use_goal")),
                    rng=rng,
                )
                corrected, projection_info = project_physical_constraints(proposal, scene_geometry, bool(config.get("use_scene")))
                log_weight = particle["log_weight"] + transition_logp - log_q
                log_weight -= 4.5 * projection_info["projection_cost_m"]
                log_weight -= 2.8 * projection_info["scene_violation_cost_m"]
                log_weight -= speed_prior_penalty(corrected)
                if observation:
                    log_weight += observation_loglikelihood(observation, corrected, calibration)

                history = dict(particle["history"])
                if step in keep_steps:
                    history[step] = clone_world_state(corrected)
                proposed_particles.append(
                    {
                        "world_state": corrected,
                        "latent_goals": {agent["id"]: list(agent["goal_latent"]) for agent in corrected},
                        "latent_body_params": {
                            agent["id"]: {
                                "body_radius_m": agent["body_radius_m"],
                                "body_height_m": agent["body_height_m"],
                                "mass_kg": agent["mass_kg"],
                            }
                            for agent in corrected
                        },
                        "log_weight": log_weight,
                        "history": history,
                        "diagnostics": [projection_info],
                    }
                )

        particles = normalize_world_particles(proposed_particles)
        ess = effective_sample_size(particles)
        ess_history.append(float(ess))
        if ess < particle_count * 0.55:
            particles = rejuvenate_particles(systematic_resample_world(particles, particle_count, rng), rng)
        else:
            particles = particles[:particle_count]
        particles = normalize_world_particles(particles)
        if step == 1 or step % 25 == 0 or step == horizon:
            print(f"  {config.get('label', 'world-smc')} step={step:03d}/{horizon} ESS={ess:.2f}", flush=True)

    return {
        "particles": particles,
        "config": {
            "horizon": horizon,
            "particles": particle_count,
            "proposal_count": proposal_count,
            "use_scene": bool(config.get("use_scene")),
            "use_goal": bool(config.get("use_goal")),
            "use_observation": bool(config.get("use_observation")),
            "mean_ess": round(float(np.mean(ess_history)), 3),
            "min_ess": round(float(np.min(ess_history)), 3),
        },
        "centroid_trajectory": weighted_centroid_trajectory(particles, horizon),
        "eval_horizons": eval_horizons,
    }


def run_teacher_forced_filter(
    initial_state: List[Dict],
    scene_tracks: pd.DataFrame,
    start_frame: int,
    calibration: Calibration,
    scene_geometry: SceneGeometry,
    horizon: int,
    eval_horizons: List[int],
    config: Dict,
) -> Dict:
    config = {**config, "use_observation": True, "horizon": horizon}
    return run_world_smc(initial_state, scene_tracks, start_frame, calibration, scene_geometry, horizon, eval_horizons, config)


def jitter_initial_state(state: List[Dict], rng: np.random.Generator) -> List[Dict]:
    jittered = []
    for agent in state:
        item = dict(agent)
        radius = float(np.clip(rng.normal(DEFAULT_BODY_RADIUS_M, 0.045), 0.23, 0.42))
        height = float(np.clip(rng.normal(DEFAULT_BODY_HEIGHT_M, 0.12), 1.45, 1.95))
        mass = float(np.clip(rng.normal(DEFAULT_MASS_KG, 8.0), 50.0, 90.0))
        item["X"] += float(rng.normal(0.0, 0.18))
        item["Y"] += float(rng.normal(0.0, 0.18))
        item["body_radius_m"] = radius
        item["body_height_m"] = height
        item["mass_kg"] = mass
        item["desired_speed_mps"] = float(np.clip(rng.normal(item["desired_speed_mps"], 0.18), 0.35, 1.85))
        goal = np.array(item["goal_latent"], dtype=float) + rng.normal(0.0, 0.8, size=2)
        item["goal_latent"] = [float(goal[0]), float(goal[1])]
        jittered.append(item)
    return jittered


def sample_world_transition(
    particle: Dict,
    scene_geometry: SceneGeometry,
    use_scene: bool,
    use_goal: bool,
    rng: np.random.Generator,
) -> Tuple[List[Dict], float, float]:
    state = particle["world_state"]
    accelerations = compute_world_accelerations(state, scene_geometry, use_scene, use_goal)
    process_std = 0.18
    proposal_logp = 0.0
    next_state = []

    for agent, accel in zip(state, accelerations):
        ax = accel[0] + float(rng.normal(0.0, process_std))
        ay = accel[1] + float(rng.normal(0.0, process_std))
        z = np.array([ax - accel[0], ay - accel[1]], dtype=float) / process_std
        proposal_logp += -0.5 * float(z @ z)

        vx = agent["Vx"] + DT * ax
        vy = agent["Vy"] + DT * ay
        speed = math.hypot(vx, vy)
        if speed > MAX_PEDESTRIAN_SPEED_MPS:
            vx *= MAX_PEDESTRIAN_SPEED_MPS / speed
            vy *= MAX_PEDESTRIAN_SPEED_MPS / speed
        x = agent["X"] + DT * vx
        y = agent["Y"] + DT * vy
        direction = unit_vector((agent["goal_latent"][0] - x, agent["goal_latent"][1] - y))
        item = dict(agent)
        item.update(
            {
                "X": float(x),
                "Y": float(y),
                "Z": 0.0,
                "Vx": float(vx),
                "Vy": float(vy),
                "Vz": 0.0,
                "Ax": float(ax),
                "Ay": float(ay),
                "desired_direction_world": [direction[0], direction[1]],
            }
        )
        next_state.append(item)

    transition_logp = proposal_logp
    return next_state, transition_logp, proposal_logp


def compute_world_accelerations(state: List[Dict], scene: SceneGeometry, use_scene: bool, use_goal: bool) -> List[Tuple[float, float]]:
    accelerations = []
    for i, agent in enumerate(state):
        ax, ay = -0.12 * agent["Vx"], -0.12 * agent["Vy"]

        if use_goal:
            gx, gy = agent["goal_latent"]
            goal_dir = unit_vector((gx - agent["X"], gy - agent["Y"]))
            desired_v = np.array(goal_dir) * agent["desired_speed_mps"]
            ax += 0.75 * (desired_v[0] - agent["Vx"])
            ay += 0.75 * (desired_v[1] - agent["Vy"])

        social = social_repulsion(agent, state, i)
        ax += social[0]
        ay += social[1]

        group = group_cohesion(agent, state)
        ax += group[0]
        ay += group[1]

        if use_scene:
            obstacle = obstacle_repulsion((agent["X"], agent["Y"]), scene)
            boundary = boundary_repulsion((agent["X"], agent["Y"]), scene.walkable_polygon, scene.soft_boundary_margin_m)
            ax += obstacle[0] + boundary[0]
            ay += obstacle[1] + boundary[1]

        accelerations.append((float(ax), float(ay)))
    return accelerations


def social_repulsion(agent: Dict, state: List[Dict], index: int) -> Tuple[float, float]:
    force = np.zeros(2, dtype=float)
    pos = np.array([agent["X"], agent["Y"]], dtype=float)
    vel = np.array([agent["Vx"], agent["Vy"]], dtype=float)
    for j, other in enumerate(state):
        if j == index:
            continue
        other_pos = np.array([other["X"], other["Y"]], dtype=float)
        diff = pos - other_pos
        dist = max(1e-4, float(np.linalg.norm(diff)))
        combined = agent["body_radius_m"] + other["body_radius_m"] + 0.16
        gap = dist - combined
        if gap > 2.2:
            continue
        normal = diff / dist
        closing = -float(np.dot(vel - np.array([other["Vx"], other["Vy"]], dtype=float), normal))
        strength = 0.55 * math.exp(-max(gap, -0.5) / 0.75)
        if closing > 0:
            strength += 0.18 * min(2.0, closing)
        force += normal * strength
    return float(force[0]), float(force[1])


def group_cohesion(agent: Dict, state: List[Dict]) -> Tuple[float, float]:
    group_members = [other for other in state if other["id"] != agent["id"] and other["group_id_latent"] == agent["group_id_latent"]]
    if not group_members:
        return 0.0, 0.0
    points = np.array([[other["X"], other["Y"]] for other in group_members], dtype=float)
    center = points.mean(axis=0)
    vec = center - np.array([agent["X"], agent["Y"]], dtype=float)
    dist = float(np.linalg.norm(vec))
    if dist < 1.2 or dist > 4.0:
        return 0.0, 0.0
    cohesion = 0.05 * vec / max(1e-6, dist)
    return float(cohesion[0]), float(cohesion[1])


def obstacle_repulsion(position: Tuple[float, float], scene: SceneGeometry) -> Tuple[float, float]:
    force = np.zeros(2, dtype=float)
    for poly in scene.obstacle_polygons:
        nearest = nearest_point_on_polygon(position, poly)
        vec = np.array(position, dtype=float) - np.array(nearest, dtype=float)
        dist = max(1e-6, float(np.linalg.norm(vec)))
        inside = point_in_polygon(position, poly)
        if inside:
            force += vec / dist * 1.8
        elif dist < 1.4:
            force += vec / dist * (0.8 * (1.4 - dist))
    return float(force[0]), float(force[1])


def boundary_repulsion(position: Tuple[float, float], walkable: List[Tuple[float, float]], margin: float) -> Tuple[float, float]:
    min_x, min_y, max_x, max_y = polygon_bounds(walkable)
    x, y = position
    force = np.zeros(2, dtype=float)
    distances = [(x - min_x, (1.0, 0.0)), (max_x - x, (-1.0, 0.0)), (y - min_y, (0.0, 1.0)), (max_y - y, (0.0, -1.0))]
    for dist, normal in distances:
        if dist < margin:
            force += np.array(normal) * (0.9 * (margin - dist))
    return float(force[0]), float(force[1])


def project_physical_constraints(state: List[Dict], scene: SceneGeometry, use_scene: bool) -> Tuple[List[Dict], Dict]:
    corrected, collision_info = project_collisions_world(state)
    scene_info = project_scene_constraints_world(corrected, scene) if use_scene else empty_scene_projection_info(corrected, scene)
    info = {
        **collision_info,
        **scene_info,
        "projection_cost_m": collision_info["projection_cost_m"] + scene_info["scene_projection_cost_m"],
    }
    return scene_info["state"], info


def project_collisions_world(state: List[Dict], iterations: int = 5, comfort_margin: float = 0.05) -> Tuple[List[Dict], Dict]:
    corrected = [dict(agent) for agent in state]
    projection_cost = 0.0
    num_collisions = 0
    max_penetration = 0.0

    for _ in range(iterations):
        moved = False
        for i, a in enumerate(corrected):
            for b in corrected[i + 1 :]:
                dx = b["X"] - a["X"]
                dy = b["Y"] - a["Y"]
                dist = math.hypot(dx, dy)
                min_dist = a["body_radius_m"] + b["body_radius_m"] + comfort_margin
                penetration = min_dist - dist
                if penetration <= 0:
                    continue
                moved = True
                num_collisions += 1
                max_penetration = max(max_penetration, penetration)
                if dist < 1e-6:
                    dx, dy, dist = 1.0, 0.0, 1.0
                nx, ny = dx / dist, dy / dist
                inv_mass_a = 1.0 / max(1.0, a["mass_kg"])
                inv_mass_b = 1.0 / max(1.0, b["mass_kg"])
                total_inv = inv_mass_a + inv_mass_b
                push_a = penetration * inv_mass_a / total_inv
                push_b = penetration * inv_mass_b / total_inv
                a["X"] -= nx * push_a
                a["Y"] -= ny * push_a
                b["X"] += nx * push_b
                b["Y"] += ny * push_b
                projection_cost += penetration
        if not moved:
            break

    return corrected, {
        "collision_projection_cost_m": float(projection_cost),
        "num_collisions": int(num_collisions),
        "min_gap_m": float(min_gap_world(corrected)),
        "max_penetration_m": float(max_penetration),
        "projection_cost_m": float(projection_cost),
    }


def project_scene_constraints_world(state: List[Dict], scene: SceneGeometry) -> Dict:
    corrected = [dict(agent) for agent in state]
    min_x, min_y, max_x, max_y = polygon_bounds(scene.walkable_polygon)
    projection_cost = 0.0
    boundary_violation = 0.0
    obstacle_violations = 0

    for agent in corrected:
        x, y = agent["X"], agent["Y"]
        clamped_x = float(np.clip(x, min_x + agent["body_radius_m"], max_x - agent["body_radius_m"]))
        clamped_y = float(np.clip(y, min_y + agent["body_radius_m"], max_y - agent["body_radius_m"]))
        boundary_delta = math.hypot(clamped_x - x, clamped_y - y)
        projection_cost += boundary_delta
        boundary_violation += boundary_delta
        agent["X"], agent["Y"] = clamped_x, clamped_y

        for poly in scene.obstacle_polygons:
            if point_in_polygon((agent["X"], agent["Y"]), poly):
                obstacle_violations += 1
                nearest = nearest_point_on_polygon((agent["X"], agent["Y"]), poly)
                center = polygon_centroid(poly)
                outward = unit_vector((nearest[0] - center[0], nearest[1] - center[1]))
                target = (nearest[0] + outward[0] * (agent["body_radius_m"] + 0.08), nearest[1] + outward[1] * (agent["body_radius_m"] + 0.08))
                projection_cost += distance((agent["X"], agent["Y"]), target)
                agent["X"], agent["Y"] = target

    return {
        "state": corrected,
        "scene_projection_cost_m": float(projection_cost),
        "scene_violation_cost_m": float(boundary_violation + obstacle_violations * 0.8),
        "boundary_violation_m": float(boundary_violation),
        "obstacle_violation_count": int(obstacle_violations),
    }


def empty_scene_projection_info(state: List[Dict], scene: SceneGeometry) -> Dict:
    boundary, obstacle = scene_violation_metrics(state, scene)
    return {
        "state": [dict(agent) for agent in state],
        "scene_projection_cost_m": 0.0,
        "scene_violation_cost_m": float(boundary + obstacle * 0.8),
        "boundary_violation_m": float(boundary),
        "obstacle_violation_count": int(obstacle),
    }


def observation_loglikelihood(observation: List[Dict], latent_world_state: List[Dict], calibration: Calibration) -> float:
    by_id = {obs["id"]: obs for obs in observation}
    loglik = 0.0
    sigma2 = OBS_SIGMA_PX**2
    for agent in latent_world_state:
        obs = by_id.get(agent["id"])
        if not obs:
            loglik -= 4.0
            continue
        u, v = ground_to_image_homography(agent["X"], agent["Y"], calibration.H)
        du, dv = obs["u"] - u, obs["v"] - v
        loglik += -0.5 * (du * du + dv * dv) / sigma2
        observed_radius_px = max(obs["bbox"][2], obs["bbox"][3]) * 0.5
        projected_radius_px = agent["body_radius_m"] / calibration.meter_per_pixel
        loglik += -0.5 * ((observed_radius_px - projected_radius_px) ** 2) / (3.0**2)
        loglik += math.log(max(1e-3, obs["confidence"]))
    return float(loglik)


def transition_logprob_placeholder() -> float:
    return 0.0


def speed_prior_penalty(state: List[Dict]) -> float:
    penalty = 0.0
    for agent in state:
        speed = math.hypot(agent["Vx"], agent["Vy"])
        if speed > MAX_PEDESTRIAN_SPEED_MPS:
            penalty += (speed - MAX_PEDESTRIAN_SPEED_MPS) * 2.0
        acc = math.hypot(agent["Ax"], agent["Ay"])
        if acc > 3.0:
            penalty += (acc - 3.0) * 0.4
    return float(penalty)


def normalize_world_particles(particles: List[Dict]) -> List[Dict]:
    max_log = max(p["log_weight"] for p in particles)
    weights = np.array([math.exp(p["log_weight"] - max_log) for p in particles], dtype=float)
    total = float(weights.sum()) or 1.0
    for particle, weight in zip(particles, weights):
        particle["probability"] = float(weight / total)
    return sorted(particles, key=lambda p: p["probability"], reverse=True)


def effective_sample_size(particles: List[Dict]) -> float:
    probs = np.array([p["probability"] for p in particles], dtype=float)
    return float(1.0 / max(1e-12, np.sum(probs * probs)))


def systematic_resample_world(particles: List[Dict], count: int, rng: np.random.Generator) -> List[Dict]:
    probs = np.array([p["probability"] for p in particles], dtype=float)
    probs = probs / max(1e-12, probs.sum())
    positions = (rng.random() + np.arange(count)) / count
    cumulative = np.cumsum(probs)
    indexes = np.searchsorted(cumulative, positions, side="left")
    resampled = []
    for index in indexes:
        source = particles[int(min(index, len(particles) - 1))]
        resampled.append(
            {
                "world_state": clone_world_state(source["world_state"]),
                "latent_goals": {key: list(value) for key, value in source["latent_goals"].items()},
                "latent_body_params": json.loads(json.dumps(source["latent_body_params"])),
                "log_weight": 0.0,
                "probability": 1.0 / count,
                "history": {int(step): clone_world_state(state) for step, state in source["history"].items()},
                "diagnostics": list(source["diagnostics"]),
            }
        )
    return resampled


def rejuvenate_particles(particles: List[Dict], rng: np.random.Generator) -> List[Dict]:
    for particle in particles:
        for agent in particle["world_state"]:
            if rng.random() < 0.18:
                goal = np.array(agent["goal_latent"], dtype=float) + rng.normal(0.0, 0.35, size=2)
                agent["goal_latent"] = [float(goal[0]), float(goal[1])]
            if rng.random() < 0.10:
                agent["desired_speed_mps"] = float(np.clip(rng.normal(agent["desired_speed_mps"], 0.08), 0.35, 1.9))
    return particles


def weighted_centroid_trajectory(particles: List[Dict], horizon: int) -> List[Dict]:
    steps = sorted(set().union(*(p["history"].keys() for p in particles)))
    output = []
    for step in steps:
        if step > horizon:
            continue
        cx, cy = weighted_centroid_at_step(particles, step)
        output.append({"step": int(step), "centroid_world": [round(cx, 3), round(cy, 3)]})
    return output


def weighted_centroid_at_step(particles: List[Dict], step: int) -> Tuple[float, float]:
    total = 0.0
    cx, cy = 0.0, 0.0
    for particle in particles:
        state = particle["history"].get(step, particle["world_state"])
        p = particle.get("probability", 1.0)
        c = centroid_world(state)
        cx += p * c[0]
        cy += p * c[1]
        total += p
    return cx / max(1e-12, total), cy / max(1e-12, total)


def weighted_mean_state_at_step(particles: List[Dict], step: int) -> List[Dict]:
    ids = [agent["id"] for agent in particles[0]["world_state"]]
    result = []
    for agent_id in ids:
        accum: Dict[str, float] = {}
        total = 0.0
        template = None
        for particle in particles:
            state = particle["history"].get(step, particle["world_state"])
            agent = next((item for item in state if item["id"] == agent_id), None)
            if not agent:
                continue
            template = agent
            p = particle.get("probability", 1.0)
            total += p
            for key in ["X", "Y", "Z", "Vx", "Vy", "Vz", "Ax", "Ay", "body_radius_m", "body_height_m", "mass_kg", "desired_speed_mps"]:
                accum[key] = accum.get(key, 0.0) + p * float(agent[key])
        if template is not None and total > 0:
            item = dict(template)
            for key, value in accum.items():
                item[key] = value / total
            result.append(item)
    return result


def run_baselines(
    scene_tracks: pd.DataFrame,
    start_frame: int,
    active_ids: List[int],
    calibration: Calibration,
    eval_horizons: List[int],
    max_observed_horizon: int,
) -> Dict:
    baseline0 = constant_velocity_baseline(scene_tracks, start_frame, active_ids, calibration, max_observed_horizon)
    baseline1 = legacy_pixel_social_mlp_baseline(scene_tracks, start_frame, active_ids, calibration, max_observed_horizon)
    return {
        "Baseline 0: old pixel linear/constant-velocity": baseline0,
        "Baseline 1: current Social-MLP + SMC": baseline1,
    }


def constant_velocity_baseline(
    scene_tracks: pd.DataFrame,
    start_frame: int,
    active_ids: List[int],
    calibration: Calibration,
    horizon: int,
) -> Dict:
    initial = initialize_world_state_from_observation(scene_tracks, start_frame, active_ids, calibration, build_scene_geometry("none", calibration))
    predicted = []
    for agent in initial:
        item = dict(agent)
        item["X"] = agent["X"] + agent["Vx"] * DT * horizon
        item["Y"] = agent["Y"] + agent["Vy"] * DT * horizon
        predicted.append(item)
    actual = actual_world_state(scene_tracks, start_frame + horizon, active_ids, calibration)
    metrics = endpoint_metrics(predicted, actual, calibration)
    return {"mode": "pixel x/y/vx/vy converted through weak scale; no world constraints", **metrics}


def legacy_pixel_social_mlp_baseline(
    scene_tracks: pd.DataFrame,
    start_frame: int,
    active_ids: List[int],
    calibration: Calibration,
    horizon: int,
) -> Dict:
    if not LEGACY_PIXEL_MODEL.exists():
        return {"available": False, "reason": "legacy model artifact missing"}
    try:
        import __main__

        setattr(__main__, "SocialNonlinearModel", SocialNonlinearModel)
        loaded = load(LEGACY_PIXEL_MODEL)
        model = loaded["model"]
        initial_pixel_state = [agent for agent in state_at_frame(scene_tracks, start_frame) if int(agent["id"]) in set(active_ids)]
        prediction = smc_rollout(initial_pixel_state, model, scene_tracks, horizon, seed=616)
        representative = prediction["outcomes"][0]["representative"]
        actual = actual_world_state(scene_tracks, start_frame + horizon, active_ids, calibration)
        actual_centroid = centroid_world(actual)
        pred_u, pred_v = representative["center"]
        pred_world = image_to_ground_homography(pred_u, pred_v, calibration.H_inv)
        centroid_error_m = distance(pred_world, actual_centroid)
        centroid_error_px = centroid_error_m / calibration.meter_per_pixel
        return {
            "available": True,
            "mode": "legacy pixel-space Social-MLP + SMC evaluated only by centroid for this start frame",
            "centroid_error_px": round(centroid_error_px, 3),
            "centroid_error_m": round(centroid_error_m, 3),
            "min_gap_m": round(representative["min_gap_px"] * calibration.meter_per_pixel, 3),
            "collision_count": representative["collisions"],
            "terminal_event": representative["label"],
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def build_horizon_diagnostics(
    free_run: Dict,
    teacher_forced: Dict,
    scene_tracks: pd.DataFrame,
    start_frame: int,
    active_ids: List[int],
    calibration: Calibration,
    eval_horizons: List[int],
) -> List[Dict]:
    rows = []
    for mode, run in [("free-run world rollout", free_run), ("teacher-forced filtering", teacher_forced)]:
        for horizon in eval_horizons:
            predicted = weighted_mean_state_at_step(run["particles"], horizon)
            actual = actual_world_state(scene_tracks, start_frame + horizon, active_ids, calibration)
            metrics = trajectory_metrics(run["particles"], scene_tracks, start_frame, active_ids, calibration, horizon)
            endpoint = endpoint_metrics(predicted, actual, calibration)
            rows.append(
                {
                    "mode": mode,
                    "horizon": horizon,
                    "ADE_px": metrics["ADE_px"],
                    "ADE_m": metrics["ADE_m"],
                    "FDE_px": endpoint["FDE_px"],
                    "FDE_m": endpoint["FDE_m"],
                    "centroid_error_px": endpoint["centroid_error_px"],
                    "centroid_error_m": endpoint["centroid_error_m"],
                    "speed_error_mps": endpoint["speed_error_mps"],
                    "heading_error_deg": endpoint["heading_error_deg"],
                    "min_gap_m": endpoint["min_gap_m"],
                    "collision_count": endpoint["collision_count"],
                    "boundary_violation_m": endpoint["boundary_violation_m"],
                    "obstacle_violation_count": endpoint["obstacle_violation_count"],
                    "ESS": run["config"]["mean_ess"],
                    "terminal_modes": len(cluster_terminal_world_states(run["particles"], calibration, build_scene_geometry("none", calibration), centroid_world(predicted))),
                }
            )
    for horizon in [20, 50, 100]:
        if horizon not in eval_horizons:
            rows.append(
                {
                    "mode": "evaluation unavailable",
                    "horizon": horizon,
                    "reason": "AerialMPT scene has no ground-truth frame at this horizon from the chosen start frame.",
                }
            )
    return rows


def build_ablation_table(
    baselines: Dict,
    world_runs: Dict,
    scene_tracks: pd.DataFrame,
    start_frame: int,
    active_ids: List[int],
    calibration: Calibration,
    eval_horizon: int,
) -> List[Dict]:
    rows = []
    for name, baseline in baselines.items():
        row = {"version": name, "runtime_s": None, "ESS": None}
        row.update({key: baseline.get(key) for key in ["centroid_error_px", "centroid_error_m", "min_gap_m", "collision_count"]})
        row["FDE_px@observed"] = baseline.get("FDE_px")
        row["FDE_m@observed"] = baseline.get("FDE_m")
        row["ADE_px@100"] = None
        row["ADE_m@100"] = None
        row["terminal_event_acc"] = None
        row["terminal_spatial_acc"] = None
        row["Brier"] = None
        rows.append(row)

    for name, run in world_runs.items():
        predicted = weighted_mean_state_at_step(run["particles"], eval_horizon)
        actual = actual_world_state(scene_tracks, start_frame + eval_horizon, active_ids, calibration)
        metrics = endpoint_metrics(predicted, actual, calibration)
        terminal_clusters = cluster_terminal_world_states(run["particles"], calibration, build_scene_geometry("none", calibration), centroid_world(predicted))
        top_event = terminal_clusters[0]["event_cluster"] if terminal_clusters else None
        actual_event = classify_event(actual)
        rows.append(
            {
                "version": name,
                "FDE_px@observed": metrics["FDE_px"],
                "FDE_m@observed": metrics["FDE_m"],
                "ADE_px@100": None,
                "ADE_m@100": None,
                "centroid_error_px": metrics["centroid_error_px"],
                "centroid_error_m": metrics["centroid_error_m"],
                "min_gap_m": metrics["min_gap_m"],
                "collision_error": abs(metrics["collision_count"] - collision_count_world(actual)),
                "boundary_violation": metrics["boundary_violation_m"],
                "terminal_event_acc": 1.0 if top_event == actual_event else 0.0,
                "terminal_spatial_acc": spatial_accuracy(centroid_world(predicted), centroid_world(actual), calibration),
                "Brier": brier_for_event(terminal_clusters, actual_event),
                "ESS": run["config"]["mean_ess"],
                "runtime_s": run.get("runtime_s"),
                "note": "t+100 ADE unavailable; sequence is too short for t+100 ground truth.",
            }
        )
    return rows


def trajectory_metrics(
    particles: List[Dict],
    scene_tracks: pd.DataFrame,
    start_frame: int,
    active_ids: List[int],
    calibration: Calibration,
    horizon: int,
) -> Dict:
    errors_m = []
    errors_px = []
    for step in range(1, horizon + 1):
        actual = actual_world_state(scene_tracks, start_frame + step, active_ids, calibration)
        if not actual:
            continue
        predicted = weighted_mean_state_at_step(particles, step)
        by_pred = {agent["id"]: agent for agent in predicted}
        for actual_agent in actual:
            pred = by_pred.get(actual_agent["id"])
            if not pred:
                continue
            err_m = distance((pred["X"], pred["Y"]), (actual_agent["X"], actual_agent["Y"]))
            errors_m.append(err_m)
            errors_px.append(err_m / calibration.meter_per_pixel)
    return {"ADE_m": round(float(np.mean(errors_m)), 3) if errors_m else None, "ADE_px": round(float(np.mean(errors_px)), 3) if errors_px else None}


def endpoint_metrics(predicted: List[Dict], actual: List[Dict], calibration: Calibration) -> Dict:
    by_pred = {agent["id"]: agent for agent in predicted}
    errors_m = []
    speed_errors = []
    heading_errors = []
    for actual_agent in actual:
        pred = by_pred.get(actual_agent["id"])
        if not pred:
            continue
        errors_m.append(distance((pred["X"], pred["Y"]), (actual_agent["X"], actual_agent["Y"])))
        speed_errors.append(abs(math.hypot(pred["Vx"], pred["Vy"]) - math.hypot(actual_agent["Vx"], actual_agent["Vy"])))
        heading_errors.append(heading_error(pred, actual_agent))

    c_pred = centroid_world(predicted)
    c_actual = centroid_world(actual)
    centroid_error_m = distance(c_pred, c_actual) if actual else None
    boundary, obstacle = scene_violation_metrics(predicted, build_scene_geometry("none", calibration))
    return {
        "FDE_m": round(float(np.mean(errors_m)), 3) if errors_m else None,
        "FDE_px": round(float(np.mean(errors_m) / calibration.meter_per_pixel), 3) if errors_m else None,
        "centroid_error_m": round(centroid_error_m, 3) if centroid_error_m is not None else None,
        "centroid_error_px": round(centroid_error_m / calibration.meter_per_pixel, 3) if centroid_error_m is not None else None,
        "speed_error_mps": round(float(np.mean(speed_errors)), 3) if speed_errors else None,
        "acceleration_error_mps2": None,
        "heading_error_deg": round(float(np.mean(heading_errors)), 3) if heading_errors else None,
        "min_gap_m": round(min_gap_world(predicted), 3) if predicted else None,
        "collision_count": collision_count_world(predicted),
        "penetration_m": round(max(0.0, -min_gap_world(predicted)), 3) if predicted else None,
        "density_people_per_m2": round(density_people_per_m2(predicted), 3) if predicted else None,
        "boundary_violation_m": round(boundary, 3),
        "obstacle_violation_count": int(obstacle),
    }


def actual_world_state(scene_tracks: pd.DataFrame, frame: int, active_ids: List[int], calibration: Calibration) -> List[Dict]:
    current_obs = {obs["id"]: obs for obs in observations_at_frame(scene_tracks, frame, active_ids)}
    previous_obs = {obs["id"]: obs for obs in observations_at_frame(scene_tracks, frame - 1, active_ids)}
    output = []
    for agent_id, obs in current_obs.items():
        X, Y = image_to_ground_homography(obs["u"], obs["v"], calibration.H_inv)
        Vx, Vy = 0.0, 0.0
        if agent_id in previous_obs:
            px, py = image_to_ground_homography(previous_obs[agent_id]["u"], previous_obs[agent_id]["v"], calibration.H_inv)
            Vx = (X - px) / DT
            Vy = (Y - py) / DT
        output.append(
            {
                "id": int(agent_id),
                "X": X,
                "Y": Y,
                "Z": 0.0,
                "Vx": Vx,
                "Vy": Vy,
                "Vz": 0.0,
                "Ax": 0.0,
                "Ay": 0.0,
                "body_radius_m": DEFAULT_BODY_RADIUS_M,
                "body_height_m": DEFAULT_BODY_HEIGHT_M,
                "mass_kg": DEFAULT_MASS_KG,
                "desired_speed_mps": max(0.1, math.hypot(Vx, Vy)),
                "desired_direction_world": list(unit_vector((Vx, Vy))),
                "goal_latent": [X, Y],
                "group_id_latent": int(agent_id),
                "uncertainty_covariance": [[0.0, 0.0], [0.0, 0.0]],
            }
        )
    return output


def cluster_terminal_world_states(
    particles: List[Dict],
    calibration: Calibration,
    scene_geometry: SceneGeometry,
    initial_centroid: Tuple[float, float],
) -> List[Dict]:
    if not particles:
        return []
    feature_rows = []
    for particle in particles:
        state = particle["world_state"]
        c = centroid_world(state)
        spread_x, spread_y = spread_xy(state, c)
        heading, speed = dominant_heading_speed(state)
        min_gap = min_gap_world(state)
        collisions = collision_count_world(state)
        boundary, obstacle = scene_violation_metrics(state, scene_geometry)
        density = density_people_per_m2(state)
        feature_rows.append([c[0], c[1], spread_x, spread_y, heading, speed, min_gap, collisions, boundary, obstacle, density])

    x = np.asarray(feature_rows, dtype=float)
    k = min(3, len(particles))
    if k <= 1:
        labels = np.zeros(len(particles), dtype=int)
    else:
        scaled = x.copy()
        scaled[:, :4] /= 10.0
        labels = KMeans(n_clusters=k, n_init=10, random_state=7).fit_predict(scaled)

    clusters = []
    for label in sorted(set(int(item) for item in labels)):
        members = [particle for particle, particle_label in zip(particles, labels) if int(particle_label) == label]
        mass = sum(member.get("probability", 0.0) for member in members)
        representative = max(members, key=lambda item: item.get("probability", 0.0))
        state = representative["world_state"]
        c = centroid_world(state)
        u, v = ground_to_image_homography(c[0], c[1], calibration.H)
        heading, speed = dominant_heading_speed(state)
        min_gap = min_gap_world(state)
        collisions = collision_count_world(state)
        boundary, obstacle = scene_violation_metrics(state, scene_geometry)
        clusters.append(
            {
                "cluster_id": int(label),
                "event_cluster": classify_event(state),
                "spatial_cluster": classify_spatial(c, initial_centroid),
                "probability_mass": round(float(mass), 5),
                "centroid_world": [round(c[0], 3), round(c[1], 3)],
                "centroid_image": [round(u, 2), round(v, 2)],
                "min_gap_m": round(min_gap, 3),
                "collision_count": int(collisions),
                "boundary_violation_count": int(boundary > 0.01),
                "obstacle_violation_count": int(obstacle),
                "dominant_heading_rad": round(heading, 3),
                "dominant_speed_mps": round(speed, 3),
                "representative_particle_probability": round(representative.get("probability", 0.0), 5),
            }
        )
    return sorted(clusters, key=lambda item: item["probability_mass"], reverse=True)


def classify_event(state: List[Dict]) -> str:
    min_gap = min_gap_world(state)
    collisions = collision_count_world(state)
    _, obstacle = scene_violation_metrics(state, build_scene_geometry("none", setup_weak_homography((612, 552))))
    _, speed = dominant_heading_speed(state)
    if collisions > 0 or min_gap < 0:
        return "collision-risk"
    if obstacle > 0:
        return "boundary-conflict"
    if speed < 0.25:
        return "jammed"
    if spread_xy(state, centroid_world(state))[0] + spread_xy(state, centroid_world(state))[1] > 6.0:
        return "dispersed"
    return "free-flow"


def classify_spatial(centroid: Tuple[float, float], initial: Tuple[float, float]) -> str:
    dx, dy = centroid[0] - initial[0], centroid[1] - initial[1]
    if abs(dx) < 2.0 and abs(dy) < 2.0:
        return "center-stable"
    horizontal = "east" if dx > 0 else "west"
    vertical = "south" if dy > 0 else "north"
    if abs(dx) > abs(dy) * 1.5:
        return f"{horizontal}-shifted"
    if abs(dy) > abs(dx) * 1.5:
        return f"{vertical}-shifted"
    return f"{vertical}-{horizontal}-shifted"


def spatial_accuracy(predicted_centroid: Tuple[float, float], actual_centroid: Tuple[float, float], calibration: Calibration) -> float:
    error_px = distance(predicted_centroid, actual_centroid) / calibration.meter_per_pixel
    if error_px < 20:
        return 1.0
    if error_px < 40:
        return 0.5
    return 0.0


def brier_for_event(clusters: List[Dict], actual_event: str) -> Optional[float]:
    if not clusters:
        return None
    probability = sum(cluster["probability_mass"] for cluster in clusters if cluster["event_cluster"] == actual_event)
    return round(float((probability - 1.0) ** 2), 4)


def render_visualizations(
    image_path: Path,
    initial_state: List[Dict],
    scene_tracks: pd.DataFrame,
    start_frame: int,
    active_ids: List[int],
    calibration: Calibration,
    scene_geometry: SceneGeometry,
    free_run: Dict,
    teacher_forced: Dict,
    terminal_clusters: List[Dict],
    eval_horizon: int,
) -> Dict:
    paths = {
        "image_overlay": str(OUT_DIR / "image_overlay.png"),
        "world_topdown": str(OUT_DIR / "world_topdown.png"),
        "centroid_trajectory": str(OUT_DIR / "centroid_trajectory.png"),
        "terminal_cluster_bars": str(OUT_DIR / "terminal_cluster_probability.png"),
        "flow_goal_vector_map": str(OUT_DIR / "flow_goal_vector_map.png"),
        "collision_min_gap_map": str(OUT_DIR / "collision_min_gap_map.png"),
    }
    actual = actual_world_state(scene_tracks, start_frame + eval_horizon, active_ids, calibration)
    render_image_overlay(image_path, initial_state, actual, free_run, terminal_clusters, calibration, paths["image_overlay"])
    render_world_topdown(initial_state, actual, free_run, terminal_clusters, calibration, scene_geometry, paths["world_topdown"])
    render_centroid_trajectory(initial_state, actual, free_run, teacher_forced, calibration, eval_horizon, paths["centroid_trajectory"])
    render_probability_bars(terminal_clusters, paths["terminal_cluster_bars"])
    render_flow_goal_vector_map(initial_state, scene_geometry, calibration, paths["flow_goal_vector_map"])
    render_collision_min_gap_map(free_run["particles"][0]["world_state"], calibration, scene_geometry, paths["collision_min_gap_map"])
    return paths


def render_image_overlay(
    image_path: Path,
    initial: List[Dict],
    actual: List[Dict],
    run: Dict,
    clusters: List[Dict],
    calibration: Calibration,
    out_path: str,
) -> None:
    with Image.open(image_path).convert("RGB") as base:
        image = base.copy()
    draw = ImageDraw.Draw(image, "RGBA")
    font = load_font(12)

    for agent in initial:
        u, v = ground_to_image_homography(agent["X"], agent["Y"], calibration.H)
        r = agent["body_radius_m"] / calibration.meter_per_pixel
        draw.ellipse((u - r, v - r, u + r, v + r), outline=(255, 255, 255, 230), width=2)
        draw.line((u, v, u + agent["Vx"] / calibration.meter_per_pixel, v + agent["Vy"] / calibration.meter_per_pixel), fill=(120, 210, 255, 220), width=2)

    for agent in actual:
        u, v = ground_to_image_homography(agent["X"], agent["Y"], calibration.H)
        draw.line((u - 5, v - 5, u + 5, v + 5), fill=(255, 0, 0, 230), width=2)
        draw.line((u - 5, v + 5, u + 5, v - 5), fill=(255, 0, 0, 230), width=2)

    colors = [(255, 176, 0, 230), (0, 180, 150, 230), (80, 130, 255, 230)]
    for cluster, color in zip(clusters[:3], colors):
        u, v = cluster["centroid_image"]
        radius = 9 + 10 * cluster["probability_mass"]
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), outline=color, width=3)
        draw.text((u + 8, v - 10), f"p={cluster['probability_mass']:.2f}", fill=color, font=font)

    pred_c = clusters[0]["centroid_world"] if clusters else centroid_world(run["particles"][0]["world_state"])
    pred_u, pred_v = ground_to_image_homography(pred_c[0], pred_c[1], calibration.H)
    actual_c = centroid_world(actual)
    actual_u, actual_v = ground_to_image_homography(actual_c[0], actual_c[1], calibration.H)
    draw.line((pred_u, pred_v, actual_u, actual_v), fill=(255, 40, 40, 230), width=3)
    draw.ellipse((pred_u - 5, pred_v - 5, pred_u + 5, pred_v + 5), fill=(255, 176, 0, 230))
    draw.ellipse((actual_u - 5, actual_v - 5, actual_u + 5, actual_v + 5), fill=(255, 0, 0, 230))

    panel = Image.new("RGB", (image.width, 96), (245, 248, 250))
    pdraw = ImageDraw.Draw(panel)
    pdraw.text((10, 8), "Pseudo-3D image overlay: white=initial cylinders, red=true observed horizon, colored=top terminal clusters", fill=(40, 50, 60), font=font)
    pdraw.text((10, 32), "Projection is weak homography from GSD; vertical body height is latent, not directly recovered.", fill=(65, 80, 95), font=font)
    combined = Image.new("RGB", (image.width, image.height + panel.height), (255, 255, 255))
    combined.paste(image, (0, 0))
    combined.paste(panel, (0, image.height))
    combined.save(out_path)


def render_world_topdown(
    initial: List[Dict],
    actual: List[Dict],
    run: Dict,
    clusters: List[Dict],
    calibration: Calibration,
    scene: SceneGeometry,
    out_path: str,
) -> None:
    canvas = WorldCanvas(scene.walkable_polygon, 900, 760, margin=50)
    image = Image.new("RGB", (canvas.width, canvas.height), (250, 252, 253))
    draw = ImageDraw.Draw(image, "RGBA")
    font = load_font(12)

    draw_polygon(draw, canvas, scene.walkable_polygon, fill=(230, 240, 235, 255), outline=(50, 70, 80, 255), width=2)
    for poly in scene.obstacle_polygons:
        draw_polygon(draw, canvas, poly, fill=(90, 90, 90, 170), outline=(40, 40, 40, 255), width=2)
    for region in scene.goal_regions:
        x, y = canvas.to_px(region["center"][0], region["center"][1])
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=(0, 150, 210, 160))
        draw.text((x + 12, y - 8), region["name"], fill=(0, 80, 120), font=font)

    for particle in run["particles"]:
        p = max(0.05, particle.get("probability", 0.0) * 8)
        for agent in particle["world_state"]:
            x, y = canvas.to_px(agent["X"], agent["Y"])
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 176, 0, int(30 + 120 * p)))

    for agent in initial:
        draw_world_disk(draw, canvas, agent, (255, 255, 255, 230), outline=(0, 120, 220, 255))
        gx, gy = agent["goal_latent"]
        x1, y1 = canvas.to_px(agent["X"], agent["Y"])
        x2, y2 = canvas.to_px(gx, gy)
        draw.line((x1, y1, x2, y2), fill=(0, 120, 220, 90), width=1)

    for agent in actual:
        draw_world_cross(draw, canvas, agent, (255, 0, 0, 230))

    colors = [(255, 176, 0, 255), (0, 160, 130, 255), (80, 130, 255, 255)]
    for cluster, color in zip(clusters[:3], colors):
        x, y = canvas.to_px(cluster["centroid_world"][0], cluster["centroid_world"][1])
        draw.ellipse((x - 12, y - 12, x + 12, y + 12), outline=color, width=4)
        draw.text((x + 14, y - 8), f"{cluster['event_cluster']} p={cluster['probability_mass']:.2f}", fill=color, font=font)

    draw.text((18, 14), "World top-down plot: metric pseudo-3D ground plane", fill=(30, 40, 50), font=font)
    draw.text((18, 34), "green=walkable, gray=manual obstacles, blue disks=initial cylinders, red X=true observed, amber=terminal particles", fill=(70, 80, 90), font=font)
    image.save(out_path)


def render_centroid_trajectory(
    initial: List[Dict],
    actual: List[Dict],
    free_run: Dict,
    teacher_forced: Dict,
    calibration: Calibration,
    eval_horizon: int,
    out_path: str,
) -> None:
    bounds_poly = [
        (-calibration.image_width * calibration.meter_per_pixel / 2, -calibration.image_height * calibration.meter_per_pixel / 2),
        (calibration.image_width * calibration.meter_per_pixel / 2, -calibration.image_height * calibration.meter_per_pixel / 2),
        (calibration.image_width * calibration.meter_per_pixel / 2, calibration.image_height * calibration.meter_per_pixel / 2),
        (-calibration.image_width * calibration.meter_per_pixel / 2, calibration.image_height * calibration.meter_per_pixel / 2),
    ]
    canvas = WorldCanvas(bounds_poly, 760, 520, margin=50)
    image = Image.new("RGB", (canvas.width, canvas.height), (250, 252, 253))
    draw = ImageDraw.Draw(image, "RGBA")
    font = load_font(12)
    draw_polygon(draw, canvas, bounds_poly, fill=(245, 248, 248, 255), outline=(100, 115, 120, 255), width=2)

    draw_centroid_line(draw, canvas, free_run["centroid_trajectory"], (255, 176, 0, 230), width=3)
    draw_centroid_line(draw, canvas, teacher_forced["centroid_trajectory"], (0, 130, 220, 230), width=3)
    c0 = centroid_world(initial)
    c_actual = centroid_world(actual)
    draw_circle_at_world(draw, canvas, c0, 6, (0, 120, 220, 255))
    draw_circle_at_world(draw, canvas, c_actual, 7, (255, 0, 0, 255))
    draw.text((18, 14), "Centroid trajectory comparison", fill=(30, 40, 50), font=font)
    draw.text((18, 36), "amber=free-run, blue=teacher-forced filtering, red=true observed endpoint", fill=(70, 80, 90), font=font)
    draw.text((18, canvas.height - 28), f"actual evaluation endpoint: t+{eval_horizon}; t+100 has no observation in this sequence", fill=(70, 80, 90), font=font)
    image.save(out_path)


def render_probability_bars(clusters: List[Dict], out_path: str) -> None:
    image = Image.new("RGB", (760, 360), (250, 252, 253))
    draw = ImageDraw.Draw(image)
    font = load_font(13)
    draw.text((20, 18), "Terminal cluster probability mass", fill=(30, 40, 50), font=font)
    for index, cluster in enumerate(clusters[:5]):
        y = 68 + index * 52
        width = int(560 * cluster["probability_mass"])
        draw.rectangle((180, y, 180 + width, y + 24), fill=(255, 176, 0))
        draw.rectangle((180, y, 740, y + 24), outline=(110, 120, 130))
        draw.text((20, y + 2), f"{cluster['event_cluster']} / {cluster['spatial_cluster']}", fill=(40, 50, 60), font=font)
        draw.text((185 + width, y + 2), f"{cluster['probability_mass']:.3f}", fill=(40, 50, 60), font=font)
    image.save(out_path)


def render_flow_goal_vector_map(initial: List[Dict], scene: SceneGeometry, calibration: Calibration, out_path: str) -> None:
    canvas = WorldCanvas(scene.walkable_polygon, 820, 650, margin=50)
    image = Image.new("RGB", (canvas.width, canvas.height), (250, 252, 253))
    draw = ImageDraw.Draw(image, "RGBA")
    font = load_font(12)
    draw_polygon(draw, canvas, scene.walkable_polygon, fill=(238, 245, 241, 255), outline=(70, 90, 100, 255), width=2)
    for poly in scene.obstacle_polygons:
        draw_polygon(draw, canvas, poly, fill=(100, 100, 100, 130), outline=(60, 60, 60, 255), width=2)
    for agent in initial:
        x1, y1 = canvas.to_px(agent["X"], agent["Y"])
        gx, gy = agent["goal_latent"]
        x2, y2 = canvas.to_px(gx, gy)
        draw.line((x1, y1, x2, y2), fill=(0, 130, 220, 160), width=2)
        draw.ellipse((x1 - 4, y1 - 4, x1 + 4, y1 + 4), fill=(0, 130, 220, 230))
    draw.text((18, 14), "Flow / latent goal vector map", fill=(30, 40, 50), font=font)
    draw.text((18, 36), "Vectors are inferred from observed velocity history and scene goal regions before rollout.", fill=(70, 80, 90), font=font)
    image.save(out_path)


def render_collision_min_gap_map(state: List[Dict], calibration: Calibration, scene: SceneGeometry, out_path: str) -> None:
    canvas = WorldCanvas(scene.walkable_polygon, 820, 650, margin=50)
    image = Image.new("RGB", (canvas.width, canvas.height), (250, 252, 253))
    draw = ImageDraw.Draw(image, "RGBA")
    font = load_font(12)
    draw_polygon(draw, canvas, scene.walkable_polygon, fill=(245, 248, 248, 255), outline=(100, 115, 120, 255), width=2)
    for agent in state:
        draw_world_disk(draw, canvas, agent, (255, 255, 255, 200), outline=(255, 176, 0, 230))
    risky_pairs = []
    for i, a in enumerate(state):
        for b in state[i + 1 :]:
            gap = distance((a["X"], a["Y"]), (b["X"], b["Y"])) - (a["body_radius_m"] + b["body_radius_m"])
            if gap < 0.25:
                risky_pairs.append((gap, a, b))
    for gap, a, b in sorted(risky_pairs)[:20]:
        x1, y1 = canvas.to_px(a["X"], a["Y"])
        x2, y2 = canvas.to_px(b["X"], b["Y"])
        color = (255, 0, 0, 220) if gap < 0 else (255, 110, 0, 150)
        draw.line((x1, y1, x2, y2), fill=color, width=2)
    draw.text((18, 14), "Collision / min-gap map", fill=(30, 40, 50), font=font)
    draw.text((18, 36), f"terminal min_gap_m={min_gap_world(state):.3f}, collision_count={collision_count_world(state)}", fill=(70, 80, 90), font=font)
    image.save(out_path)


class WorldCanvas:
    def __init__(self, polygon: Sequence[Tuple[float, float]], width: int, height: int, margin: int) -> None:
        self.width = width
        self.height = height
        self.margin = margin
        min_x, min_y, max_x, max_y = polygon_bounds(polygon)
        self.min_x, self.min_y, self.max_x, self.max_y = min_x, min_y, max_x, max_y
        sx = (width - 2 * margin) / max(1e-6, max_x - min_x)
        sy = (height - 2 * margin) / max(1e-6, max_y - min_y)
        self.scale = min(sx, sy)

    def to_px(self, x: float, y: float) -> Tuple[float, float]:
        px = self.margin + (x - self.min_x) * self.scale
        py = self.margin + (y - self.min_y) * self.scale
        return px, py

    def radius_px(self, radius_m: float) -> float:
        return radius_m * self.scale


def draw_polygon(draw: ImageDraw.ImageDraw, canvas: WorldCanvas, poly: Sequence[Tuple[float, float]], fill, outline, width: int) -> None:
    points = [canvas.to_px(x, y) for x, y in poly]
    draw.polygon(points, fill=fill, outline=outline)
    if width > 1:
        draw.line(points + [points[0]], fill=outline, width=width)


def draw_world_disk(draw: ImageDraw.ImageDraw, canvas: WorldCanvas, agent: Dict, fill, outline) -> None:
    x, y = canvas.to_px(agent["X"], agent["Y"])
    r = max(3.0, canvas.radius_px(agent["body_radius_m"]))
    draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline, width=2)


def draw_world_cross(draw: ImageDraw.ImageDraw, canvas: WorldCanvas, agent: Dict, color) -> None:
    x, y = canvas.to_px(agent["X"], agent["Y"])
    draw.line((x - 5, y - 5, x + 5, y + 5), fill=color, width=2)
    draw.line((x - 5, y + 5, x + 5, y - 5), fill=color, width=2)


def draw_circle_at_world(draw: ImageDraw.ImageDraw, canvas: WorldCanvas, point: Tuple[float, float], radius: float, color) -> None:
    x, y = canvas.to_px(point[0], point[1])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def draw_centroid_line(draw: ImageDraw.ImageDraw, canvas: WorldCanvas, trajectory: List[Dict], color, width: int) -> None:
    points = [canvas.to_px(item["centroid_world"][0], item["centroid_world"][1]) for item in trajectory]
    if len(points) > 1:
        draw.line(points, fill=color, width=width)


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def polygon_bounds(poly: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [point[0] for point in poly]
    ys = [point[1] for point in poly]
    return min(xs), min(ys), max(xs), max(ys)


def point_in_polygon(point: Tuple[float, float], polygon: Sequence[Tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, pi in enumerate(polygon):
        xi, yi = pi
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / max(1e-12, yj - yi) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def nearest_point_on_polygon(point: Tuple[float, float], polygon: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    best = None
    best_dist = float("inf")
    for i, a in enumerate(polygon):
        b = polygon[(i + 1) % len(polygon)]
        candidate = nearest_point_on_segment(point, a, b)
        d = distance(point, candidate)
        if d < best_dist:
            best = candidate
            best_dist = d
    return best if best is not None else polygon[0]


def nearest_point_on_segment(point: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    p = np.array(point, dtype=float)
    av = np.array(a, dtype=float)
    bv = np.array(b, dtype=float)
    ab = bv - av
    t = float(np.dot(p - av, ab) / max(1e-12, np.dot(ab, ab)))
    t = float(np.clip(t, 0.0, 1.0))
    q = av + t * ab
    return float(q[0]), float(q[1])


def polygon_centroid(poly: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    pts = np.array(poly, dtype=float)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def clamp_to_walkable(point: Tuple[float, float], walkable: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    min_x, min_y, max_x, max_y = polygon_bounds(walkable)
    return float(np.clip(point[0], min_x, max_x)), float(np.clip(point[1], min_y, max_y))


def centroid_world(state: List[Dict]) -> Tuple[float, float]:
    if not state:
        return 0.0, 0.0
    return float(np.mean([agent["X"] for agent in state])), float(np.mean([agent["Y"] for agent in state]))


def spread_xy(state: List[Dict], center: Tuple[float, float]) -> Tuple[float, float]:
    if not state:
        return 0.0, 0.0
    xs = np.array([agent["X"] for agent in state], dtype=float)
    ys = np.array([agent["Y"] for agent in state], dtype=float)
    return float(np.std(xs - center[0])), float(np.std(ys - center[1]))


def dominant_heading_speed(state: List[Dict]) -> Tuple[float, float]:
    if not state:
        return 0.0, 0.0
    vx = float(np.mean([agent["Vx"] for agent in state]))
    vy = float(np.mean([agent["Vy"] for agent in state]))
    return math.atan2(vy, vx), math.hypot(vx, vy)


def min_gap_world(state: List[Dict]) -> float:
    gaps = []
    for i, a in enumerate(state):
        for b in state[i + 1 :]:
            gaps.append(distance((a["X"], a["Y"]), (b["X"], b["Y"])) - (a["body_radius_m"] + b["body_radius_m"]))
    return min(gaps) if gaps else 999.0


def collision_count_world(state: List[Dict]) -> int:
    count = 0
    for i, a in enumerate(state):
        for b in state[i + 1 :]:
            if distance((a["X"], a["Y"]), (b["X"], b["Y"])) < a["body_radius_m"] + b["body_radius_m"]:
                count += 1
    return count


def density_people_per_m2(state: List[Dict]) -> float:
    if len(state) < 2:
        return 0.0
    xs = [agent["X"] for agent in state]
    ys = [agent["Y"] for agent in state]
    area = max(1.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))
    return len(state) / area


def scene_violation_metrics(state: List[Dict], scene: SceneGeometry) -> Tuple[float, int]:
    min_x, min_y, max_x, max_y = polygon_bounds(scene.walkable_polygon)
    boundary = 0.0
    obstacle_count = 0
    for agent in state:
        x, y = agent["X"], agent["Y"]
        boundary += max(0.0, min_x + agent["body_radius_m"] - x)
        boundary += max(0.0, x - (max_x - agent["body_radius_m"]))
        boundary += max(0.0, min_y + agent["body_radius_m"] - y)
        boundary += max(0.0, y - (max_y - agent["body_radius_m"]))
        for poly in scene.obstacle_polygons:
            if point_in_polygon((x, y), poly):
                obstacle_count += 1
    return float(boundary), int(obstacle_count)


def heading_error(pred: Dict, actual: Dict) -> float:
    hp = math.atan2(pred["Vy"], pred["Vx"])
    ha = math.atan2(actual["Vy"], actual["Vx"])
    if math.hypot(pred["Vx"], pred["Vy"]) < 0.05 or math.hypot(actual["Vx"], actual["Vy"]) < 0.05:
        return 0.0
    diff = math.atan2(math.sin(hp - ha), math.cos(hp - ha))
    return abs(math.degrees(diff))


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit_vector(vec: Tuple[float, float]) -> Tuple[float, float]:
    norm = math.hypot(vec[0], vec[1])
    if norm < 1e-9:
        return 1.0, 0.0
    return vec[0] / norm, vec[1] / norm


def clone_world_state(state: List[Dict]) -> List[Dict]:
    return [json.loads(json.dumps(agent)) for agent in state]


def summarize_world_run(run: Dict) -> Dict:
    terminal = weighted_mean_state_at_step(run["particles"], run["config"]["horizon"])
    return {
        "config": run["config"],
        "terminal_summary": {
            "centroid_world": [round(x, 3) for x in centroid_world(terminal)],
            "mean_speed_mps": round(dominant_heading_speed(terminal)[1], 3),
            "min_gap_m": round(min_gap_world(terminal), 3),
            "collision_count": collision_count_world(terminal),
        },
        "runtime_s": run.get("runtime_s"),
    }


def observability_report() -> Dict:
    return {
        "directly_observed": ["image u/v point annotations", "track identity within a sequence", "bbox-like 4px annotation proxy", "frame time"],
        "derived_from_observation": ["ground X/Y under weak homography", "Vx/Vy from filtered image history", "density under scale assumption"],
        "latent_inferred": ["body radius", "body height", "mass", "desired speed", "goal", "group id", "state covariance"],
        "not_observed": ["true camera intrinsics", "true camera extrinsics", "real depth", "true 3D body pose", "obstacle semantics"],
        "largest_uncertainty": "camera scale/homography first, then scene geometry and latent goal inference",
    }


def build_report(summary: Dict) -> str:
    horizon_table = markdown_table(summary["horizon_diagnostics"])
    ablation_table = markdown_table(summary["ablation_table"])
    clusters_table = markdown_table(summary["terminal_world_clusters"])
    artifacts = summary["artifacts"]
    return f"""# Pseudo-3D Human Collision World Model Report

## 1. Current Baseline Limits

- Current Social-MLP/GNN-lite + SMC is still mainly a pixel-space model.
- It does not understand camera geometry; pixel radii were not metric body radii.
- It does not separate image observation error from latent physical state.
- Its collision projection is pixel-circle interaction, not world-space body interaction.
- It can label collision-risk events, but cannot reliably explain long-horizon spatial dynamics.

Important correction: this AerialMPT scene has only {summary['dataset']['frames_available']} frames. Therefore t+100 has no ground-truth observation from start frame {summary['dataset']['start_frame']}; t+100 is free-run only.

## 2. Pseudo-3D Design

- Image coordinates: [u, v] in pixels.
- World coordinates: [X, Y, Z] in meters, with Z = 0 on a flat ground plane.
- Body model: vertical cylinder/capsule with latent radius, height, mass, desired speed, and goal.
- Camera model: weak homography from DLR GSD range because no K/R/t or control points are present.
- Observation model: p(O_t | S_t) from projected world point error plus body-radius proxy error.
- Transition model: inertia + desired-velocity goal seeking + social repulsion + group cohesion + obstacle/boundary terms + process noise.
- SMC model: particles carry world_state, latent_goals, latent_body_params, log_weight, history, diagnostics.

## 3. Key Formulas

Image-to-ground homography:

```text
[X, Y, 1]^T ~ H_inv [u, v, 1]^T
```

Ground-to-image:

```text
[u, v, 1]^T ~ H [X, Y, 1]^T
```

Observation likelihood:

```text
log p(O_t | S_t) =
  - || [u_obs, v_obs] - project(X,Y) ||^2 / (2 sigma_px^2)
  - (r_obs_px - r_body_m / meter_per_pixel)^2 / (2 sigma_r^2)
```

World transition:

```text
V_i(t+1) = V_i(t) + dt * A_i(t)
X_i(t+1) = X_i(t) + dt * V_i(t+1)
A_i = A_goal + A_social + A_obstacle + A_boundary + A_group + epsilon
```

Collision penetration:

```text
penetration_ij = r_i + r_j + comfort_margin - ||center_i - center_j||
```

Importance update:

```text
log w_t = log w_{{t-1}}
        + log p(S_t | S_{{t-1}})
        - log q(S_t)
        + log p(O_t | S_t)
        - lambda_projection * projection_cost
        - lambda_scene * scene_violation_cost
```

Terminal cluster probability:

```text
P(cluster_k) = sum_i normalized_weight_i * 1[particle_i in cluster_k]
```

## 4. Key Pseudocode

```text
calibration:
  if K,R,t available: use pinhole projection
  elif control points available: estimate H
  else: set weak H from rough meter_per_pixel and report scale uncertainty

initialize_particles_from_observation:
  image observations -> ground points
  sample body radius, height, mass, desired speed, latent goal
  initialize covariance and log weight

world_transition_step:
  compute A_goal, A_social, A_obstacle, A_boundary, A_group
  sample process noise
  integrate velocity and position in meters

collision_projection_world:
  for each pair:
    if penetration > 0:
      separate along normal by inverse-mass ratio
      accumulate projection_cost

scene_constraint_projection:
  clamp walkable boundary
  push out of obstacle polygons
  record violation cost

observation_loglikelihood:
  project latent ground point to image
  score pixel error and projected body radius error

SMC:
  propose particles
  project physical constraints
  update log weights
  add observation likelihood if O_t exists
  normalize, compute ESS, resample when needed

terminal_world_clustering:
  cluster weighted particles by event and spatial features in world coordinates
```

## 5. Experiments

### Horizon Diagnostics

{horizon_table}

### Ablation Table

{ablation_table}

### Terminal Clusters

{clusters_table}

## 6. Visualization Artifacts

- Image overlay: `{artifacts['image_overlay']}`
- World top-down plot: `{artifacts['world_topdown']}`
- Centroid trajectory comparison: `{artifacts['centroid_trajectory']}`
- Terminal cluster probability bars: `{artifacts['terminal_cluster_bars']}`
- Flow/goal vector map: `{artifacts['flow_goal_vector_map']}`
- Collision/min-gap map: `{artifacts['collision_min_gap_map']}`

## 7. Judgment

This is now a real state-space world-model scaffold, but not true recovered 3D. It is pseudo-3D / 2.5D because ground X/Y are metric only under a weak GSD homography and body height is latent. The highest-value next data to add is camera/control-point calibration, then scene geometry labels, then longer trajectories for t+100 evaluation.
"""


def markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "_No rows._"
    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        values = []
        for key in keys:
            value = row.get(key, "")
            if isinstance(value, float):
                value = round(value, 4)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def key_console_summary(summary: Dict) -> Dict:
    full = summary["models"]["free_run_versions"]["Version D: full pseudo-3D world model"]
    observed_rows = [row for row in summary["horizon_diagnostics"] if row.get("mode") == "free-run world rollout" and row.get("horizon") == summary["dataset"]["max_observed_horizon"]]
    return {
        "dataset": summary["dataset"],
        "calibration_quality": summary["calibration_quality"],
        "full_model_terminal": full["terminal_summary"],
        "observed_endpoint_metrics": observed_rows[0] if observed_rows else None,
        "top_terminal_clusters": summary["terminal_world_clusters"][:3],
        "artifacts": summary["artifacts"],
    }


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items() if key != "particles"}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
