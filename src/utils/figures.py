from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.physics.collision import min_gap_and_collisions
from src.physics.scene_geometry import SceneSpec
from src.physics.social_force import local_density


PALETTE = {
    "gt": (20, 110, 90),
    "cv": (120, 120, 120),
    "physics": (45, 90, 170),
    "learned": (200, 80, 70),
    "branch": (140, 90, 180),
    "cluster": (230, 140, 30),
    "obstacle": (70, 72, 78),
    "exit": (50, 140, 80),
}


def generate_stage2_figures(
    output_dir: str | Path,
    episode: Dict,
    predictions: Dict[str, Dict],
    clusters: Dict[str, Dict],
    metrics_rows: List[Dict],
    training_log: Dict,
) -> List[str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scene = episode["scene"]
    states = episode["states"]
    paths = []
    paths.append(str(save_scene_map(out / "synthetic_scene_map.png", scene)))
    paths.append(str(save_rollout_plot(out / "synthetic_ground_truth_rollout.png", scene, {"ground truth": states[:101]}, title="Synthetic Ground Truth Rollout")))
    overlay = {"ground truth": states[:101]}
    for name in ["constant_velocity_baseline", "hand_physics_baseline", "deterministic_neural_residual"]:
        if name in predictions:
            traj = predictions[name]["trajectories"][0] if predictions[name]["trajectories"].ndim == 4 else predictions[name]["trajectories"]
            overlay[name] = traj
    paths.append(str(save_rollout_plot(out / "prediction_vs_ground_truth_t100.png", scene, overlay, title="Prediction vs Ground Truth t+100")))
    if "physics_plus_neural_residual_SMC" in predictions:
        paths.append(str(save_branch_plot(out / "multi_branch_rollout_t100.png", scene, states[:101], predictions["physics_plus_neural_residual_SMC"]["trajectories"], "Hybrid SMC Branches")))
        paths.append(str(save_branch_plot(out / "representative_cluster_rollouts.png", scene, states[:101], predictions["physics_plus_neural_residual_SMC"]["trajectories"][:8], "Representative Cluster Rollouts")))
    paths.append(str(save_cluster_chart(out / "terminal_clusters_semantic.png", clusters)))
    paths.append(str(save_heatmap(out / "collision_heatmap.png", scene, states[:101], "collision")))
    paths.append(str(save_heatmap(out / "density_heatmap.png", scene, states[:101], "density")))
    paths.append(str(save_heatmap(out / "obstacle_violation_map.png", scene, states[:101], "obstacle")))
    paths.append(str(save_metrics_table(out / "metrics_comparison_table.png", metrics_rows)))
    paths.append(str(save_loss_curves(out / "loss_curves_stage2.png", training_log)))
    paths.append(str(save_rollout_gif(out / "rollout_stage2.gif", scene, states[:101], predictions)))
    return paths


def save_scene_map(path: Path, scene: SceneSpec) -> Path:
    image, draw = base_canvas(scene)
    draw_scene(draw, scene)
    draw_text(draw, (20, 18), "SyntheticPhysicalCrowd2.5D scene map")
    image.save(path)
    return path


def save_rollout_plot(path: Path, scene: SceneSpec, trajectories: Dict[str, np.ndarray], title: str) -> Path:
    image, draw = base_canvas(scene)
    draw_scene(draw, scene)
    colors = [(20, 110, 90), (120, 120, 120), (45, 90, 170), (200, 80, 70), (140, 90, 180)]
    for color, (name, traj) in zip(colors, trajectories.items()):
        draw_trajectory_bundle(draw, scene, traj, color)
        draw_text(draw, (20, 24 + 18 * list(trajectories).index(name)), name, fill=color)
    draw_text(draw, (20, 8), title)
    image.save(path)
    return path


def save_branch_plot(path: Path, scene: SceneSpec, truth: np.ndarray, branches: np.ndarray, title: str) -> Path:
    image, draw = base_canvas(scene)
    draw_scene(draw, scene)
    draw_trajectory_bundle(draw, scene, truth, PALETTE["gt"], width=2)
    for i, branch in enumerate(branches[: min(32, branches.shape[0])]):
        color = (120 + (i * 23) % 120, 70 + (i * 17) % 100, 170)
        draw_trajectory_bundle(draw, scene, branch, color, width=1, every_agent=max(1, branch.shape[1] // 8))
    draw_text(draw, (20, 8), title)
    image.save(path)
    return path


def save_cluster_chart(path: Path, clusters: Dict[str, Dict]) -> Path:
    width, height = 980, 520
    image = Image.new("RGB", (width, height), (248, 248, 245))
    draw = ImageDraw.Draw(image)
    draw_text(draw, (20, 18), "Semantic Terminal Clusters")
    y = 60
    for method, payload in clusters.items():
        draw_text(draw, (20, y), method)
        x = 230
        for cluster in payload.get("clusters", [])[:6]:
            mass = float(cluster["probability_mass"])
            bar = int(320 * mass)
            draw.rectangle((x, y, x + bar, y + 14), fill=PALETTE["cluster"])
            draw_text(draw, (x + bar + 8, y - 2), f"{cluster['semantic_label']} {mass:.2f}")
            y += 22
        y += 18
    image.save(path)
    return path


def save_heatmap(path: Path, scene: SceneSpec, trajectory: np.ndarray, mode: str) -> Path:
    image, draw = base_canvas(scene)
    draw_scene(draw, scene)
    bins = np.zeros((32, 52), dtype=np.float32)
    for frame in trajectory:
        if mode == "collision":
            min_gap, _ = min_gap_and_collisions(frame)
            weight = 1.0 if min_gap < 0.2 else 0.15
        elif mode == "density":
            dens = local_density(frame)
            weight = float(np.clip(np.mean(dens) / 5.0, 0.1, 1.0))
        else:
            weight = 1.0
        for agent in frame:
            bx = int(np.clip(agent[0] / scene.width * bins.shape[1], 0, bins.shape[1] - 1))
            by = int(np.clip(agent[1] / scene.height * bins.shape[0], 0, bins.shape[0] - 1))
            bins[by, bx] += weight
    bins /= max(1e-6, bins.max())
    for by in range(bins.shape[0]):
        for bx in range(bins.shape[1]):
            if bins[by, bx] <= 0.03:
                continue
            x1 = int(bx / bins.shape[1] * 900) + 50
            y1 = int(by / bins.shape[0] * 560) + 40
            val = int(210 * bins[by, bx])
            draw.rectangle((x1, y1, x1 + 16, y1 + 16), fill=(255, 210 - val // 2, 80))
    draw_text(draw, (20, 8), mode.replace("_", " ").title())
    image.save(path)
    return path


def save_metrics_table(path: Path, rows: List[Dict]) -> Path:
    image = Image.new("RGB", (1200, 720), (250, 250, 247))
    draw = ImageDraw.Draw(image)
    draw_text(draw, (20, 18), "Stage 2 Metrics Comparison")
    columns = ["model", "ADE@100", "FDE@100", "best_of_64_FDE@100", "collision", "obstacle", "coverage@64", "event_acc"]
    x_positions = [20, 270, 390, 510, 700, 820, 940, 1060]
    y = 60
    for x, col in zip(x_positions, columns):
        draw_text(draw, (x, y), col)
    y += 26
    for row in rows[:18]:
        values = [
            row.get("model", ""),
            row.get("ADE@100", ""),
            row.get("FDE@100", ""),
            row.get("best_of_64_FDE@100", ""),
            row.get("collision_violation_rate", ""),
            row.get("obstacle_violation_rate", ""),
            row.get("coverage@64", ""),
            row.get("semantic_event_accuracy", ""),
        ]
        for x, val in zip(x_positions, values):
            draw_text(draw, (x, y), str(val)[:28])
        y += 24
    image.save(path)
    return path


def save_loss_curves(path: Path, training_log: Dict) -> Path:
    image = Image.new("RGB", (940, 520), (249, 249, 246))
    draw = ImageDraw.Draw(image)
    draw_text(draw, (20, 18), "Loss Curves Stage 2")
    x0, y0, w, h = 70, 80, 800, 360
    draw.rectangle((x0, y0, x0 + w, y0 + h), outline=(80, 80, 80))
    colors = [(200, 80, 70), (45, 90, 170)]
    for color, key in zip(colors, ["deterministic_neural_residual", "stochastic_neural_residual"]):
        history = training_log.get(key, {}).get("loss_components", [])
        if not history:
            continue
        vals = np.asarray([row["val_total_loss"] for row in history], dtype=float)
        vals = (vals - vals.min()) / max(1e-6, vals.max() - vals.min())
        points = []
        for i, value in enumerate(vals):
            x = x0 + int((i / max(1, len(vals) - 1)) * w)
            y = y0 + h - int(value * (h - 20)) - 10
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
        draw_text(draw, (x0, y0 + h + 28 + 22 * colors.index(color)), key, fill=color)
    image.save(path)
    return path


def save_rollout_gif(path: Path, scene: SceneSpec, truth: np.ndarray, predictions: Dict[str, Dict]) -> Path:
    frames = []
    learned = predictions.get("deterministic_neural_residual", {}).get("trajectories")
    learned = learned[0] if learned is not None and learned.ndim == 4 else learned
    for t in range(0, min(101, truth.shape[0]), 5):
        image, draw = base_canvas(scene)
        draw_scene(draw, scene)
        draw_agents(draw, scene, truth[t], PALETTE["gt"])
        if learned is not None and t < learned.shape[0]:
            draw_agents(draw, scene, learned[t], PALETTE["learned"])
        draw_text(draw, (20, 8), f"Rollout t+{t}")
        frames.append(image)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=110, loop=0)
    return path


def base_canvas(scene: SceneSpec) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (1000, 640), (244, 243, 238))
    return image, ImageDraw.Draw(image)


def world_to_px(scene: SceneSpec, point: np.ndarray | tuple[float, float]) -> tuple[int, int]:
    x = int(50 + float(point[0]) / scene.width * 900)
    y = int(40 + float(point[1]) / scene.height * 560)
    return x, y


def draw_scene(draw: ImageDraw.ImageDraw, scene: SceneSpec) -> None:
    draw.rectangle((50, 40, 950, 600), outline=(40, 40, 40), width=2)
    for rect in scene.obstacles:
        x1, y1 = world_to_px(scene, (rect.x1, rect.y1))
        x2, y2 = world_to_px(scene, (rect.x2, rect.y2))
        draw.rectangle((x1, y1, x2, y2), fill=PALETTE["obstacle"])
    for name, point in scene.exits.items():
        x, y = world_to_px(scene, point)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=PALETTE["exit"])
        draw_text(draw, (x + 10, y - 8), name, fill=PALETTE["exit"])


def draw_trajectory_bundle(draw: ImageDraw.ImageDraw, scene: SceneSpec, trajectory: np.ndarray, color: tuple[int, int, int], width: int = 2, every_agent: int = 1) -> None:
    for agent in range(0, trajectory.shape[1], every_agent):
        points = [world_to_px(scene, p) for p in trajectory[:, agent, :2]]
        if len(points) > 1:
            draw.line(points, fill=color, width=width)
        x, y = points[-1]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)


def draw_agents(draw: ImageDraw.ImageDraw, scene: SceneSpec, frame: np.ndarray, color: tuple[int, int, int]) -> None:
    for agent in frame:
        x, y = world_to_px(scene, agent[:2])
        r = max(3, int(agent[7] / scene.width * 900))
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int] = (35, 35, 35)) -> None:
    try:
        font = ImageFont.truetype("Arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    draw.text(xy, text, fill=fill, font=font)
