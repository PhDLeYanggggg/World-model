#!/usr/bin/env python3
"""Train a tiny top-down MOBA-like world model and run branched rollouts.

This is intentionally a compact proof-of-concept:
- public Dota 2 top-down ticks from wolframko/betty-dota2
- one-step transition model trained with ridge regression
- t+100 stochastic beam rollout
- terminal-state clustering into likely outcomes

It is not a neural world model yet. It is a calibrated transition baseline that
lets us test the branch-predict-and-cluster idea against real replay statistics.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download


DATASET = "wolframko/betty-dota2"
TICKS_FILE = "ticks/shard_0000.parquet"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "hf" / "betty-dota2"
OUT_DIR = ROOT / "experiments" / "outputs" / "moba_branch_world_model"
MAP_MIN = 0.0
MAP_MAX = 16000.0
MAP_CENTER = np.array([8000.0, 8000.0])


FEATURES = [
    "bias",
    "x_norm",
    "y_norm",
    "vx_norm",
    "vy_norm",
    "hp_pct",
    "mana_pct",
    "net_worth_norm",
    "xp_norm",
    "level_norm",
    "move_speed_norm",
    "alive",
    "team",
    "time_norm",
    "toward_center_x",
    "toward_center_y",
]

TARGETS = ["dx_norm", "dy_norm", "dhp", "dmana", "alive_next"]


@dataclass
class TransitionModel:
    weights: np.ndarray
    residual_std: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    metrics: Dict[str, float]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = hf_hub_download(
        repo_id=DATASET,
        repo_type="dataset",
        filename=TICKS_FILE,
        local_dir=DATA_DIR,
    )

    df = load_ticks(Path(parquet_path))
    train_df, holdout_df = split_matches(df)
    train_rows = build_transition_rows(train_df)
    valid_rows = build_transition_rows(holdout_df)
    model = train_transition_model(train_rows, valid_rows)

    start_time = choose_start_time(holdout_df)
    initial_state = state_at_time(holdout_df, start_time)
    actual_future = actual_future_summary(holdout_df, start_time, horizon=100)
    prediction = branch_rollout(initial_state, model, horizon=100, seed=17)
    prediction["actual_future"] = actual_future
    prediction["dataset"] = {
        "source": DATASET,
        "file": TICKS_FILE,
        "rows": int(len(df)),
        "matches": int(df["match_id"].nunique()),
        "train_matches": int(train_df["match_id"].nunique()),
        "holdout_match": int(holdout_df["match_id"].iloc[0]),
        "start_time": float(start_time),
    }
    prediction["model"] = model.metrics

    summary_path = OUT_DIR / "summary.json"
    svg_path = OUT_DIR / "rollout.svg"
    model_path = OUT_DIR / "model_weights.json"
    summary_path.write_text(json.dumps(prediction, indent=2), encoding="utf-8")
    model_path.write_text(
        json.dumps(
            {
                "features": FEATURES,
                "targets": TARGETS,
                "weights": model.weights.tolist(),
                "residual_std": model.residual_std.tolist(),
                "feature_mean": model.feature_mean.tolist(),
                "feature_scale": model.feature_scale.tolist(),
                "target_mean": model.target_mean.tolist(),
                "target_scale": model.target_scale.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    render_svg(svg_path, initial_state, prediction, actual_future)

    print(json.dumps({"summary": str(summary_path), "svg": str(svg_path), "model": str(model_path)}, indent=2))
    print(json.dumps(key_console_summary(prediction), indent=2))


def load_ticks(path: Path) -> pd.DataFrame:
    columns = [
        "match_id",
        "game_time",
        "slot",
        "hero",
        "x",
        "y",
        "hp",
        "max_hp",
        "mana",
        "max_mana",
        "net_worth",
        "xp",
        "is_alive",
        "level",
        "move_speed",
    ]
    df = pd.read_parquet(path, columns=columns)
    df = df.dropna(subset=["x", "y", "game_time", "slot", "match_id"]).copy()
    # This dataset normalizes players to slots 0-9: 0-4 Radiant, 5-9 Dire.
    df["team"] = (df["slot"].astype(int) >= 5).astype(int)
    df["unit_id"] = df["match_id"].astype(str) + ":" + df["slot"].astype(str)
    df["hp_pct"] = safe_ratio(df["hp"], df["max_hp"])
    df["mana_pct"] = safe_ratio(df["mana"], df["max_mana"])
    df["alive"] = df["is_alive"].astype(float)
    df["x"] = df["x"].clip(MAP_MIN, MAP_MAX)
    df["y"] = df["y"].clip(MAP_MIN, MAP_MAX)
    return df.sort_values(["match_id", "slot", "game_time"]).reset_index(drop=True)


def split_matches(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    matches = sorted(df["match_id"].unique())
    holdout = matches[-1]
    return df[df["match_id"] != holdout].copy(), df[df["match_id"] == holdout].copy()


def build_transition_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in df.groupby("unit_id", sort=False):
        group = group.sort_values("game_time").copy()
        group["prev_x"] = group["x"].shift(1)
        group["prev_y"] = group["y"].shift(1)
        group["next_x"] = group["x"].shift(-1)
        group["next_y"] = group["y"].shift(-1)
        group["next_hp_pct"] = group["hp_pct"].shift(-1)
        group["next_mana_pct"] = group["mana_pct"].shift(-1)
        group["next_alive"] = group["alive"].shift(-1)
        group["next_time"] = group["game_time"].shift(-1)
        group["prev_time"] = group["game_time"].shift(1)
        group["dt_next"] = group["next_time"] - group["game_time"]
        group["dt_prev"] = group["game_time"] - group["prev_time"]
        group = group[(group["dt_next"] > 0.4) & (group["dt_next"] < 3.0)].copy()
        group["vx"] = ((group["x"] - group["prev_x"]) / group["dt_prev"]).fillna(0)
        group["vy"] = ((group["y"] - group["prev_y"]) / group["dt_prev"]).fillna(0)
        group["dx"] = group["next_x"] - group["x"]
        group["dy"] = group["next_y"] - group["y"]
        rows.append(group)

    if not rows:
        raise RuntimeError("No transition rows were built from the ticks data.")
    transitions = pd.concat(rows, ignore_index=True)
    transitions = transitions.replace([np.inf, -np.inf], np.nan).dropna(subset=["dx", "dy", "next_hp_pct", "next_mana_pct"])
    return transitions


def train_transition_model(train_rows: pd.DataFrame, valid_rows: pd.DataFrame) -> TransitionModel:
    x_train = make_features(train_rows)
    y_train = make_targets(train_rows)
    x_valid = make_features(valid_rows)
    y_valid = make_targets(valid_rows)

    feature_mean = x_train.mean(axis=0)
    feature_scale = x_train.std(axis=0) + 1e-6
    target_mean = y_train.mean(axis=0)
    target_scale = y_train.std(axis=0) + 1e-6
    x_train_n = (x_train - feature_mean) / feature_scale
    y_train_n = (y_train - target_mean) / target_scale
    x_valid_n = (x_valid - feature_mean) / feature_scale

    reg = 0.06
    identity = np.eye(x_train_n.shape[1])
    weights = np.linalg.solve(x_train_n.T @ x_train_n + reg * identity, x_train_n.T @ y_train_n)
    pred_train = ((x_train_n @ weights) * target_scale) + target_mean
    pred_valid = ((x_valid_n @ weights) * target_scale) + target_mean
    residual = y_train - pred_train
    residual_std = np.clip(residual.std(axis=0), [0.0006, 0.0006, 0.003, 0.003, 0.01], [0.08, 0.08, 0.2, 0.2, 0.45])

    valid_error = y_valid - pred_valid
    rmse = np.sqrt(np.mean(valid_error[:, :4] ** 2, axis=0))
    alive_pred = (pred_valid[:, 4] > 0.5).astype(float)
    alive_acc = float((alive_pred == (y_valid[:, 4] > 0.5)).mean())
    metrics = {
        "train_rows": int(len(train_rows)),
        "valid_rows": int(len(valid_rows)),
        "position_rmse_map_units": float(np.sqrt(np.mean((valid_error[:, :2] * MAP_MAX) ** 2))),
        "dx_rmse": float(rmse[0]),
        "dy_rmse": float(rmse[1]),
        "hp_delta_rmse": float(rmse[2]),
        "mana_delta_rmse": float(rmse[3]),
        "alive_accuracy": alive_acc,
        "residual_dx_map_units": float(residual_std[0] * MAP_MAX),
        "residual_dy_map_units": float(residual_std[1] * MAP_MAX),
    }
    return TransitionModel(weights, residual_std, feature_mean, feature_scale, target_mean, target_scale, metrics)


def make_features(rows: pd.DataFrame) -> np.ndarray:
    x = rows["x"].to_numpy(dtype=float)
    y = rows["y"].to_numpy(dtype=float)
    vectors_to_center = MAP_CENTER.reshape(1, 2) - np.stack([x, y], axis=1)
    norms = np.linalg.norm(vectors_to_center, axis=1, keepdims=True) + 1e-6
    center_dirs = vectors_to_center / norms
    features = np.column_stack(
        [
            np.ones(len(rows)),
            x / MAP_MAX,
            y / MAP_MAX,
            rows["vx"].to_numpy(dtype=float) / 900,
            rows["vy"].to_numpy(dtype=float) / 900,
            rows["hp_pct"].to_numpy(dtype=float),
            rows["mana_pct"].to_numpy(dtype=float),
            rows["net_worth"].fillna(0).to_numpy(dtype=float) / 50000,
            rows["xp"].fillna(0).to_numpy(dtype=float) / 40000,
            rows["level"].fillna(1).to_numpy(dtype=float) / 30,
            rows["move_speed"].fillna(300).to_numpy(dtype=float) / 600,
            rows["alive"].to_numpy(dtype=float),
            rows["team"].to_numpy(dtype=float),
            np.clip(rows["game_time"].to_numpy(dtype=float), 0, 3600) / 3600,
            center_dirs[:, 0],
            center_dirs[:, 1],
        ]
    )
    return features


def make_targets(rows: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            rows["dx"].to_numpy(dtype=float) / MAP_MAX,
            rows["dy"].to_numpy(dtype=float) / MAP_MAX,
            rows["next_hp_pct"].to_numpy(dtype=float) - rows["hp_pct"].to_numpy(dtype=float),
            rows["next_mana_pct"].to_numpy(dtype=float) - rows["mana_pct"].to_numpy(dtype=float),
            rows["next_alive"].to_numpy(dtype=float),
        ]
    )


def choose_start_time(df: pd.DataFrame) -> float:
    times = sorted(df["game_time"].unique())
    candidates = [time for time in times if 600 <= time <= 1200]
    return candidates[len(candidates) // 2] if candidates else times[len(times) // 3]


def state_at_time(df: pd.DataFrame, time: float) -> List[Dict[str, float]]:
    idx = (df["game_time"] - time).abs().groupby(df["slot"]).idxmin()
    rows = df.loc[idx].sort_values("slot")
    state = []
    for _, row in rows.iterrows():
      state.append(row_to_unit(row, vx=0.0, vy=0.0))

    # Estimate velocity from the previous observation for each unit.
    for unit in state:
        history = df[(df["slot"] == unit["slot"]) & (df["game_time"] <= time)].sort_values("game_time").tail(2)
        if len(history) == 2:
            before, now = history.iloc[0], history.iloc[1]
            dt = max(0.1, now["game_time"] - before["game_time"])
            unit["vx"] = float((now["x"] - before["x"]) / dt)
            unit["vy"] = float((now["y"] - before["y"]) / dt)
    return state


def row_to_unit(row: pd.Series, vx: float, vy: float) -> Dict[str, float]:
    return {
        "slot": int(row["slot"]),
        "hero": str(row["hero"]),
        "team": int(row["team"]),
        "x": float(row["x"]),
        "y": float(row["y"]),
        "z": 0.0,
        "vx": float(vx),
        "vy": float(vy),
        "hp_pct": float(row["hp_pct"]),
        "mana_pct": float(row["mana_pct"]),
        "net_worth": float(row["net_worth"] or 0),
        "xp": float(row["xp"] or 0),
        "level": float(row["level"] or 1),
        "move_speed": float(row["move_speed"] or 300),
        "alive": float(row["alive"]),
    }


def branch_rollout(initial_state: List[Dict[str, float]], model: TransitionModel, horizon: int, seed: int) -> Dict:
    rng = random.Random(seed)
    beam = [{"state": clone_state(initial_state), "logp": 0.0, "trace": [team_centers(initial_state)]}]
    samples_per_step = 10
    local_top_k = 3
    beam_width = 96

    for step in range(horizon):
        expanded = []
        for path in beam:
            base_pred = predict_unit_deltas(path["state"], model)
            branches = make_branches(base_pred, model, rng, samples_per_step)[:local_top_k]
            for branch in branches:
                next_state = apply_branch(path["state"], branch["deltas"])
                expanded.append(
                    {
                        "state": next_state,
                        "logp": path["logp"] + math.log(branch["probability"]),
                        "trace": append_trace(path["trace"], next_state, step, horizon),
                    }
                )
        expanded.sort(key=lambda item: item["logp"], reverse=True)
        beam = expanded[:beam_width]

    weighted_paths = normalize_paths(beam)
    outcomes = cluster_terminal_states(weighted_paths)
    return {
        "config": {
            "horizon": horizon,
            "samples_per_step": samples_per_step,
            "local_top_k": local_top_k,
            "beam_width": beam_width,
            "paths_kept": len(weighted_paths),
        },
        "outcomes": outcomes,
        "top_paths": [
            {
                "probability": round(path["probability"], 5),
                "terminal": terminal_summary(path["state"]),
                "trace": path["trace"],
            }
            for path in weighted_paths[:8]
        ],
    }


def predict_unit_deltas(state: List[Dict[str, float]], model: TransitionModel) -> np.ndarray:
    rows = pd.DataFrame(
        [
            {
                "x": unit["x"],
                "y": unit["y"],
                "vx": unit["vx"],
                "vy": unit["vy"],
                "hp_pct": unit["hp_pct"],
                "mana_pct": unit["mana_pct"],
                "net_worth": unit["net_worth"],
                "xp": unit["xp"],
                "level": unit["level"],
                "move_speed": unit["move_speed"],
                "alive": unit["alive"],
                "team": unit["team"],
                "game_time": 900,
            }
            for unit in state
        ]
    )
    features = make_features(rows)
    normalized = (features - model.feature_mean) / model.feature_scale
    return ((normalized @ model.weights) * model.target_scale) + model.target_mean


def make_branches(base_pred: np.ndarray, model: TransitionModel, rng: random.Random, samples_per_step: int) -> List[Dict]:
    branches = []
    patterns = [
        ("central", 0.0, 0.34),
        ("aggressive-east", 0.75, 0.23),
        ("defensive-west", -0.75, 0.21),
    ]
    while len(patterns) < samples_per_step:
        z = rng.gauss(0, 1.0)
        patterns.append((f"sample-{len(patterns)}", z, math.exp(-0.5 * z * z) * 0.08))

    total = sum(weight for _, _, weight in patterns)
    for _, z, weight in sorted(patterns, key=lambda item: item[2], reverse=True):
        noise = np.zeros_like(base_pred)
        for unit_index in range(len(base_pred)):
            direction = 1 if unit_index % 2 == 0 else -1
            noise[unit_index, 0] = z * model.residual_std[0] * direction
            noise[unit_index, 1] = z * model.residual_std[1] * (1 if unit_index % 3 == 0 else -1)
            noise[unit_index, 2] = -abs(z) * model.residual_std[2] * (0.25 if unit_index % 4 == 0 else 0.08)
            noise[unit_index, 3] = z * model.residual_std[3] * 0.15
            noise[unit_index, 4] = -abs(z) * model.residual_std[4] * 0.08
        branches.append({"probability": weight / total, "deltas": base_pred + noise})
    return branches


def apply_branch(state: List[Dict[str, float]], deltas: np.ndarray) -> List[Dict[str, float]]:
    next_state = []
    for unit, delta in zip(state, deltas):
        x = float(np.clip(unit["x"] + delta[0] * MAP_MAX, MAP_MIN, MAP_MAX))
        y = float(np.clip(unit["y"] + delta[1] * MAP_MAX, MAP_MIN, MAP_MAX))
        next_unit = dict(unit)
        next_unit["vx"] = x - unit["x"]
        next_unit["vy"] = y - unit["y"]
        next_unit["x"] = x
        next_unit["y"] = y
        next_unit["z"] = float(max(0.0, 48.0 * (1.0 - next_unit["hp_pct"])))
        next_unit["hp_pct"] = float(np.clip(unit["hp_pct"] + delta[2], 0, 1))
        next_unit["mana_pct"] = float(np.clip(unit["mana_pct"] + delta[3], 0, 1))
        alive_score = float(np.clip(delta[4], 0, 1))
        next_unit["alive"] = 1.0 if alive_score > 0.35 and next_unit["hp_pct"] > 0.03 else 0.0
        if next_unit["alive"] < 0.5:
            next_unit["hp_pct"] = 0.0
        next_state.append(next_unit)
    return next_state


def cluster_terminal_states(paths: List[Dict]) -> List[Dict]:
    clusters: Dict[str, Dict] = {}
    for path in paths:
        summary = terminal_summary(path["state"])
        key = "|".join(
            [
                summary["winner_hint"],
                summary["fight_state"],
                summary["map_control"],
                str(int(summary["radiant_center"][0] // 2200)),
                str(int(summary["dire_center"][0] // 2200)),
            ]
        )
        cluster = clusters.setdefault(
            key,
            {"probability": 0.0, "paths": 0, "representative": summary, "best": 0.0},
        )
        cluster["probability"] += path["probability"]
        cluster["paths"] += 1
        if path["probability"] > cluster["best"]:
            cluster["best"] = path["probability"]
            cluster["representative"] = summary

    return [
        {
            "label": describe_cluster(cluster["representative"]),
            "probability": round(cluster["probability"], 4),
            "paths": cluster["paths"],
            "representative": cluster["representative"],
        }
        for cluster in sorted(clusters.values(), key=lambda item: item["probability"], reverse=True)[:6]
    ]


def terminal_summary(state: List[Dict]) -> Dict:
    radiant = [unit for unit in state if unit["team"] == 0]
    dire = [unit for unit in state if unit["team"] == 1]
    radiant_alive = sum(unit["alive"] > 0.5 for unit in radiant)
    dire_alive = sum(unit["alive"] > 0.5 for unit in dire)
    radiant_hp = sum(unit["hp_pct"] for unit in radiant)
    dire_hp = sum(unit["hp_pct"] for unit in dire)
    radiant_center = center(radiant)
    dire_center = center(dire)
    distance = float(np.linalg.norm(np.array(radiant_center) - np.array(dire_center)))
    if radiant_alive - dire_alive >= 2 or radiant_hp - dire_hp > 1.3:
        winner_hint = "radiant"
    elif dire_alive - radiant_alive >= 2 or dire_hp - radiant_hp > 1.3:
        winner_hint = "dire"
    else:
        winner_hint = "contested"
    fight_state = "teamfight" if distance < 2600 else "split"
    map_control = "radiant-advanced" if radiant_center[0] + radiant_center[1] > dire_center[0] + dire_center[1] else "dire-advanced"
    return {
        "radiant_alive": int(radiant_alive),
        "dire_alive": int(dire_alive),
        "radiant_hp_sum": round(float(radiant_hp), 3),
        "dire_hp_sum": round(float(dire_hp), 3),
        "radiant_center": [round(float(v), 1) for v in radiant_center],
        "dire_center": [round(float(v), 1) for v in dire_center],
        "team_distance": round(distance, 1),
        "winner_hint": winner_hint,
        "fight_state": fight_state,
        "map_control": map_control,
    }


def actual_future_summary(df: pd.DataFrame, start_time: float, horizon: int) -> Dict:
    target_time = start_time + horizon
    state = state_at_time(df, target_time)
    return terminal_summary(state)


def normalize_paths(paths: List[Dict]) -> List[Dict]:
    max_log = max(path["logp"] for path in paths)
    weights = [math.exp(path["logp"] - max_log) for path in paths]
    total = sum(weights)
    for path, weight in zip(paths, weights):
        path["probability"] = weight / total
    return sorted(paths, key=lambda item: item["probability"], reverse=True)


def team_centers(state: List[Dict]) -> Dict:
    radiant = center([unit for unit in state if unit["team"] == 0])
    dire = center([unit for unit in state if unit["team"] == 1])
    return {"radiant": [round(radiant[0], 1), round(radiant[1], 1)], "dire": [round(dire[0], 1), round(dire[1], 1)]}


def append_trace(trace: List[Dict], state: List[Dict], step: int, horizon: int) -> List[Dict]:
    if step % 10 == 0 or step == horizon - 1:
        return trace + [team_centers(state)]
    return trace


def center(units: Iterable[Dict]) -> Tuple[float, float]:
    units = list(units)
    if not units:
        return (MAP_CENTER[0], MAP_CENTER[1])
    weights = np.array([max(0.05, unit["hp_pct"]) for unit in units])
    points = np.array([[unit["x"], unit["y"]] for unit in units])
    center_point = (points * weights.reshape(-1, 1)).sum(axis=0) / weights.sum()
    return (float(center_point[0]), float(center_point[1]))


def describe_cluster(summary: Dict) -> str:
    if summary["winner_hint"] == "radiant":
        return "radiant-favored branch after 100s"
    if summary["winner_hint"] == "dire":
        return "dire-favored branch after 100s"
    if summary["fight_state"] == "teamfight":
        return "contested teamfight remains likely"
    return "split-map neutral branch"


def render_svg(path: Path, initial_state: List[Dict], prediction: Dict, actual: Dict) -> None:
    width, height = 900, 900

    def sx(x: float) -> float:
        return 40 + (x / MAP_MAX) * 820

    def sy(y: float) -> float:
        return 860 - (y / MAP_MAX) * 820

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="900" height="900" fill="#f7fafc"/>',
        '<rect x="40" y="40" width="820" height="820" fill="#eef4f2" stroke="#9aa8a8" stroke-width="2"/>',
        '<line x1="40" y1="860" x2="860" y2="40" stroke="#b6c5bc" stroke-width="10" opacity="0.35"/>',
        '<line x1="40" y1="40" x2="860" y2="860" stroke="#c9d4dd" stroke-width="2" opacity="0.35"/>',
        '<text x="42" y="26" font-family="Inter,Arial" font-size="18" font-weight="700" fill="#17202a">MOBA Branch World Model: t+100 rollout</text>',
        '<text x="42" y="884" font-family="Inter,Arial" font-size="11" fill="#526171">blue=radiant, amber=dire, thin paths=top beam traces, X=actual replay t+100 center</text>',
    ]
    for i in range(1, 8):
        pos = 40 + i * 102.5
        lines.append(f'<line x1="{pos}" y1="40" x2="{pos}" y2="860" stroke="#d6dee6" stroke-width="1"/>')
        lines.append(f'<line x1="40" y1="{pos}" x2="860" y2="{pos}" stroke="#d6dee6" stroke-width="1"/>')

    for unit in initial_state:
        color = "#2f80ed" if unit["team"] == 0 else "#d9902f"
        lines.append(f'<circle cx="{sx(unit["x"]):.1f}" cy="{sy(unit["y"]):.1f}" r="7" fill="{color}" stroke="#17202a" stroke-width="1"/>')
        lines.append(f'<text x="{sx(unit["x"]) + 9:.1f}" y="{sy(unit["y"]) + 4:.1f}" font-family="Inter,Arial" font-size="9" fill="#17202a">{escape(unit["hero"][:10])}</text>')

    colors = ["#0f766e", "#4c78a8", "#e29535", "#8f63a8", "#64748b", "#d95f5f", "#2f9d55", "#b08900"]
    for idx, path_item in enumerate(prediction["top_paths"]):
        color = colors[idx % len(colors)]
        for team in ["radiant", "dire"]:
            points = " ".join(f'{sx(p[team][0]):.1f},{sy(p[team][1]):.1f}' for p in path_item["trace"])
            opacity = 0.55 if team == "radiant" else 0.36
            dash = "" if team == "radiant" else ' stroke-dasharray="5 5"'
            lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" opacity="{opacity}"{dash}/>')

    actual_rad = actual["radiant_center"]
    actual_dire = actual["dire_center"]
    lines.append(cross(sx(actual_rad[0]), sy(actual_rad[1]), "#1d4ed8"))
    lines.append(cross(sx(actual_dire[0]), sy(actual_dire[1]), "#b45309"))

    y = 66
    for outcome in prediction["outcomes"][:5]:
        lines.append(f'<rect x="575" y="{y - 16}" width="265" height="38" rx="6" fill="#ffffff" stroke="#d5dce4"/>')
        lines.append(f'<text x="588" y="{y}" font-family="Inter,Arial" font-size="11" font-weight="700" fill="#17202a">{escape(outcome["label"])}</text>')
        lines.append(f'<text x="588" y="{y + 15}" font-family="Inter,Arial" font-size="10" fill="#526171">p={outcome["probability"]:.3f} · paths={outcome["paths"]}</text>')
        y += 46

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def cross(x: float, y: float, color: str) -> str:
    return (
        f'<g stroke="{color}" stroke-width="4" stroke-linecap="round">'
        f'<line x1="{x-9:.1f}" y1="{y-9:.1f}" x2="{x+9:.1f}" y2="{y+9:.1f}"/>'
        f'<line x1="{x-9:.1f}" y1="{y+9:.1f}" x2="{x+9:.1f}" y2="{y-9:.1f}"/>'
        "</g>"
    )


def key_console_summary(prediction: Dict) -> Dict:
    return {
        "model": prediction["model"],
        "outcomes": prediction["outcomes"],
        "actual_future": prediction["actual_future"],
    }


def clone_state(state: List[Dict]) -> List[Dict]:
    return [dict(unit) for unit in state]


def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.astype(float) / b.replace(0, np.nan).astype(float)).fillna(0).clip(0, 1)


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    main()
