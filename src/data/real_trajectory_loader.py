from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.data.eth_ucy_loader import load_eth_ucy_trajectories
from src.data.ngsim_loader import load_ngsim_trajectories
from src.data.opendd_loader import load_opendd_trajectories
from src.data.sdd_loader import load_sdd_trajectories
from src.data.tgsim_loader import load_tgsim_trajectories
from src.data.trajnet_loader import load_trajnet_trajectories


REQUIRED_REAL_COLUMNS = [
    "episode_id",
    "scene_id",
    "frame_id",
    "agent_id",
    "x",
    "y",
    "vx",
    "vy",
    "ax",
    "ay",
    "heading",
    "agent_type",
    "valid",
]

OPTIONAL_REAL_COLUMNS = [
    "time",
    "dt",
    "native_vx",
    "native_vy",
    "native_ax",
    "native_ay",
    "causal_vx",
    "causal_vy",
    "causal_ax",
    "causal_ay",
    "central_vx",
    "central_vy",
    "central_ax",
    "central_ay",
]


def load_real_trajectory_table(dataset: str, data_path: str | Path, quick: bool = False, max_rows: int | None = None) -> Tuple[pd.DataFrame, Dict]:
    dataset = dataset.lower()
    if dataset in {"tgsim", "tgsim_i90", "tgsim_other"}:
        table, meta = load_tgsim_trajectories(data_path, quick=quick, max_rows=max_rows)
        if dataset != "tgsim":
            meta["dataset_name"] = "TGSIM additional public corridor"
            table["episode_id"] = dataset
            table["scene_id"] = dataset
    elif dataset == "trajnet":
        table, meta = load_trajnet_trajectories(data_path, quick=quick, max_rows=max_rows)
    elif dataset in {"eth_ucy", "eth", "ucy"}:
        table, meta = load_eth_ucy_trajectories(data_path, quick=quick, max_rows=max_rows)
    elif dataset == "sdd":
        table, meta = load_sdd_trajectories(data_path, quick=quick, max_rows=max_rows)
    elif dataset == "opendd":
        table, meta = load_opendd_trajectories(data_path, quick=quick, max_rows=max_rows)
    elif dataset == "ngsim":
        table, meta = load_ngsim_trajectories(data_path, quick=quick, max_rows=max_rows)
    else:
        raise ValueError(f"Unsupported real dataset `{dataset}`. Use tgsim, trajnet, eth_ucy, sdd, opendd, or ngsim.")
    missing = [column for column in REQUIRED_REAL_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"Loader `{dataset}` did not produce required columns: {missing}")
    keep = REQUIRED_REAL_COLUMNS + [column for column in OPTIONAL_REAL_COLUMNS if column in table.columns]
    table = table[keep].copy()
    table["valid"] = table["valid"].astype(bool)
    return table.sort_values(["scene_id", "frame_id", "agent_id"]).reset_index(drop=True), meta


def missing_data_error(dataset: str) -> str:
    return (
        f"No data path was provided for `{dataset}`. Provide a local dataset path, e.g.\n\n"
        f"python run_stage4_real_benchmark.py --dataset {dataset} --data /path/to/data --quick\n\n"
        "Stage 4 does not use AerialMPT bauma3 t+100 because that sequence has no t+100 ground truth."
    )
