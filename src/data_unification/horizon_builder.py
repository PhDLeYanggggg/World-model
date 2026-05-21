from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def available_stage5b_world_state(root: str | Path = "data/stage5b_world_state") -> List[str]:
    base = Path(root)
    if not base.exists():
        return []
    return sorted(path.name for path in base.iterdir() if (path / "world_state.csv").exists())


def horizon_audit_dataset(dataset: str, root: str | Path = "data/stage5b_world_state") -> Dict:
    table = pd.read_csv(Path(root) / dataset / "world_state.csv")
    tracks = table.sort_values(["scene_id", "agent_id", "frame_id"]).groupby(["scene_id", "agent_id"])
    lengths = tracks["frame_id"].nunique()
    dts = table["dt_s"].replace([np.inf, -np.inf], np.nan).dropna()
    dt = float(dts[dts > 0].median()) if len(dts[dts > 0]) else 1.0
    max_raw = int(max(0, lengths.max() - 10)) if len(lengths) else 0
    raw_horizons = {h: int((lengths >= 10 + h).sum()) for h in [10, 25, 50, 100]}
    downsample = {}
    for factor in [2, 4]:
        effective_len = np.floor(lengths / factor)
        downsample[str(factor)] = {
            "dt_s": dt * factor,
            "max_horizon_steps": int(max(0, effective_len.max() - 10)) if len(effective_len) else 0,
            "samples_t50": int((effective_len >= 60).sum()),
            "samples_t100": int((effective_len >= 110).sum()),
            "effective_time_t100_s": round(100 * dt * factor, 3),
            "warning": "Downsampling changes temporal resolution; it is legal for evaluation only if reported as a different physical horizon.",
        }
    return {
        "dataset_name": dataset,
        "coordinate_unit": str(table["coordinate_unit"].iloc[0]) if len(table) else "unknown",
        "dt_median_s": round(dt, 6),
        "track_count": int(len(lengths)),
        "mean_track_length": round(float(lengths.mean()), 3) if len(lengths) else 0.0,
        "p95_track_length": round(float(lengths.quantile(0.95)), 3) if len(lengths) else 0.0,
        "max_track_length": int(lengths.max()) if len(lengths) else 0,
        "max_raw_horizon_steps_after_past10": max_raw,
        "raw_samples": raw_horizons,
        "physical_time_span_s": {f"t+{h}": round(h * dt, 3) for h in [10, 25, 50, 100]},
        "downsample_options": downsample,
        "supports_raw_t50": raw_horizons[50] > 0,
        "supports_raw_t100": raw_horizons[100] > 0,
        "t100_meaningful": raw_horizons[100] > 0 and max_raw >= 100,
    }


def write_horizon_report(rows: Iterable[Dict], path: str | Path = "outputs/reports/stage5b5_horizon_audit.md") -> List[Dict]:
    audits = list(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    lines = [
        "# Stage 5B.5 Horizon Audit",
        "",
        "Downsampling is treated as a different effective physical horizon. It is not reported as raw t+100 unless explicitly supported by contiguous source tracks.",
        "",
        "| dataset | dt_s | max_track | max_raw_horizon | raw_t50_tracks | raw_t100_tracks | supports_raw_t50 | supports_raw_t100 | t+100 seconds | note |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for row in audits:
        note = "raw long horizon available" if row["supports_raw_t100"] else "no raw t+100; do not fake by stitching"
        lines.append(
            f"| {row['dataset_name']} | {row['dt_median_s']} | {row['max_track_length']} | {row['max_raw_horizon_steps_after_past10']} | "
            f"{row['raw_samples'][50]} | {row['raw_samples'][100]} | {row['supports_raw_t50']} | {row['supports_raw_t100']} | "
            f"{row['physical_time_span_s']['t+100']} | {note} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audits
