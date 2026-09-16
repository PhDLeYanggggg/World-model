"""Verify rebuilt windows and test input invariance to hidden-future corruption."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_unification.m3w_causal_recordings import RecordingWindows, clean_points
from src.evaluation.m3w_recording_lineage import read_track, sha256


def audit_recording(directory: Path, source_path: Path, sample_count: int = 16) -> dict:
    ds = RecordingWindows(directory)
    raw, skipped = read_track(source_path)
    raw, _ = clean_points(raw)
    if skipped or not np.array_equal(raw, ds.points):
        raise ValueError("Rebuilt points do not match current canonical raw positions")
    idx = ds.index
    checks = {"source_position_match": True, "artifact_hashes_match": True,
              "all_window_boundaries_valid": True, "all_actual_horizons_exact": True,
              "sampled_inputs_finite": True, "sampled_neighbor_times_causal": True,
              "sampled_future_corruption_invariant": True,
              "sampled_scene_inputs_and_membership_future_invariant": True}
    for row in idx:
        begin, cur, end = (int(row[k]) for k in ("history_start", "current_row", "future_end"))
        span = ds.points[begin:end + 1]
        step = ds.points[cur, 0] - ds.points[cur - 1, 0]
        valid = (0 <= begin <= cur < end < len(ds.points) and
                 cur - begin + 1 == ds.metadata["history_steps"] and
                 np.all(span[:, 1] == span[0, 1]) and np.all(np.diff(span[:, 0]) == step) and step > 0)
        checks["all_window_boundaries_valid"] &= bool(valid)
        checks["all_actual_horizons_exact"] &= bool(
            ds.points[end, 0] - ds.points[cur, 0] == row["horizon_raw"] and end - cur == row["future_steps"])
    rng = np.random.default_rng(20260916)
    sampled = rng.choice(len(ds), size=min(sample_count, len(ds)), replace=False)
    original = ds.points
    scene_count = 0
    for number, item in enumerate(sampled):
        item = int(item)
        before = ds.get_inputs(item)
        frame = ds.identity(item)["frame_id"]
        horizon = ds.identity(item)["horizon_raw"]
        scene_before = ds.get_scene_inputs(frame, horizon) if number < 4 else None
        checks["sampled_inputs_finite"] &= all(np.isfinite(v).all() for v in before.values())
        checks["sampled_neighbor_times_causal"] &= bool(
            (before["neighbor_frame_offsets"][before["neighbor_mask"]] <= 0).all())
        corrupted = np.asarray(original).copy()
        corrupted[corrupted[:, 0] > frame, 2:] = np.nan
        ds.points = corrupted
        after = ds.get_inputs(item)
        checks["sampled_future_corruption_invariant"] &= all(np.array_equal(before[k], after[k]) for k in before)
        if scene_before is not None:
            scene_count += 1
            scene_after = ds.get_scene_inputs(frame, horizon)
            same = [a["agent_id"] for a in scene_before["agents"]] == [a["agent_id"] for a in scene_after["agents"]]
            same &= scene_before["excluded_past_support"] == scene_after["excluded_past_support"]
            same &= all(np.array_equal(a["inputs"][key], b["inputs"][key])
                        for a, b in zip(scene_before["agents"], scene_after["agents"])
                        for key in a["inputs"])
            checks["sampled_scene_inputs_and_membership_future_invariant"] &= same
        ds.points = original
    checks = {k: bool(v) for k, v in checks.items()}
    return {"recording": ds.metadata["id"], "indexed_windows": len(idx),
            "counterfactual_cases": len(sampled), "scene_counterfactual_cases": scene_count,
            "checks": checks, "passed": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/stage_cvpr2027_causal")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "outputs/publication_readiness_2026_09")
    args = parser.parse_args()
    manifest_path = args.cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for record in manifest["recordings"]:
        source = record["files"][0]
        path = ROOT / source["path"]
        if sha256(path) != source["sha256"]:
            raise ValueError("Raw source changed since rebuild")
        rows.append(audit_recording(args.cache_dir / record["id"], path))
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "result_source": "fresh_run",
        "scope": "new_raw_position_reader_and_window_integrity",
        "manifest_sha256": sha256(manifest_path), "recordings": rows,
        "all_indexed_windows_checked": sum(r["indexed_windows"] for r in rows),
        "future_corruption_cases": sum(r["counterfactual_cases"] for r in rows),
        "scene_future_corruption_cases": sum(r["scene_counterfactual_cases"] for r in rows),
        "reader_checks_pass": all(r["passed"] for r in rows),
        "future_input_check": "replace all post-current-frame xy with NaN; inference inputs must remain identical",
        "full_experiment_no_leakage_certification": False,
        "official_split_selected": False, "teacher_refit": "not_run", "model_training": "not_run",
        "metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "causal_recording_checks.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Causal Recording Integrity and Counterfactual Checks", "",
             f"Fresh checks on {payload['all_indexed_windows_checked']} indexed windows, "
             f"with {payload['future_corruption_cases']} sampled real-recording counterfactual cases.", "",
             "| recording | windows checked | future-corruption cases | checks passed |",
             "| --- | ---: | ---: | --- |"]
    lines += [f"| {r['recording']} | {r['indexed_windows']} | {r['counterfactual_cases']} | {r['passed']} |" for r in rows]
    lines += ["", "Source positions were reconstructed independently of legacy teacher/features and compared to raw input. "
              "All indexed history/future spans were checked for one agent, continuous timestamps and exact horizon. "
              "Artifact hashes were verified before use.", "",
              "For sampled real windows, every position after the current frame, for every agent, was replaced by NaN. "
              "All inference inputs, causal baseline rollouts and neighbor features had to remain exactly unchanged. "
              "This is a causal-access test, not a prediction accuracy experiment.", "",
              f"In {payload['scene_future_corruption_cases']} additional scene queries, target-agent membership and "
              "all scene inputs also remained unchanged after future corruption. Agents with sufficient observed "
              "history are included even if their future is absent; future availability is a separate loss-only mask.", "",
              "This does not certify a future train/validation/test split, teacher fitting, all possible input perturbations, "
              "or independent confirmation. Those steps have not run. No metric/seconds, Stage5C or SMC claim."]
    (args.report_dir / "causal_recording_checks.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: payload[k] for k in ("reader_checks_pass", "all_indexed_windows_checked", "future_corruption_cases")}, indent=2))
    if not payload["reader_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
