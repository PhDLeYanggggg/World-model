"""Audit recording identity and inherited teacher exposure, without training."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Mapping

import numpy as np


SPLITS = ("train", "val", "test")
CACHE_DIR = Path("data/stage43_full_waypoint_supervision_cache")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_keys(array: np.ndarray, decimals: int = 5) -> np.ndarray:
    """Sorted unique numeric identities; no translation/scale matching is claimed."""
    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Expected a two-dimensional numeric row matrix")
    values = values[np.isfinite(values).all(axis=1)]
    rounded = np.round(values, decimals=decimals)
    rounded[rounded == 0] = 0.0
    dtype = np.dtype([(f"f{i}", "<f8") for i in range(values.shape[1])])
    return np.unique(np.ascontiguousarray(rounded, dtype="<f8").view(dtype).ravel())


def read_track(path: Path) -> tuple[np.ndarray, int]:
    # Match the local OpenTraj loaders: obsmat x/y are columns 2/4.
    columns = (0, 1, 2, 4) if path.name == "obsmat.txt" else (0, 1, 2, 3)
    rows = []
    skipped = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            tokens = line.replace(",", " ").split()
            if not tokens or tokens[0].startswith("#"):
                continue
            try:
                row = [float(tokens[i]) for i in columns]
            except (ValueError, IndexError):
                skipped += 1
                continue
            if not np.isfinite(row).all():
                skipped += 1
                continue
            rows.append(row)
    return np.asarray(rows, dtype=np.float64).reshape(-1, 4), skipped


def _counts(values: np.ndarray) -> dict[str, int]:
    return {str(k): int(v) for k, v in zip(*np.unique(values, return_counts=True))}


def audit_caches(cache_paths: Mapping[str, Path], root: Path) -> dict:
    """Hash current inputs and compare contents, not just disjoint path strings."""
    root = root.resolve()
    by_source: dict[str, dict] = {}
    cached_rows: dict[str, list[np.ndarray]] = defaultdict(list)
    split_summary = {}
    old_to_new = {}
    horizon_alignment = {}
    inherited_split_metadata = {}
    input_files = []
    missing = []
    metadata_errors = []
    for split, path in cache_paths.items():
        if not path.exists():
            missing.append(str(path))
            continue
        input_files.append({"path": str(path), "sha256": sha256(path)})
        with np.load(path, allow_pickle=False) as z:
            sources = z["source_file"].astype(str)
            n = len(sources)
            inherited_split_metadata[split] = "old_split" in z
            old = z["old_split"].astype(str) if "old_split" in z else np.full(n, split)
            h = z["horizon"].astype(float)
            dt = z["dt_frame_step"].astype(float)
            if not set(old).issubset(SPLITS):
                metadata_errors.append(f"{split}: unknown old_split")
            if not np.isfinite(dt).all() or not np.isfinite(h).all():
                metadata_errors.append(f"{split}: nonfinite horizon/frame delta")
            split_summary[split] = {
                "rows": n, "sources": len(set(sources)),
                "datasets": _counts(z["dataset"]), "horizons": _counts(z["horizon"]),
            }
            old_to_new[split] = _counts(old)
            horizon_alignment[split] = {
                str(int(v)): {
                    "rows": int((h == v).sum()),
                    "endpoint_frame_delta_equals_requested": int(np.isclose(dt[h == v], v).sum()),
                    "min_actual_frame_delta": float(dt[h == v].min()),
                    "max_actual_frame_delta": float(dt[h == v].max()),
                } for v in np.unique(h)
            }
            current = z["current_xy"] if "current_xy" in z else np.column_stack([z["current_x"], z["current_y"]])
            future = z["future_xy"] if "future_xy" in z else np.column_stack([z["future_endpoint_x"], z["future_endpoint_y"]])
            geometry = np.column_stack([z["agent_id"], z["frame_id"], h, current, future])
            for src in sorted(set(sources)):
                mask = sources == src
                entry = by_source.setdefault(src, {"cache_rows": {}, "datasets": set()})
                entry["cache_rows"][split] = int(mask.sum())
                entry["datasets"].update(z["dataset"][mask].astype(str))
                cached_rows[src].append(geometry[mask])
    raw_hashes = defaultdict(list)
    raw_points = {}
    row_sets = {}
    def display(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(root))
        except ValueError:
            return str(path.resolve())
    source_records = []
    for src, entry in sorted(by_source.items()):
        path = Path(src)
        if not path.is_absolute():
            path = root / path
        row_sets[src] = row_keys(np.concatenate(cached_rows[src]))
        record = {"source": display(path), "cache_rows": entry["cache_rows"],
                  "datasets": sorted(entry["datasets"]), "exists": path.exists()}
        if path.exists():
            digest = sha256(path)
            points, skipped = read_track(path)
            raw_points[src] = row_keys(points)
            raw_hashes[digest].append(src)
            record.update(raw_sha256=digest, parsed_rows=len(points), skipped_lines=skipped,
                          parsed_unique_rows=len(raw_points[src]),
                          canonical_numeric_sha256=hashlib.sha256(raw_points[src].tobytes()).hexdigest())
            if len(points) == 0:
                metadata_errors.append(f"Empty parsed source: {display(path)}")
        else:
            missing.append(display(path))
        source_records.append(record)
    exact_groups = []
    for digest, sources in sorted(raw_hashes.items()):
        if len(sources) < 2:
            continue
        exact_groups.append({
            "sha256": digest,
            "sources": [display(Path(s)) for s in sources],
            "splits": sorted({sp for s in sources for sp in by_source[s]["cache_rows"]}),
            "cache_rows": {display(Path(s)): by_source[s]["cache_rows"] for s in sources},
        })
    content_pairs = []
    for a, b in combinations(sorted(raw_points), 2):
        pa, pb = raw_points[a], raw_points[b]
        common = len(np.intersect1d(pa, pb, assume_unique=True))
        if common == 0:
            continue
        smaller = min(len(pa), len(pb))
        overlap = common / smaller if smaller else 0.0
        cache_common = len(np.intersect1d(row_sets[a], row_sets[b], assume_unique=True))
        content_pairs.append({
            "a": display(Path(a)), "b": display(Path(b)),
            "a_splits": sorted(by_source[a]["cache_rows"]),
            "b_splits": sorted(by_source[b]["cache_rows"]),
            "shared_frame_agent_xy_rows": common,
            "fraction_of_smaller_recording": overlap,
            "same_numeric_recording": common == len(pa) == len(pb),
            "strong_subset_match": common >= 100 and overlap >= 0.9,
            "identical_cached_window_geometries": cache_common,
            "crosses_splits": any(x != y for x in by_source[a]["cache_rows"] for y in by_source[b]["cache_rows"]),
        })
    raw_cross = [g for g in exact_groups if len(g["splits"]) > 1]
    numeric_cross = [p for p in content_pairs if p["crosses_splits"] and
                     (p["same_numeric_recording"] or p["strong_subset_match"])]
    path_cross = [r for r in source_records if len(r["cache_rows"]) > 1]
    teacher_exposure = {sp: old_to_new.get(sp, {}).get("train", 0) for sp in ("val", "test")}
    complete = set(cache_paths) == set(SPLITS) and set(split_summary) == set(SPLITS) and all(
        s["rows"] > 0 for s in split_summary.values())
    recording_boundary_pass = complete and not (raw_cross or numeric_cross or path_cross or missing or metadata_errors)
    # Repartitioning a cache does not refit its inherited supervised components.
    teacher_boundary_pass = (complete and not any(teacher_exposure.values())
                             if all(inherited_split_metadata.values()) else None)
    return {
        "result_source": "fresh_run", "scope": "content_identity_and_inherited_training_exposure",
        "cache_inputs": input_files, "sources": source_records, "split_summary": split_summary,
        "exact_duplicate_groups": exact_groups, "numeric_overlap_pairs": content_pairs,
        "source_path_cross_split": path_cross,
        "old_split_by_new_split": old_to_new,
        "inherited_split_metadata_available": inherited_split_metadata,
        "new_evaluation_rows_from_legacy_teacher_train": teacher_exposure,
        "horizon_alignment": horizon_alignment, "missing_inputs": missing,
        "metadata_errors": metadata_errors,
        "recording_boundary_pass": bool(recording_boundary_pass),
        "legacy_teacher_boundary_pass": teacher_boundary_pass,
        "training_allowed_with_unchanged_legacy_caches": bool(recording_boundary_pass and teacher_boundary_pass),
        "full_no_leakage_certification": False,
        "not_proven": ["transformed_or_reindexed_recording_duplicates", "complete_feature_causality",
                       "independent_confirmation_holdout", "metric_or_seconds_calibration"],
        "old_reports_overwritten": False, "new_model_training": "not_run",
        "stage5c_executed": False, "smc_enabled": False,
    }


def assert_legacy_cache_safe(root: Path = Path(".")) -> dict:
    paths = {s: root / CACHE_DIR / f"stage43_full_waypoint_supervision_{s}.npz" for s in SPLITS}
    audit = audit_caches(paths, root)
    if not audit["training_allowed_with_unchanged_legacy_caches"]:
        raise RuntimeError(
            "M3W lineage preflight failed: recording duplicates and/or inherited teacher "
            "training exposure are unresolved. Run scripts/audit_m3w_recording_lineage.py; "
            "rebuild by recording identity and refit all supervised preprocessing before training."
        )
    return audit


def markdown_report(payload: dict) -> str:
    lines = ["# M3W Recording and Teacher Lineage Audit", "",
             "Fresh content audit. No retraining, no prediction replay, no historical output replacement.", "",
             f"- Recording boundary pass: {payload['recording_boundary_pass']}",
             f"- Inherited teacher boundary pass: {payload['legacy_teacher_boundary_pass']}",
             f"- Training with unchanged caches allowed: {payload['training_allowed_with_unchanged_legacy_caches']}",
             "", "## Byte-Identical Source Files", ""]
    for group in payload["exact_duplicate_groups"]:
        lines += [f"- SHA256 `{group['sha256']}`; splits: {group['splits']}"]
        lines += [f"  - `{s}`: {group['cache_rows'][s]}" for s in group["sources"]]
    lines += ["", "## Numeric Recording Overlap", "",
              "Frame/agent/x/y identities rounded to 5 decimal places. No unit conversion or trajectory realignment.", "",
              "| source pair | shared raw rows | fraction of smaller | identical cached windows | crosses splits |",
              "| --- | ---: | ---: | ---: | --- |"]
    for p in payload["numeric_overlap_pairs"]:
        lines.append(f"| `{p['a']}` / `{p['b']}` | {p['shared_frame_agent_xy_rows']} | "
                     f"{p['fraction_of_smaller_recording']:.4f} | {p['identical_cached_window_geometries']} | {p['crosses_splits']} |")
    lines += ["", "## Teacher Exposure", "",
              f"Old split counts within each new split: `{payload['old_split_by_new_split']}`.", "",
              f"New val/test rows that were legacy teacher-training rows: `{payload['new_evaluation_rows_from_legacy_teacher_train']}`.", "",
              f"Inherited old_split metadata available: `{payload['inherited_split_metadata_available']}`. "
              "For raw Stage35 caches, the supplied split is used; no inherited teacher audit is claimed.", "",
              "Stage35 fits supervised models on its old train split. Reusing those outputs in newly assigned val/test "
              "does not provide a held-out teacher. Some old-train rows also reach new train, so its cached "
              "teacher predictions are not out-of-fold gain/harm supervision.", "",
              "Stage37 also ranks selector variants using test metrics (src/stage37_t50_history.py:t50_selector). "
              "Its reported bootstrap interval does not correct test-based selection bias.", "",
              "## Horizon Alignment", "",
              "| split | requested horizon | rows | exact frame delta | min actual | max actual |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for sp, horizons in payload["horizon_alignment"].items():
        for h, r in horizons.items():
            lines.append(f"| {sp} | {h} | {r['rows']} | {r['endpoint_frame_delta_equals_requested']} | "
                         f"{r['min_actual_frame_delta']} | {r['max_actual_frame_delta']} |")
    lines += ["", "## Consequence and Repair", "",
              "Distinct dataset labels or file paths do not establish independent recordings. "
              "The affected Stage43/44 generalization results are exploratory, not clean cross-dataset confirmation. "
              "This audit does not reproduce or invalidate every SDD experiment.", "",
              "Group aliases by original recording, retain one canonical source, quarantine unresolved aliases, "
              "and declare scene/recording-held-out protocols before training. Rebuild goals, baselines, "
              "normalizers and supervised teachers exclusively within each new training fold. "
              "Do not reuse legacy target-derived labels/features as a supposedly held-out floor.", "",
              "All local historic evaluation sources are already exposed to research. A repaired split is not "
              "an untouched confirmation set; prospectively frozen replication or a new source is still required.", "",
              "Dataset-local/raw-frame only. No true-3D, foundation, metric or seconds claim. "
              "Stage5C and SMC remain disabled."]
    return "\n".join(lines) + "\n"
