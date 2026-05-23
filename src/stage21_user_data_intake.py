from __future__ import annotations

import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md


REPORT_DIR = Path("outputs/reports")
STAGE21_INDEX_DIR = Path("data/stage21_user_data_index")
SDD_DIR = Path("external_data/StanfordDroneDataset")
SDD_ARCHIVE = Path("external_data/archive.zip")
OPENTRAJ_DIR = Path("external_data/OpenTraj")


def _annotation_files(root: Path) -> List[Path]:
    return sorted(root.glob("annotations/*/video*/annotations.txt"))


def _scene_video_from_path(path: Path) -> tuple[str, str]:
    parts = path.parts
    idx = parts.index("annotations")
    return parts[idx + 1], parts[idx + 2]


def _parse_sdd_annotations(root: Path) -> Dict[str, Any]:
    files = _annotation_files(root)
    scene_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    video_rows: List[Dict[str, Any]] = []
    total_rows = 0
    global_track_count = 0
    horizon_counts = {10: 0, 25: 0, 50: 0, 100: 0}
    max_track_length = 0
    longest_tracks: List[Dict[str, Any]] = []
    for path in files:
        scene, video = _scene_video_from_path(path)
        scene_counts[scene] += 1
        tracks: Dict[str, Dict[str, Any]] = {}
        row_count = 0
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 10:
                    continue
                track_id = parts[0]
                try:
                    xmin, ymin, xmax, ymax = map(float, parts[1:5])
                    frame = int(float(parts[5]))
                    lost = int(float(parts[6]))
                    occluded = int(float(parts[7]))
                    generated = int(float(parts[8]))
                except ValueError:
                    continue
                label = " ".join(parts[9:]).strip().strip('"')
                rec = tracks.setdefault(
                    track_id,
                    {
                        "track_id": track_id,
                        "label": label,
                        "first_frame": frame,
                        "last_frame": frame,
                        "rows": 0,
                        "lost_rows": 0,
                        "occluded_rows": 0,
                        "generated_rows": 0,
                        "min_x": math.inf,
                        "min_y": math.inf,
                        "max_x": -math.inf,
                        "max_y": -math.inf,
                    },
                )
                rec["first_frame"] = min(rec["first_frame"], frame)
                rec["last_frame"] = max(rec["last_frame"], frame)
                rec["rows"] += 1
                rec["lost_rows"] += int(lost == 1)
                rec["occluded_rows"] += int(occluded == 1)
                rec["generated_rows"] += int(generated == 1)
                rec["min_x"] = min(rec["min_x"], (xmin + xmax) / 2.0)
                rec["min_y"] = min(rec["min_y"], (ymin + ymax) / 2.0)
                rec["max_x"] = max(rec["max_x"], (xmin + xmax) / 2.0)
                rec["max_y"] = max(rec["max_y"], (ymin + ymax) / 2.0)
                label_counts[label] += 1
                row_count += 1
        local_h = {h: sum(max(0, rec["rows"] - h) for rec in tracks.values()) for h in horizon_counts}
        for h, count in local_h.items():
            horizon_counts[h] += count
        local_max = max((rec["rows"] for rec in tracks.values()), default=0)
        max_track_length = max(max_track_length, local_max)
        global_track_count += len(tracks)
        total_rows += row_count
        longest_tracks.extend(
            {
                "scene": scene,
                "video": video,
                "track_id": rec["track_id"],
                "label": rec["label"],
                "rows": rec["rows"],
                "first_frame": rec["first_frame"],
                "last_frame": rec["last_frame"],
            }
            for rec in tracks.values()
        )
        video_rows.append(
            {
                "scene": scene,
                "video": video,
                "annotation_path": str(path),
                "reference_image_path": str(path.parent / "reference.jpg"),
                "video_path": str(root / "video" / scene / video / "video.mp4"),
                "track_count": len(tracks),
                "row_count": row_count,
                "max_track_length": local_max,
                "samples_t10": local_h[10],
                "samples_t25": local_h[25],
                "samples_t50": local_h[50],
                "samples_t100": local_h[100],
                "coordinate_unit": "pixel",
                "metric_status": "pixel-space; no homography/scale verified",
            }
        )
    longest_tracks = sorted(longest_tracks, key=lambda x: x["rows"], reverse=True)[:25]
    return {
        "dataset_name": "Stanford Drone Dataset via user-provided Kaggle archive",
        "source_path": str(root),
        "archive_path": str(SDD_ARCHIVE),
        "scene_count": len(scene_counts),
        "video_count": len(files),
        "annotation_files": len(files),
        "track_count": global_track_count,
        "annotation_rows": total_rows,
        "agent_type_counts": dict(label_counts),
        "scene_video_counts": dict(scene_counts),
        "max_track_length_raw_frames": max_track_length,
        "samples_t10_raw_frame": horizon_counts[10],
        "samples_t25_raw_frame": horizon_counts[25],
        "samples_t50_raw_frame": horizon_counts[50],
        "samples_t100_raw_frame": horizon_counts[100],
        "actual_verified_t50_raw_frame": horizon_counts[50] > 0,
        "actual_verified_t100_raw_frame": horizon_counts[100] > 0,
        "official_metric_t50": False,
        "official_metric_t100": False,
        "coordinate_unit": "pixel",
        "metric_status": "pixel-space; no homography/scale verified for Kaggle archive",
        "fps_or_dt": "raw frame index present; effective seconds not claimed without video fps audit",
        "video_rows": video_rows,
        "longest_tracks": longest_tracks,
    }


def _inspect_archive(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    return {
        "exists": True,
        "path": str(path),
        "file_count": len(names),
        "annotation_txt": sum(n.startswith("annotations/") and n.endswith("annotations.txt") for n in names),
        "reference_images": sum(n.startswith("annotations/") and n.endswith("reference.jpg") for n in names),
        "videos": sum(n.startswith("video/") and n.endswith((".mp4", ".avi", ".mov")) for n in names),
        "top_level": sorted(set(n.split("/")[0] for n in names if "/" in n)),
    }


def _inspect_opentraj(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    return {
        "exists": True,
        "path": str(path),
        "remote": "https://github.com/amiryanj/OpenTraj.git",
        "toolkit_exists": (path / "opentraj" / "toolkit").exists(),
        "dataset_dirs": sorted(p.name for p in (path / "datasets").iterdir() if p.is_dir()) if (path / "datasets").exists() else [],
        "sdd_annotation_files": len(list((path / "datasets" / "SDD").glob("*/*/annotations.txt"))),
        "eth_ucy_files": len(list((path / "datasets" / "ETH").rglob("*.txt"))) + len(list((path / "datasets" / "UCY").rglob("*.txt"))),
    }


def ingest_user_data() -> Dict[str, Any]:
    ensure_dir(STAGE21_INDEX_DIR)
    archive = _inspect_archive(SDD_ARCHIVE)
    opentraj = _inspect_opentraj(OPENTRAJ_DIR)
    sdd = _parse_sdd_annotations(SDD_DIR) if SDD_DIR.exists() else {"exists": False}
    if sdd.get("annotation_files"):
        write_json(STAGE21_INDEX_DIR / "sdd_kaggle_user_archive_index.json", sdd)
    result = {
        "stage": "stage21_user_provided_data_intake",
        "opentraj": opentraj,
        "sdd_archive": archive,
        "sdd_ingestion": {
            k: v
            for k, v in sdd.items()
            if k not in {"video_rows", "longest_tracks", "agent_type_counts"}
        },
        "license_audit": {
            "opentraj": "GitHub toolkit/repo; repository license must be respected; underlying datasets keep their own terms.",
            "kaggle_sdd": "Kaggle mirror page reports CC0, but official Stanford SDD source/licensing is non-commercial/custom. Treat as user-provided local research data and do not redistribute raw files.",
            "legal_status": "accepted_by_user_implied_for_local archive; still not committed to GitHub; official license conflict recorded.",
        },
        "no_leakage": {
            "candidate_goals_built": False,
            "test_endpoints_used": False,
            "future_endpoint_as_input": False,
            "central_velocity_official": False,
            "status": "raw intake only; full Stage21/22 episode split audit still required",
        },
        "stage5c_ready": False,
        "smc_ready": False,
    }
    write_json(REPORT_DIR / "stage21_user_provided_data_intake.json", result)
    write_md(
        REPORT_DIR / "stage21_user_provided_data_intake.md",
        [
            "# Stage 21 User-Provided Data Intake",
            "",
            "- No model training was run.",
            "- Latent generative Stage 5C remains blocked.",
            "- SMC remains blocked.",
            "- Raw external data stays under `external_data/` and is ignored by git.",
            "",
            "## OpenTraj",
            "",
            f"- exists: `{opentraj.get('exists')}`",
            f"- toolkit exists: `{opentraj.get('toolkit_exists')}`",
            f"- dataset dirs: `{len(opentraj.get('dataset_dirs', []))}`",
            f"- SDD annotation files in OpenTraj tree: `{opentraj.get('sdd_annotation_files')}`",
            f"- ETH/UCY txt files in OpenTraj tree: `{opentraj.get('eth_ucy_files')}`",
            "",
            "## Kaggle SDD Archive",
            "",
            f"- archive exists: `{archive.get('exists')}`",
            f"- archive files: `{archive.get('file_count')}`",
            f"- annotation files: `{archive.get('annotation_txt')}`",
            f"- reference images: `{archive.get('reference_images')}`",
            f"- videos: `{archive.get('videos')}`",
            "",
            "## Parsed SDD Summary",
            "",
            f"- scenes: `{sdd.get('scene_count', 0)}`",
            f"- videos: `{sdd.get('video_count', 0)}`",
            f"- tracks: `{sdd.get('track_count', 0)}`",
            f"- annotation rows: `{sdd.get('annotation_rows', 0)}`",
            f"- max track length raw frames: `{sdd.get('max_track_length_raw_frames', 0)}`",
            f"- raw-frame t+50 samples: `{sdd.get('samples_t50_raw_frame', 0)}`",
            f"- raw-frame t+100 samples: `{sdd.get('samples_t100_raw_frame', 0)}`",
            "- coordinate status: `pixel-space; no homography/scale verified`",
            "- effective seconds: not claimed until fps/video audit.",
            "",
            "## License / Access",
            "",
            "- Kaggle mirror reports CC0, but official Stanford SDD source/licensing is non-commercial/custom. The stricter/original-source status is recorded.",
            "- Raw data must not be committed or redistributed.",
            "",
            "## Next Step",
            "",
            "1. Build Stage21 full SDD world-state rows from annotations.",
            "2. Create train/val/test scene split and train-only candidate goal dictionaries.",
            "3. Run full horizon/no-leakage audit before any deterministic retraining.",
        ],
    )
    return result


def main() -> None:
    ingest_user_data()

