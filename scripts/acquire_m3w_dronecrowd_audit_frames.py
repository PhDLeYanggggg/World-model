"""Acquire three frozen audit frames per clip, not whole archives or model rows."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import zlib

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_m3w_dronecrowd_metadata import plain_path
from src.evaluation.m3w_dronecrowd_image_intake import (
    IMAGE_ARCHIVES, RangeArchive, image_identity, require_image_permission,
)
from src.evaluation.m3w_dronecrowd_intake import sequence_ids, sha


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "external_data/DroneCrowd_images")
    args = parser.parse_args()
    output = plain_path(args.output)
    if not output.is_relative_to(ROOT / "external_data"):
        raise SystemExit("Official images must stay in ignored external_data")
    if subprocess.run(["git", "check-ignore", "--quiet", str(output / "sentinel.jpg")], cwd=ROOT).returncode:
        raise SystemExit("Raw destination must be excluded from Git")
    permission_path = ROOT / "outputs/publication_readiness_2026_09/delegated_research_authorization_20260923.json"
    require_image_permission(json.loads(permission_path.read_text()))
    expected = set(sequence_ids((ROOT / f"external_data/DroneCrowd_release_metadata/{args.split}list.txt").read_bytes()))
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{args.split}_sparse_manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    start = time.monotonic()
    manifest = {"source": "official_DroneCrowd_public_Google_Drive", "release_split": args.split,
                "file_id": IMAGE_ARCHIVES[args.split], "authorization_sha256": sha(permission_path.read_bytes()),
                "selected_frames_one_based": [1, 150, 300], "selection_uses_forecast_errors": False,
                "scientific_roles_assigned": False, "full_archive_sha256": None,
                "status": "running", "pid": __import__("os").getpid(), "frames": []}
    with RangeArchive(IMAGE_ARCHIVES[args.split]) as remote:
        with zipfile.ZipFile(remote) as archive:
            central = [{"name": i.filename, "crc32": i.CRC, "size": i.file_size,
                        "compressed": i.compress_size, "offset": i.header_offset}
                       for i in archive.infolist()]
            central_hash = sha(json.dumps(central, sort_keys=True).encode())
            if old and old.get("central_directory_sha256") != central_hash:
                raise ValueError("Source directory differs from previous acquisition")
            manifest.update(archive_bytes=remote.size, member_count=len(central),
                            central_directory_sha256=central_hash)
            images = {}
            for member in archive.infolist():
                if member.filename.lower().endswith(".jpg"):
                    identity = image_identity(member.filename)
                    if identity in images:
                        raise ValueError("Duplicate image identity in archive")
                    if (member.flag_bits & 1 or member.file_size > 16 * 1024**2
                            or member.compress_size > 16 * 1024**2):
                        raise ValueError("Encrypted/oversized image")
                    images[identity] = member
            if set(images) != {(scene, frame) for scene in expected for frame in range(1,301)}:
                raise ValueError("Image inventory does not match official annotation lists")
            manifest["total_jpeg_members"] = len(images)
            write_json(manifest_path, manifest)
            old_frames = {(x["scene_id"], x["frame_one_based"]): x for x in (old or {}).get("frames", [])}
            for scene in sorted(expected):
                for frame in manifest["selected_frames_one_based"]:
                    info = images[scene, frame]
                    path = output / f"img{int(scene):03d}{frame:03d}.jpg"
                    previous = old_frames.get((scene, frame))
                    if path.exists():
                        payload = path.read_bytes()
                        if (len(payload) != info.file_size or zlib.crc32(payload) != info.CRC
                                or (previous and sha(payload) != previous["sha256"])):
                            raise ValueError("Modified local image; refusing overwrite")
                        source = "cached_verified"
                    else:
                        payload = archive.read(info)
                        with Image.open(io.BytesIO(payload)) as picture:
                            picture.verify()
                        with path.open("xb") as stream:
                            stream.write(payload)
                        source = "fresh_run"
                    with Image.open(io.BytesIO(payload)) as picture:
                        if picture.size != (1920, 1080) or picture.format != "JPEG":
                            raise ValueError("Unexpected image dimensions or format")
                    manifest["frames"].append({"scene_id": scene, "frame_one_based": frame,
                                              "annotation_frame_zero_based": frame-1,
                                              "member": info.filename, "crc32": info.CRC,
                                              "bytes": len(payload), "sha256": sha(payload),
                                              "local_name": path.name, "result_source": source})
                    manifest.update(downloaded_range_bytes=remote.transferred_bytes,
                                    range_requests=remote.range_requests,
                                    elapsed_seconds=round(time.monotonic()-start, 3),
                                    heartbeat_utc=datetime.now(timezone.utc).isoformat())
                    write_json(manifest_path, manifest)
                print(json.dumps({"split": args.split, "last_scene": scene,
                                  "images_verified": len(manifest["frames"]),
                                  "MB_transferred": round(remote.transferred_bytes/1e6, 2),
                                  "seconds": round(time.monotonic()-start, 1)}), flush=True)
        # Re-read directory through a new connection to catch source replacement.
        with RangeArchive(IMAGE_ARCHIVES[args.split]) as check:
            with zipfile.ZipFile(check) as archive:
                after = [{"name": i.filename, "crc32": i.CRC, "size": i.file_size,
                          "compressed": i.compress_size, "offset": i.header_offset}
                         for i in archive.infolist()]
            if check.size != remote.size or sha(json.dumps(after, sort_keys=True).encode()) != central_hash:
                raise ValueError("Archive changed during sparse acquisition")
            manifest["downloaded_range_bytes"] += check.transferred_bytes
            manifest["range_requests"] += check.range_requests
        manifest.update(status="complete_sparse_acquisition_crc_and_dimensions_verified",
                        completed_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=round(time.monotonic()-start, 3),
                        sparse_image_bytes=sum(x["bytes"] for x in manifest["frames"]))
        write_json(manifest_path, manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k != "frames"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
