"""Audit PersonPath22 training support and public camera-source lineage."""

import argparse
import ast
from collections import Counter, defaultdict
from email.parser import BytesHeaderParser
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_unification.m3w_public_zip_range import RangeArchive

BASE = "https://tracking-dataset-eccv-2022.s3.us-east-2.amazonaws.com/dataset"
PUBLISHER = ("https://raw.githubusercontent.com/amazon-science/tracking-dataset/"
             "2f9c757f847e172ce375372913bdbb3a47074e4f/download.py")
ANNOTATIONS = ("annotation/anno_amodal.zip", 11246947, '"241a58af33ced51af25683072566926b-2"')
VIDEO = ("raw_data/videos.zip", 12244881784, '"637e9940f6407010a068fd95949a7d4a-1460"')
SPLITS = ("annotation/splits.json", 6888, '"11288ac51a82cc0b869c97001acc1326"')
FE_ARCHIVES = (
    ("https://cos.twcc.ai/fishlentrafficdataset/Fisheye8K_all_including_train%26test_update_2024Jan.zip",
     13362946197, '"0b794863fa3708a69fe363b066760ff8-1275"'),
    ("https://cos.twcc.ai/fishlentrafficdataset/Fisheye8K.zip", 13362641763,
     '"7e64a221bb4586d27c2467d561e82975-1593"'),
)
OUT = ROOT / "outputs/publication_readiness_2026_09/personpath22_source_intake_v1"
LOCAL = ROOT / "external_data/PersonPath22_source_intake"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_atomic(path, body):
    pending = path.with_suffix(".pending")
    pending.write_bytes(body)
    pending.replace(path)


def transfer(url, size, interval=None):
    args = ["curl", "--fail", "--silent", "--show-error", "--include", "--suppress-connect-headers",
            "--max-time", "40", "--max-filesize", str(size), "--header", "Accept-Encoding: identity",
            "--proto", "=https"]
    if interval is not None:
        args += ["--range", f"{interval[0]}-{interval[1]}"]
    raw = subprocess.check_output([*args, url])
    head, sep, body = raw.partition(b"\r\n\r\n")
    status, _, headers = head.partition(b"\r\n")
    status_code = b"206" if interval is not None else b"200"
    h = BytesHeaderParser().parsebytes(headers)
    if (not sep or status.split()[1:2] != [status_code] or len(body) > size
            or h.get("Content-Encoding", "identity") != "identity"):
        raise ValueError("Publisher transfer differs from the requested representation")
    return body, h


class PinnedArchive(RangeArchive):
    def __init__(self, spec, byte_limit):
        url, size, self.etag = spec
        super().__init__(url, size, byte_limit=byte_limit)

    def _fetch(self, first, last):
        n = last - first + 1
        if not 0 <= first <= last < self.length or self.total + n > self.byte_limit:
            raise ValueError("Range or network byte budget exceeded")
        body, h = transfer(self.url, n, (first, last))
        if (len(body) != n or h.get("ETag") != self.etag
                or h.get("Content-Range") != f"bytes {first}-{last}/{self.length}"):
            raise ValueError("Archive snapshot or requested range changed")
        self.total += n
        self.version = self.etag
        return body


def inspect_fe(spec):
    with PinnedArchive(spec, 16 << 20) as stream, ZipFile(stream) as archive:
        entries = [PurePosixPath(x.filename) for x in archive.infolist() if not x.is_dir()]
        kinds = Counter(x.suffix.lower() for x in entries)
        mot = [str(x) for x in entries if x.name.lower() in {"gt.txt", "seqinfo.ini", "tracks.csv"}]
        return dict(url=stream.url, etag=stream.etag, archive_bytes=stream.length,
                    fetched_bytes=stream.total, suffix_counts=dict(sorted(kinds.items())),
                    mot_index_members=mot, coco_tables=sum(kinds[k] for k in (".json",)))


def source_groups(script):
    """Recover documented source grouping without executing the publisher script."""
    assigns = {node.targets[0].id: node.value for node in ast.parse(script).body
               if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    raw = assigns["external_video_id_mapping"]
    sources = {ast.literal_eval(k): ast.literal_eval(v) for k, v in zip(raw.keys, raw.values, strict=True)
               if ast.literal_eval(k) != "Virat"}
    sources["Virat"] = {k: pair[0] for k, pair in ast.literal_eval(assigns["virat_id_to_uid_and_urlid"]).items()}
    groups = {}
    for source, table in sources.items():
        for original, uid in table.items():
            if uid in groups:
                raise ValueError("Ambiguous publisher UID mapping")
            original_id, camera = original, None
            if source == "Meva":
                original_id = original.rsplit("_", 1)[0]
                camera = original_id.rsplit(".", 1)[-1]
            elif source == "PathTrack":
                original_id = original.rsplit("_", 2)[0]
            elif source == "Virat":
                parts = original.split("_")
                camera = parts[2]
            groups[uid] = dict(source=source, original=original_id, camera=camera)
    return groups


def split_overlaps(groups, split):
    result = {}
    for field in ("original", "camera"):
        seen = defaultdict(lambda: defaultdict(list))
        for role, ids in split.items():
            for uid in ids:
                if uid in groups and groups[uid][field] is not None:
                    info = groups[uid]
                    seen[(info["source"], info[field])][role].append(uid)
        result[field] = [{"source": source, "key": group, **roles}
                         for (source, group), roles in sorted(seen.items()) if len(roles) > 1]
    return result


def grid_support(data, video):
    m = data["metadata"]
    if m["data_path"] != video:
        raise ValueError("Reference UID mismatch")
    fps, length = float(m["fps"]), int(m["number_of_frames"])
    if not 5 <= fps <= 120 or length < 1:
        raise ValueError("Invalid publisher video clock")
    seen, ticks_by_id, frame_by_tick = set(), defaultdict(set), {}
    off_grid = clock_mismatch = conflicting_ticks = 0
    max_error_ms = 0.
    for row in data["entities"]:
        idx, key = row["blob"]["frame_idx"], row["id"]
        if type(idx) is not int or not 0 <= idx < length or (idx, key) in seen:
            raise ValueError("Frame out of range or repeated identity/frame")
        seen.add((idx, key))
        time_ms = 1000 * idx / fps
        clock_mismatch += abs(row["time"] - time_ms) > 1.05
        tick = round(time_ms / 200)
        error = abs(time_ms - tick * 200)
        max_error_ms = max(max_error_ms, error)
        if error > 500 / fps + 1:
            off_grid += 1
            continue
        if frame_by_tick.setdefault(tick, idx) != idx:
            conflicting_ticks += 1
        ticks_by_id[key].add(tick)
    last_tick = round((length - 1) * 5 / fps)
    starts = range(0, last_tick - 37, 40)
    counts = dict(camera_grid_queries=len(starts), reference_past=0, reference_complete=0,
                  reference_pair_queries=0, complete_pair_queries=0, full_reference_roster_queries=0)
    if off_grid or clock_mismatch or conflicting_ticks:
        return dict(fps=fps, frames=length, rows=len(seen), clock_ok=False,
                    max_keyframe_rounding_ms=max_error_ms, clock_mismatch_rows=clock_mismatch,
                    off_grid_rows=off_grid, conflicting_ticks_rows=conflicting_ticks,
                    counts=None)
    for t in starts:
        past, future = set(range(t, t + 16, 2)), set(range(t + 16, t + 40, 2))
        observed = [ticks for ticks in ticks_by_id.values() if past <= ticks]
        n = sum(future <= ticks for ticks in observed)
        counts["reference_past"] += len(observed)
        counts["reference_complete"] += n
        counts["reference_pair_queries"] += len(observed) >= 2
        counts["complete_pair_queries"] += n >= 2
        counts["full_reference_roster_queries"] += n >= 2 and n == len(observed)
    return dict(fps=fps, frames=length, rows=len(seen), clock_ok=True,
                max_keyframe_rounding_ms=max_error_ms, clock_mismatch_rows=0, off_grid_rows=0,
                conflicting_ticks_rows=0, counts=counts)


def read_small(spec):
    member, size, etag = spec
    body, h = transfer(f"{BASE}/{member}", size)
    if len(body) != size or h.get("ETag") != etag:
        raise ValueError("Small publisher object changed")
    return body


def report(include_fe):
    split_raw = read_small(SPLITS)
    split = json.loads(split_raw)
    if set(split) != {"train", "test"} or any(len(x) != len(set(x)) for x in split.values()) or (
            set(split["train"]) & set(split["test"])):
        raise ValueError("Publisher split has conflicting UIDs")
    source, _ = transfer(PUBLISHER, 80 << 10)
    groups = source_groups(source.decode("utf-8"))
    with PinnedArchive((f"{BASE}/{VIDEO[0]}", *VIDEO[1:]), 4 << 20) as stream, ZipFile(stream) as videos:
        embedded = {i.filename: i.file_size for i in videos.infolist()
                    if re.fullmatch(r"uid_vid_\d{5}\.mp4", i.filename)}
        video_bytes = stream.total
    archive_body = read_small(ANNOTATIONS)
    with ZipFile(io.BytesIO(archive_body)) as ann:
        index = json.loads(ann.read("anno_amodal_2022.json"))["samples"]
        if set(index) != set(split["train"]) | set(split["test"]) or any(
                set(v) != {"metadata", "sample_file"} or v["sample_file"] is not True for v in index.values()):
            raise ValueError("Expected a metadata-only aggregate and external per-clip records")
        diagnostics = {}
        for video in sorted(split["train"]):
            data = json.loads(ann.read(f"anno_amodal_2022/{video}.json"))
            if data["metadata"] != index[video]["metadata"]:
                raise ValueError("Clip metadata differs from the aggregate")
            diagnostics[video] = dict(**grid_support(data, video),
                                      in_video_zip=video in embedded, documented_source=groups.get(video))
    aggregate = Counter()
    for row in diagnostics.values():
        if row["counts"] is not None:
            aggregate.update(row["counts"])
    return dict(
        evaluation_status="source_support_only", test_annotation_members_read=0, official_roles_reassigned=False,
        split_sha256=sha(split_raw), annotation_archive_sha256=sha(archive_body),
        download_script_sha256=sha(source), public_camera_archive_etag=VIDEO[2],
        public_camera_zip_catalog_bytes=video_bytes, public_video_members=len(embedded),
        role_counts={k: len(v) for k, v in split.items()},
        raw_video_available_in_zip={k: sum(x in embedded for x in v) for k, v in split.items()},
        verified_documented_source_overlap=split_overlaps(groups, split),
        train_clip_clock_ok=sum(r["clock_ok"] for r in diagnostics.values()),
        train_clock_compatible_counts=dict(aggregate), training_support=diagnostics,
        fisheye_older_archive_catalogs=[inspect_fe(x) for x in FE_ARCHIVES] if include_fe else [],
        no_forecast_or_risk_result=True, independent_confirmation=False,
        code_sha256=sha(Path(__file__).read_bytes()),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--include-fisheye-catalogs", action="store_true")
    args = p.parse_args()
    result = report(args.include_fisheye_catalogs)
    OUT.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    dest = OUT / "analysis.json"
    if dest.exists() and dest.read_bytes() != body:
        raise ValueError("Previous source audit differs")
    write_atomic(dest, body)
    print(json.dumps({k: v for k, v in result.items() if k != "training_support"}, indent=2))


if __name__ == "__main__":
    main()
