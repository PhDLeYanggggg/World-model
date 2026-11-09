"""Separate reference-box and identity losses on the frozen PersonPath22 pilot."""

from collections import defaultdict
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_personpath22_camera_probe as probe
from src.world_model.m3w_streaming_boxes import overlap

VISIBLE = ("annotation/anno_visible.zip", 9809815, '"5825f42f171f471dcd6d4512aa1f19df-2"')


def references(blob, uid):
    with ZipFile(io.BytesIO(blob)) as z:
        members = [f for f in z.namelist() if f.endswith("/" + uid + ".json")]
        if len(members) != 1:
            raise ValueError("Expected one published training reference member")
        data = json.loads(z.read(members[0]))
    if data["metadata"]["data_path"] != uid:
        raise ValueError("Reference UID changed")
    by_frame, id_ticks = defaultdict(list), defaultdict(set)
    for row in data["entities"]:
        f, ident = row["blob"]["frame_idx"], row["id"]
        if any(row["labels"].get(x, 0) for x in ("reflection", "person in vehicle", "person on open vehicle")):
            continue
        x, y, w, h = row["bb"]
        if w > 0 and h > 0:
            by_frame[f].append((ident, (x, y, x + w, y + h)))
            id_ticks[ident].add(f)
    return by_frame, id_ticks


def point_support(frames, future, by_frame, id_ticks):
    matched, references_at_time = [], []
    boxes = references_count = pairs = 0
    for f in frames:
        online = np.asarray(f["rows"]).reshape(-1, 7)
        refs = by_frame.get(f["camera_frame"], [])
        matrix = overlap(online[:, 1:5], np.asarray([b for _, b in refs]).reshape(-1, 4))
        matrix[matrix < .5] = 0
        i, j = linear_sum_assignment(-matrix)
        kept = {refs[b][0] for a, b in zip(i, j, strict=True) if matrix[a, b] > 0}
        matched.append(kept)
        references_at_time.append({i for i, _ in refs})
        boxes += len(online)
        references_count += len(refs)
        pairs += len(kept)
    all_past = set.intersection(*references_at_time)
    all_hits = set.intersection(*matched)
    full = {i for i in all_past if set(future) <= id_ticks[i]}
    return dict(framewise_online_boxes=boxes, framewise_reference_boxes=references_count,
                framewise_pairs=pairs, released_complete=len(full),
                complete_without_online_identity=len(full & all_hits))


def main():
    from scripts import run_m3w_acme_hybrid_detector as h

    reg = probe.intake.OUT / "camera_registration_v2.json"
    parent = probe.intake.OUT / "camera_probe.json"
    r, previous = [json.loads(p.read_text()) for p in (reg, parent)]
    key = h.sha(reg)
    if (previous["registration_sha256"] != key or r["runner_sha256"] != h.sha(Path(probe.__file__))
            or previous["official_test_read"]):
        raise ValueError("Missing fixed training-camera evidence")
    schedule = r["native_frame_schedule"]
    frames = [h.old.reassociate(probe.load_window(i, q, key)["frames"])
              for i, q in enumerate(schedule, 1)]
    outputs = {}
    for mode, source in (("amodal", probe.intake.ANNOTATIONS), ("visible", VISIBLE)):
        blob = probe.intake.read_small(source)
        digest = probe.intake.sha(blob)
        if mode == "amodal" and digest != r["annotation_archive_sha256"]:
            raise ValueError("Amodal reference snapshot changed")
        by_frame, ticks = references(blob, r["uid"])
        rows = []
        for k, (obs, q) in enumerate(zip(frames, schedule, strict=True)):
            counts = probe.coverage(obs, q[8:], by_frame, ticks)
            if mode == "amodal" and counts != previous["windows"][k]:
                raise ValueError("Prior coverage calculation did not replay")
            points = point_support(obs, q[8:], by_frame, ticks)
            rows.append({**counts, **points})
        outputs[mode] = dict(source_sha256=digest, windows=rows,
                             totals={k: sum(row[k] for row in rows) for k in rows[0]})
    result = dict(scope="reference geometry diagnostic on one exposed training clip",
                  camera_registration_sha256=key, parent_readout_sha256=h.sha(parent),
                  code_sha256=h.sha(Path(__file__)), independent_confirmation=False,
                  official_test_read=False, comparison=outputs)
    h.immutable(probe.intake.OUT / "camera_box_reference_comparison.json", result)
    print(json.dumps({mode: arm["totals"] for mode, arm in outputs.items()}))


if __name__ == "__main__":
    main()
