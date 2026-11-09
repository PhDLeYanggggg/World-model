"""Score pinned camera support with the publisher's one-based raster convention."""

import fcntl
import json
from pathlib import Path
import sys
from zipfile import ZipFile
from zlib import crc32

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_mot20_camera_support as pilot


def zero_based_references(body):
    gt = pilot.parse_gt(body)
    for boxes in gt.values():
        boxes[:, 2:6] -= 1
    return gt


def reference_bytes():
    label = pilot.LOCAL / "gt.txt"
    with pilot.archive() as stream, ZipFile(stream) as z:
        if label.is_file():
            body = label.read_bytes()
            info = z.getinfo(f"{pilot.PREFIX}/gt/gt.txt")
            if (stream.version != pilot.ARCHIVE_VERSION or len(body) != info.file_size
                    or crc32(body) != info.CRC):
                raise ValueError("Reference snapshot changed")
        else:
            body = pilot.member(z, stream, "gt/gt.txt")
            pilot.write_bytes(label, body)
    return label, body


def evaluate(hybrid):
    reg = pilot.PUBLIC / "registration.json"
    data = json.loads(reg.read_text())
    if data["bindings"] != pilot.bindings(hybrid) or data["starts"] != pilot.starts():
        raise ValueError("Frozen camera inputs changed")
    key = hybrid.sha(reg)
    caches = [pilot.read_window(s, key) for s in pilot.starts()]
    backends = {row["backend"] for row in caches}
    if len(backends) != 1:
        raise ValueError("Mixed detector backends")
    evaluation = pilot.PUBLIC / "evaluation_registration.json"
    hybrid.immutable(evaluation, dict(
        registration_sha256=key, evaluator_sha256=hybrid.sha(Path(__file__)),
        official_format="https://arxiv.org/pdf/2003.09003#page=4",
        geometry="MOT xywh corner from one-based raster -> zero-based xyxy; width and height preserved",
        iou_floor=.5, stable_distractor_exclusion="evaluation-only sensitivity",
        motion_diagnostic_sha256=hybrid.sha(ROOT / "scripts/score_m3w_mot20_motion_diagnostic.py"),
    ))
    label, body = reference_bytes()
    gt = zero_based_references(body)
    ticks = {}
    for t, rows in gt.items():
        for i in rows[(rows[:, 6] != 0) & (rows[:, 7] == 1), 1].astype(int):
            ticks.setdefault(i, set()).add(t)
    result = {"last_box": [], "causal_transport": []}
    for item in caches:
        frames = item["frames"]
        for name, source in (("last_box", frames), ("causal_transport", hybrid.old.reassociate(frames))):
            result[name].append(dict(start=frames[0]["camera_frame"], **pilot.window_support(source, gt, ticks)))
    hybrid.immutable(pilot.PUBLIC / "analysis.json", dict(
        registration_sha256=key, evaluation_registration_sha256=hybrid.sha(evaluation),
        backend=next(iter(backends)), label_sha256=hybrid.sha(label), result_source="fresh_run",
        independent_confirmation=False, policy_evaluation="not_run", reserved_outcomes_opened=False,
        original_2pct=True, windows=result,
    ))
    for name, rows in result.items():
        counts = {mode: {field: sum(row[mode][field] for row in rows) for field in rows[0][mode]}
                  for mode in ("strict", "ignore_only_diagnostic")}
        print(json.dumps(dict(tracker=name, **counts, reference_complete=sum(r["reference_complete"] for r in rows),
                              complete_framewise=sum(r["complete_framewise"] for r in rows),
                              framewise_pair_windows=sum(r["complete_framewise"] >= 2 for r in rows))))


if __name__ == "__main__":
    from scripts import run_m3w_acme_hybrid_detector as hybrid

    with (pilot.PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evaluate(hybrid)
