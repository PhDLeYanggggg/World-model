"""Acquire only the observed images needed by the fixed fitting roster."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

if sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 before importing numerical libraries")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_past_background as frames
from scripts import run_m3w_moving_past_background_own as own
from scripts import run_m3w_conditional_gain as conditional

PUBLIC = frames.PUBLIC.parent / "native_fit_past_pixels_v1"
PRIVATE = frames.PRIVATE.parent / PUBLIC.name
REG = PUBLIC / "registration.json"
FILES = (Path(__file__), PUBLIC / "protocol.md")
np = frames.np


def roster(jobs):
    f = conditional.base.fit_source
    cfg = frames.read(own.CONFIG)
    adm = f.prep.load_admission(ROOT, ROOT / f.read(f.source.CONFIG)["admission"])
    structure = f.read(f.bank.old.HOME / "movingdronecrowd_intake_v1/analysis.json")
    population = f.read(f.PUBLIC / "fit_population.json")
    lookup = {v["clip"]: v for v in structure["recordings"]}
    idx = {}
    for item in frames.read(ROOT / cfg["archive_inventory"]):
        clip = frames.clip_key(item["name"], "frames")
        if clip is not None:
            idx[clip, int(Path(item["name"]).stem)] = item
    prior_caches = {}
    for previous in (own.PARENT_REG, own.REG):
        previous = frames.read(previous)
        prior_caches.update({(r["node"], r["annotation_frame"]): r for r in previous["roster"]["frames"]})
    records, rows, ids = [], [], []
    offset = 0
    identity = jobs[0]["identity"]
    for node, meta in sorted(adm["recordings"].items()):
        c = lookup[meta["clip"]]
        n = c["window_support"]["obs8_pred12_native_stride1"]["history_complete_agent_instances"]
        start, offset = offset, offset + n
        if not n or meta["exclusion_group"] not in identity["normalizers"]:
            continue
        path = ROOT / meta["source_annotation_path"]
        f.prep.require_fit_record(adm, node, "fit", path)
        rec = f.prep.mot_record(path.read_bytes(), meta["clip"], range(1, c["image_frames"] + 1))
        qs = []
        for q in range(7, rec.valid.shape[1] - 12):
            count = int(rec.valid[:, q-7:q+1].all(1).sum())
            if count:
                qs.append([q, count])
        if sum(n for _, n in qs) != n:
            raise ValueError("Fixed fitting population count changed")
        ids.extend(range(start, start+n))
        records.append(dict(node=node, clip=meta["clip"], source=meta["exclusion_group"],
            annotation=frames.artifact(path), annotation_frames=c["image_frames"], queries=qs, rows=n))
        needed = sorted({s for q, _ in qs for s in range(q-7, q+1)})
        for q in needed:
            row = dict(node=node, annotation_frame=q, image_frame=q+1, member=idx[meta["clip"], q+1])
            if (node, q) in prior_caches and prior_caches[node, q] != row:
                raise ValueError("Previously acquired frame member differs")
            rows.append(row)
    if ([r["node"] for r in records] != population["recordings"] or len(ids) != 69746
            or sum(len(r["queries"]) for r in records) != 1176
            or conditional.base.fit_source.source.array_hash(np.asarray(ids)) != identity["train_ids_sha256"]):
        raise ValueError("Require the original full fitting roster")
    return dict(records=records, frames=rows, rows=len(ids), queries=1176, future_labels_accessed=False)


def register():
    _, jobs, _ = conditional.register()
    conditional.prev.receipt(conditional.PUBLIC, "replay")
    cfg = frames.read(own.CONFIG)
    reg = frames.stable(dict(files=[frames.artifact(p) for p in (*FILES, own.CONFIG,
        own.REG, own.PARENT_REG, conditional.REG, Path(frames.__file__), Path(frames.method.__file__))],
        source_config=frames.read(ROOT / cfg["source_config"]),
        archive_url=frames.intake.cached_source()["archive_url"], roster=roster(jobs),
        role="fixed_fitting_only_past_pixels", raw_jpeg_retained=False, original_rule_unchanged=True))
    frames.once(REG, reg)
    return cfg, reg


def verified_receipt(row, roots):
    found = []
    for home, identity in roots:
        p = home / "rasters" / row["node"] / f'{row["annotation_frame"]:06d}.json'
        if p.is_file():
            v = frames.read(p)
            if v["row"] != row or v["registration_sha256"] != frames.sha(identity) or not v["ZIP_CRC_checked"]:
                raise ValueError("Raster cache does not bind to this image")
            frames.checked(v["cache"])
            found.append((p, v))
    if not found:
        return None
    if len({v["JPEG_sha256"] for _, v in found}) != 1:
        raise ValueError("Cached variants decode different publisher bytes")
    return found[0]


def prepare(cfg, reg, resume, beat):
    old = ((own.PRIVATE, own.REG), (own.PARENT_PRIVATE, own.PARENT_REG))
    local = ((PRIVATE, REG),)
    refs, pending = [], []
    for r in reg["roster"]["frames"]:
        done = verified_receipt(r, old + local)
        if done is None:
            pending.append(r)
        else:
            refs.append(done[0])
    beat(state="cache_audit", cached=len(refs), pending=len(pending))
    if pending:
        frames.PUBLIC, frames.PRIVATE, frames.REG = PUBLIC, PRIVATE, REG
        acquisition = frames.acquire(cfg, reg, pending, beat, resume)
        for r in pending:
            refs.append(verified_receipt(r, local)[0])
    else:
        acquisition = dict(fresh_images=0, range_bytes=0, range_requests=0)
    index = {}
    for p in refs:
        v = frames.read(p)
        row = v["row"]
        key = f'{row["node"]}:{row["annotation_frame"]}'
        if key in index:
            raise ValueError("Duplicate past image")
        index[key] = frames.artifact(p)
    out = PRIVATE / "past_frame_index.json"
    frames.once(out, frames.stable(dict(registration=frames.sha(REG), images=index)))
    return out, {k: acquisition[k] for k in ("fresh_images", "range_bytes", "range_requests")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("register", "prepare", "verify"))
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    cfg, reg = register()
    if a.phase == "register":
        return
    for path in (*FILES, REG):
        if subprocess.check_output(["git", "show", f"HEAD:{path.relative_to(ROOT)}"], cwd=ROOT) != path.read_bytes():
            raise ValueError("Commit exact scope before acquiring new imagery")
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**items):
            event = dict(pid=os.getpid(), phase=a.phase, seconds=time.monotonic()-start, **items)
            frames.atomic(PRIVATE / "heartbeat.json", event)
            if items.get("completed", 1) % 25 == 0 or "completed" not in items:
                print(event, flush=True)

        mark = PUBLIC / f"{a.phase}_execution.json"
        if mark.exists():
            frames.checked(frames.read(mark)["index"])
            return
        if a.phase == "verify":
            frames.checked(frames.read(PUBLIC / "prepare_execution.json")["index"])
        out, counts = prepare(cfg, reg, a.resume, beat)
        frames.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-start, index=frames.artifact(out),
            frames=len(reg["roster"]["frames"]), **counts))


if __name__ == "__main__":
    main()
