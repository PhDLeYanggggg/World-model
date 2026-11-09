"""Paired past-image camera compensation on the frozen Okutama training roster."""

import argparse
from collections import Counter, defaultdict
import fcntl
from hashlib import sha256
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from zipfile import ZipFile

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2

from scripts import run_m3w_okutama_label_histories as source
from src.world_model import m3w_camera_association as association
from src.world_model import m3w_causal_camera_motion as camera

np, study, base, turn = source.np, source.study, source.base, source.turn
OUT = source.OUT.parent/"okutama_past_camera_v1"
LOCAL = source.LOCAL.parent/OUT.name
STRIDE = 10
BYTES = 60_000_000
ARMS = camera.ARMS
REFERENCE = dict(raw_ols4="raw_cv", stable_cv="raw_cv", camera_cv="raw_cv",
                 stable_ols4="raw_ols4", camera_ols4="raw_ols4")


def roster(frozen):
    plan = {}
    for clip, inputs in sorted(frozen["clips"].items()):
        with np.load(study.checked(inputs["nongenerated_10"]), allow_pickle=False) as z:
            t, counts = np.unique(z["start"], return_counts=True)
        keep = t[counts >= 2]
        if len(keep):
            start = int(keep[0])
            plan[clip] = dict(start=start, past=[start+STRIDE*i for i in range(8)],
                              scenario=clip.split(".", 1)[1], frozen_prefix=inputs["nongenerated_10"])
    if not plan:
        raise ValueError("No past-complete multi-agent windows")
    return plan


def register():
    parent, archive = source.register()
    path = source.OUT/"freeze.json"
    frozen = study.read(path)
    if (frozen["registration"] != parent or frozen["future_features_used"]
            or not frozen["forecast_scoring_not_run"] or archive["test_payload_read"]):
        raise ValueError("Require exact train-only prefix and archive")
    plan = roster(frozen)
    dst = OUT/"registration.json"
    base.immutable(dst, dict(parent=study.artifact(path), code=[study.artifact(Path(p)) for p in
        (__file__, camera.__file__, association.__file__, source.__file__)], plan=plan,
        window="earliest nongenerated stride-10 roster with at least two observed agents per clip",
        image_size=[1280, 720], annotation_image_size=[3840, 2160], observed_steps=8, forecast_steps=12,
        image_transfer_budget_bytes=BYTES, opencv_version=cv2.__version__, opencv_threads=1,
        camera_fit="registered background similarity: masks use only visible nongenerated past boxes",
        camera_rollout="last-three median center shift, angle and log scale; 0.9 damp per future step",
        fallback="seven accepted pairs required; exact corresponding raw forecast otherwise",
        future_images_accessed=False, diagnostics_only=True, independent_confirmation=False))
    return base.sha(dst), archive, plan


def beat(phase, **fields):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(pid=os.getpid(), phase=phase, **fields))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def image_path(clip, frame):
    return LOCAL/"images"/clip/f"{frame}.jpg"


def receipt(clip, frame, member, key, raw):
    return dict(clip=clip, frame=frame, member=member, registration=key, bytes=len(raw),
                sha256=sha256(raw).hexdigest())


def acquire(key, manifest, plan):
    subprocess.run(["git", "check-ignore", "--quiet", str(image_path(next(iter(plan)), 0))], cwd=ROOT, check=True)
    if shutil.disk_usage(ROOT).free < 2_000_000_000 or any(p.is_symlink() for p in (LOCAL, *LOCAL.parents)):
        raise OSError("Insufficient local space or unsafe cache path")
    requested = {(c, i) for c, p in plan.items() for i in p["past"]}
    done, transferred, nrequests = {}, 0, 0
    for c, i in sorted(requested):
        p = image_path(c, i)
        r = p.with_suffix(".json")
        if r.exists():
            entry = study.read(r)
            if entry != receipt(c, i, entry["member"], key, p.read_bytes()):
                raise ValueError("Cached image hash or role changed")
            done[f"{c}_{i}"] = entry
    if len(done) != len(requested):
        with source.DropboxArchive(source.URL, source.SIZE, budget=BYTES, max_read=12_000_000) as remote, ZipFile(remote) as z:
            _, _, digest = source.inventory(z)
            if digest != manifest["directory_sha256"]:
                raise ValueError("Publisher archive directory changed")
            items = {}
            directory = sorted(z.infolist(), key=lambda x: x.header_offset)
            for item, next_item in zip(directory[:-1], directory[1:], strict=True):
                path = Path(item.filename)
                if path.suffix == ".jpg" and (k := (path.parent.name, int(path.stem))) in requested:
                    if k in items:
                        raise ValueError("Duplicate archive image key")
                    items[k] = (item, next_item.header_offset-item.header_offset)
            if set(items) != requested:
                raise ValueError("The source archive does not cover the registered past roster")
            for c, i in sorted(requested):
                name = f"{c}_{i}"
                if name in done:
                    continue
                item, span = items[c, i]
                if not 0 < item.file_size <= 1_500_000 or not 0 < span <= 1_600_000:
                    raise ValueError("Unexpected past image member size")
                remote.span(item.header_offset, span)
                raw = z.read(item)
                p = image_path(c, i)
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(".pending")
                tmp.write_bytes(raw)
                tmp.replace(p)
                done[name] = receipt(c, i, item.filename, key, raw)
                base.immutable(p.with_suffix(".json"), done[name])
                beat("acquire", last=name, done=len(done), total=len(requested), transferred=remote.transferred)
                if len(done) % 16 == 0:
                    print("past images", len(done), "/", len(requested), flush=True)
            transferred, nrequests = remote.transferred, remote.requests
    p = OUT/"image_freeze.json"
    if p.exists():
        if study.read(p)["images"] != done:
            raise ValueError("Frozen image roster differs")
    else:
        base.immutable(p, dict(registration=key, images=done, bytes_last_invocation=transferred, requests_last_invocation=nrequests,
            member_bytes=sum(r["bytes"] for r in done.values()), image_role="past_only"))


def gray(key, clip, t, records):
    p = image_path(clip, t)
    row, raw = records[f"{clip}_{t}"], p.read_bytes()
    if row != receipt(clip, t, row["member"], key, raw):
        raise ValueError("Past image failed verification")
    result = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if result is None or result.shape != (720, 1280):
        raise ValueError("Expected the publisher's 1280x720 training raster")
    return result


def boxes(a, frame):
    at = (a[:, 5] == frame) & (a[:, 6:9] == 0).all(1) & (a[:, 3:5] > a[:, 1:3]).all(1)
    return a[at, 1:5]/3.


def freeze(key, manifest, plan):
    path = OUT/"image_freeze.json"
    image = study.read(path)
    if image["registration"] != key or image["image_role"] != "past_only":
        raise ValueError("Require exact acquired past-image roster")
    pred, motion = {}, {}
    for clip, row in plan.items():
        t = row["past"]
        a = source.parse((source.SOURCE/f"{clip}.txt").read_bytes(), manifest["frames"][clip])
        im = [gray(key, clip, i, image["images"]) for i in t]
        transforms, diagnostics = [], []
        for i in range(7):
            cv2.setRNGSeed(27+i)
            m, v = association.camera_step(im[i], im[i+1], boxes(a, t[i]), boxes(a, t[i+1]))
            transforms.append(m)
            diagnostics.append(dict(before=t[i], after=t[i+1], **v))
        with np.load(study.checked(row["frozen_prefix"]), allow_pickle=False) as z:
            at = z["start"] == row["start"]
            h, ids = z["h"][at], z["ids"][at]
            paths = camera.compare(h/3., transforms, [640., 360.], [v["accepted"] for v in diagnostics])
            paths = {k: p*3. for k, p in paths.items()}
            np.testing.assert_allclose(paths["raw_cv"], z["cv"][at], atol=1e-9, rtol=1e-9)
            np.testing.assert_allclose(paths["raw_ols4"], z["ols4"][at], atol=1e-9, rtol=1e-9)
        pred[clip] = turn.exp.previous.save_bank(LOCAL/f"{clip}_pred.npz", dict(h=h, ids=ids, **paths))
        motion[clip] = dict(pairs=diagnostics, active=all(v["accepted"] for v in diagnostics))
        beat("freeze", clip=clip, done=len(pred), total=len(plan))
    base.immutable(OUT/"forecast_freeze.json", dict(registration=key, images=study.artifact(path),
        predictions=pred, motion=motion, future_images_accessed=False, future_scores_not_run=True))


def score(key, manifest, plan):
    path = OUT/"forecast_freeze.json"
    f = study.read(path)
    if f["registration"] != key or f["future_images_accessed"] or not f["future_scores_not_run"] or set(f["predictions"]) != set(plan):
        raise ValueError("Require all past predictions before scoring")
    rows, stats = [], Counter()
    for clip, r in plan.items():
        a = source.parse((source.SOURCE/f"{clip}.txt").read_bytes(), manifest["frames"][clip])
        points = source.trajectories(a, len(manifest["frames"][clip]), "nongenerated")
        with np.load(study.checked(f["predictions"][clip]), allow_pickle=False) as z:
            h, ids = z["h"], z["ids"]
            y = points[r["start"]+np.arange(8, 20)*STRIDE, :][:, ids].transpose(1, 0, 2)-h[:, -1:]
            v = np.isfinite(y).all(-1)
            y = np.where(v[..., None], y, 0.)
            error = {k: np.linalg.norm(z[k]-y, axis=-1) for k in ARMS}
            m = {}
            for arm in ARMS:
                ref = REFERENCE.get(arm, "raw_ols4" if arm.endswith("ols4") else "raw_cv")
                mass = error[ref][v].sum()
                ade = np.mean(np.divide((error[arm]*v).sum(1), v.sum(1).clip(1))[v.any(1)]) if v.any() else np.nan
                fde = error[arm][v[:, -1], -1].mean() if v[:, -1].any() else np.nan
                lower = float(turn.exact.headroom.lower(z[ref], z[arm], y, v).sum()*100./mass) if mass > 0. else np.nan
                p95 = np.quantile((error[arm]*v).sum(1)[v.any(1)]/v.sum(1)[v.any(1)], .95) if v.any() else np.nan
                pair = turn.expanded.frontier.pair_reduction(z[arm][None], z[ref], y, v)[0]
                if not (v.sum(0) >= 2).any():
                    pair[0] = np.nan
                m[arm] = [float(x) if np.isfinite(x) else None for x in (ade, fde, lower, p95, *pair)]
            stats.update(windows=1, histories=len(h), complete=int(v.all(1).sum()),
                         whole_roster=int(v.all()), with_two_complete=int(v.all(1).sum() >= 2))
        details = f["motion"][clip]
        active = details["active"]
        stats.update(active=int(active), accepted_steps=sum(r["accepted"] for r in details["pairs"]),
                     steps=len(details["pairs"]))
        rows.append(dict(clip=clip, scenario=r["scenario"], active=active,
                         complete=int(v.all(1).sum()), arms=m))
    # Equal-weight camera means on common metric support, paired by scenario.
    result = summarize(rows)
    active = summarize([r for r in rows if r["active"]])
    base.immutable(OUT/"readout.json", dict(registration=key, forecast_freeze=study.artifact(path),
        result_source="fresh_run", support=dict(stats), all_windows=result, accepted_only=active, by_clip=rows,
        future_images_accessed=False, independent_confirmation=False, physical_safety_evaluated=False,
        train_only=True, reserved_outcomes_opened=False))
    print(study.json.dumps(dict(support=dict(stats), all_windows=result, accepted_only=active), indent=2))


def summarize(rows):
    names = ("known_ade_px", "available_fde_px", "lower_gain_vs_matched_raw_pp", "p95_known_ade_px",
             "pair_gain_known_px", "pair_gain_lower_px")
    out = {}
    for arm in ARMS:
        out[arm] = {}
        for i, name in enumerate(names):
            grouped = defaultdict(list)
            for r in rows:
                value = r["arms"][arm][i]
                if value is not None:
                    grouped[r["scenario"]].append(value)
            means = [np.mean(x) for x in grouped.values()]
            if not means:
                out[arm][name] = None
                continue
            result = turn.effects.interval(means)
            result["scenario_groups"] = result.pop("sites")
            result["resampling_unit"] = "paired camera scenario, not independent site"
            out[arm][name] = result
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("acquire", "freeze", "score"))
    phase = p.parse_args().phase
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    for d in (OUT, LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, manifest, plan = register()
        dict(acquire=acquire, freeze=freeze, score=score)[phase](key, manifest, plan)


if __name__ == "__main__":
    main()
