"""Audit Okutama training labels and replay 8/12 causal motion baselines."""

import argparse
from collections import Counter, defaultdict
import fcntl
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from urllib.parse import urlsplit
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from scripts import run_m3w_opendd_optional_edits as optional
from src.evaluation.m3w_remote_zip import RangeFile
from src.world_model import m3w_radius_matched_turn_menu as motion

base, study, turn = optional.base, optional.study, optional.turn
OUT = ROOT/"outputs/publication_readiness_2026_09/okutama_label_histories_v1"
LOCAL = ROOT/"data/stage_cvpr2027_experiments/okutama_label_histories_v1"
SOURCE = ROOT/"external_data/Okutama_training_labels"
SIZE = 5_770_432_522
URL = ("https://www.dropbox.com/scl/fo/9qvpsb3fsamvqzsa12149/APTyV-f01XLnJ0WFpZSBLOE?"
       "dl=1&e=1&preview=TrainSetFrames.zip&rlkey=7u7131amaul29amyr4jbnnu03")
CLIP = re.compile(r"[12]\.[12]\.\d+")
MODES = ("released", "nongenerated")
STRIDES = (1, 4, 10)
ARMS = ("cv", "ols4", "turn", "hindcast")


class DropboxArchive(RangeFile):
    def span(self, start, length):
        if length == 0:
            return b""
        if start < 0 or length < 0 or start+length > self.size:
            raise ValueError("Range outside the registered training archive")
        for offset, raw in reversed(self.blocks):
            if offset <= start and start+length <= offset+len(raw):
                return raw[start-offset:start-offset+length]
        if length > self.max_read or self.transferred+length > self.budget:
            raise ValueError("Training archive range budget exceeded")
        tag = f"{start}-{start+length-1}"
        headers = {"Range": f"bytes={tag}", "Accept-Encoding": "identity"}
        with self.session.get(f"{self.url}&_range={tag}", headers=headers, stream=True, timeout=(10, 35)) as r:
            for response in (*r.history, r):
                host = urlsplit(response.url)
                if host.scheme != "https" or not (host.hostname == "www.dropbox.com" or
                        (host.hostname or "").endswith(".dropboxusercontent.com")):
                    raise ValueError("Unexpected archive redirect")
            if (r.status_code != 206 or r.headers.get("Content-Range") != f"bytes {tag}/{self.size}"
                    or r.headers.get("Content-Encoding", "identity") != "identity"):
                raise OSError("Publisher archive did not return the requested bounded range")
            raw = r.raw.read(length+1)
            if len(raw) != length:
                raise OSError("Truncated archive byte range")
        self.transferred += length
        self.requests += 1
        self.blocks.append((start, raw))
        return raw


def inventory(z):
    labels, frames, directory = {}, defaultdict(list), []
    seen = set()
    for item in z.infolist():
        path = PurePosixPath(item.filename)
        if (item.filename in seen or path.is_absolute() or ".." in path.parts or "\\" in item.filename
                or item.flag_bits & 1 or (item.external_attr >> 16) & 0o170000 == 0o120000):
            raise ValueError("Unexpected path or archive member type")
        seen.add(item.filename)
        directory.append([item.filename, item.header_offset, item.CRC, item.file_size, item.compress_size])
        if item.is_dir():
            continue
        if path.parts[:3] == ("Labels", "SingleActionTrackingLabels", "3840x2160"):
            if len(path.parts) != 4 or path.suffix != ".txt" or not CLIP.fullmatch(path.stem):
                raise ValueError("Unexpected tracking label member")
            if not 0 < item.file_size < 12_000_000:
                raise ValueError("Oversized tracking label member")
            labels[path.stem] = item
        elif path.suffix == ".jpg":
            clip = path.parent.name
            if not CLIP.fullmatch(clip) or not path.stem.isdecimal():
                raise ValueError("Unexpected frame numbering")
            frames[clip].append(int(path.stem))
    if len(labels) != 33 or set(labels) != set(frames):
        raise ValueError("Training tracking labels and frame groups disagree")
    for k, v in frames.items():
        frames[k] = sorted(v)
        if frames[k] != list(range(len(v))):
            raise ValueError("The recorded image indices are not contiguous from zero")
    return labels, dict(frames), sha256(study.json.dumps(directory, separators=(",", ":")).encode()).hexdigest()


def beat(phase, **fields):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(pid=os.getpid(), phase=phase, **fields))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def acquire():
    manifest = SOURCE/"manifest.json"
    if manifest.exists():
        return
    subprocess.run(["git", "check-ignore", "--quiet", str(manifest)], cwd=ROOT, check=True)
    if any(p.is_symlink() for p in (SOURCE, *SOURCE.parents)):
        raise ValueError("Refuse symlinked source directory")
    SOURCE.mkdir(parents=True, exist_ok=True)
    receipts = {}
    with DropboxArchive(URL, SIZE, budget=22_000_000, max_read=12_000_000) as remote, ZipFile(remote) as z:
        labels, frames, zip_digest = inventory(z)
        first = min(i.header_offset for i in labels.values())
        last = max(i.header_offset for i in labels.values())
        end = min((i.header_offset for i in z.infolist() if i.header_offset > last), default=z.start_dir)
        if end-first < 7_000_000 and all(i in labels.values() or i.is_dir()
                for i in z.infolist() if first <= i.header_offset < end):
            remote.span(first, end-first)
        for k, item in sorted(labels.items()):
            raw = z.read(item)  # ZipFile verifies CRC on the selected member.
            p = SOURCE/f"{k}.txt"
            if p.exists() and p.read_bytes() != raw:
                raise ValueError("Previously fetched member changed")
            if not p.exists():
                tmp = p.with_suffix(".pending")
                tmp.write_bytes(raw)
                tmp.replace(p)
            receipts[k] = dict(bytes=len(raw), crc32=item.CRC, sha256=base.sha(p))
            beat("acquire", clip=k, done=len(receipts), total=33)
        base.immutable(manifest, dict(url=URL, archive_bytes=SIZE, directory_sha256=zip_digest, labels=receipts,
            frames=frames, bytes_fetched=remote.transferred, requests=remote.requests,
            images_downloaded=0, test_payload_read=False))


def register():
    manifest = SOURCE/"manifest.json"
    m = study.read(manifest)
    if m["url"] != URL or m["archive_bytes"] != SIZE or len(m["labels"]) != 33 or m["test_payload_read"]:
        raise ValueError("Unexpected train-only source receipt")
    for k, ref in m["labels"].items():
        p = SOURCE/f"{k}.txt"
        if not CLIP.fullmatch(k) or base.sha(p) != ref["sha256"]:
            raise ValueError("Training label hash changed")
    p = OUT/"registration.json"
    base.immutable(p, dict(source=study.artifact(manifest), label_sha256={k: v["sha256"] for k, v in m["labels"].items()},
        code=[study.artifact(Path(x)) for x in (__file__, motion.__file__, motion.velocity.__code__.co_filename)],
        modes=MODES, strides=STRIDES, observed_steps=8, forecast_steps=12,
        windows="start at zero, skip 20 stride ticks per window; all histories from that window only",
        stride_10_role="additional annotation-cadence diagnostic, not a changed main evaluation protocol",
        roles="publisher training archive only, label-history diagnostic, no formal split assignment",
        paired_camera_group="part of day and scenario, from publisher naming",
        coordinate_unit="image pixels in 3840x2160 tracking label coordinates",
        native_frames_only=True, calibration_or_test_payload_read=False, no_policy_promotion=True))
    return base.sha(p), m


def parse(raw, frames):
    a = np.loadtxt(BytesIO(raw), dtype=np.int64, usecols=range(9), ndmin=2)
    if not len(a) or not np.isin(a[:, 5], frames).all() or not np.isin(a[:, 6:9], [0, 1]).all():
        raise ValueError("Unexpected tracking frame or flag")
    if len(np.unique(a[:, [5, 0]], axis=0)) != len(a):
        raise ValueError("Duplicate track-frame label")
    return a


def trajectories(a, count, mode):
    if mode not in MODES:
        raise ValueError("Unrecognized label mode")
    valid = (a[:, 6:8] == 0).all(1) & (a[:, 3:5] > a[:, 1:3]).all(1)
    if mode == "nongenerated":
        valid &= a[:, 8] == 0
    ids, track = np.unique(a[:, 0], return_inverse=True)
    points = np.full((count, len(ids), 2), np.nan)
    points[a[valid, 5], track[valid]] = (a[valid, 1:3]+a[valid, 3:5])*.5
    return points


def windows(count, stride):
    return np.arange(0, count-19*stride, 20*stride, dtype=int)


def prefixes(points, stride):
    start, ids, histories = [], [], []
    for t in windows(len(points), stride):
        p = points[t+np.arange(8)*stride].transpose(1, 0, 2)
        r = np.flatnonzero(np.isfinite(p).all((1, 2)))
        if len(r):
            start.append(np.full(len(r), t, int))
            ids.append(r)
            histories.append(p[r])
    return (np.concatenate(start), np.concatenate(ids), np.concatenate(histories)) if start else (
        np.empty(0, int), np.empty(0, int), np.empty((0, 8, 2)))


def forecasts(h):
    if not len(h):
        return {k: np.empty((0, 12, 2)) for k in ARMS}
    def three(x, horizon):
        cv, ols, _, _ = motion.rollout(x, horizon)
        return np.stack((cv, ols, motion.turn_path(x, horizon)), axis=1)
    predictions = three(h, 12)
    error = []
    for k in (4, 5, 6):
        z = three(h[:, :k], 8-k)
        err = np.linalg.norm(z-(h[:, None, k:]-h[:, None, k-1:k]), axis=-1).mean(-1)
        error.append(err)
    best = np.mean(error, axis=0).argmin(1)
    return dict(cv=predictions[:, 0], ols4=predictions[:, 1], turn=predictions[:, 2],
                hindcast=predictions[np.arange(len(h)), best])


def freeze(key, m):
    records = {}
    for k, frames in sorted(m["frames"].items()):
        a = parse((SOURCE/f"{k}.txt").read_bytes(), frames)
        files = {}
        for mode in MODES:
            p = trajectories(a, len(frames), mode)
            for stride in STRIDES:
                t, ids, h = prefixes(p, stride)
                name = f"{mode}_{stride}"
                files[name] = turn.exp.previous.save_bank(LOCAL/f"{k}_{name}.npz",
                    dict(start=t, ids=ids, h=h, **forecasts(h)))
        records[k] = files
        beat("freeze", clip=k, done=len(records), total=33)
    base.immutable(OUT/"freeze.json", dict(registration=key, clips=records,
        release_train_only=True, future_features_used=False, forecast_scoring_not_run=True))


def evaluate(p, z, stride):
    t, ids, h = z["start"], z["ids"], z["h"]
    y = p[t[:, None]+np.arange(8, 20)*stride, ids[:, None]]-h[:, -1:]
    valid = np.isfinite(y).all(-1)
    y = np.where(valid[..., None], y, 0.)
    complete = valid.all(1)
    data = dict(windows=len(windows(len(p), stride)), observed_rosters=len(np.unique(t)),
                histories=len(h), complete=int(complete.sum()), partial=int((valid.any(1) & ~complete).sum()),
                absent=int((~valid.any(1)).sum()),
                with_two_complete=sum(int((complete & (t == k)).sum()) >= 2 for k in np.unique(t)),
                complete_rosters=sum(bool(complete[t == k].all()) for k in np.unique(t)))
    scores = {}
    cv = z["cv"]
    for k in ARMS:
        q = z[k]
        err = np.linalg.norm(q-y, axis=-1)
        scores[k] = dict(known_ade=float(err[valid].mean()) if valid.any() else None,
                         fde=float(err[valid[:, -1], -1].mean()) if valid[:, -1].any() else None,
                         conservative_vs_cv_pp=None)
        mass = float(np.where(valid, np.linalg.norm(cv-y, axis=-1), 0.).sum())
        if mass > 0.:
            scores[k]["conservative_vs_cv_pp"] = float(100.*turn.exact.headroom.lower(cv, q, y, valid).sum()/mass)
    return dict(support=data, methods=scores)


def summarize(clips, arm):
    grouped = defaultdict(list)
    for k, views in clips.items():
        z = views[arm]
        grouped[k.split(".", 1)[1]].append(z)
    out = {k: sum(v[arm]["support"][k] for v in clips.values()) for k in next(iter(clips.values()))[arm]["support"]}
    out["scenario_groups"] = len(grouped)
    out["supported_scenario_groups"] = sum(any(v["support"]["with_two_complete"] for v in rows) for rows in grouped.values())
    scores = {}
    for method in ARMS:
        result = {}
        for metric in ("known_ade", "fde", "conservative_vs_cv_pp"):
            values = []
            for rows in grouped.values():
                observed = [r["methods"][method][metric] for r in rows if r["methods"][method][metric] is not None]
                if observed:
                    values.append(np.mean(observed))
            result[metric] = float(np.mean(values)) if values else None
            if metric == "conservative_vs_cv_pp":
                result["estimable_scenarios"] = len(values)
                result["worst_scenario_gain_pp"] = float(min(values)) if values else None
                if values:
                    draws = np.random.default_rng(3183).choice(values, size=(3000, len(values))).mean(1)
                    result["scenario_bootstrap_ci95_pp"] = np.quantile(draws, [.025, .975]).tolist()
        scores[method] = result
    return dict(support=out, methods=scores)


def score(key, m):
    fp = OUT/"freeze.json"
    frozen = study.read(fp)
    if (frozen["registration"] != key or not frozen["forecast_scoring_not_run"]
            or frozen["future_features_used"] or set(frozen["clips"]) != set(m["frames"])):
        raise ValueError("Require all label histories frozen before outcome scoring")
    results, flag_counts, residue = {}, Counter(), np.zeros(10, dtype=int)
    for k, frames in sorted(m["frames"].items()):
        a = parse((SOURCE/f"{k}.txt").read_bytes(), frames)
        flag_counts.update(dict(rows=len(a), generated=int(a[:, 8].sum()), lost=int(a[:, 6].sum()),
                                occluded=int(a[:, 7].sum())))
        clean = (a[:, 6:9] == 0).all(1)
        residue += np.bincount(a[clean, 5] % 10, minlength=10)
        results[k] = {}
        for mode in MODES:
            p = trajectories(a, len(frames), mode)
            for stride in STRIDES:
                arm = f"{mode}_{stride}"
                with np.load(study.checked(frozen["clips"][k][arm]), allow_pickle=False) as z:
                    for t, ids, h in zip(z["start"], z["ids"], z["h"], strict=True):
                        np.testing.assert_array_equal(p[t+np.arange(8)*stride, ids], h)
                    results[k][arm] = evaluate(p, z, stride)
        beat("score", clip=k, done=len(results), total=33)
    arms = [f"{mode}_{stride}" for mode in MODES for stride in STRIDES]
    total = {a: summarize(results, a) for a in arms}
    base.immutable(OUT/"readout.json", dict(registration=key, freeze=study.artifact(fp), result_source="fresh_run",
        clips=results, totals=total, flags=dict(flag_counts), archive_directory_sha256=m["directory_sha256"],
        nongenerated_visible_rows_modulo_10=residue.tolist(),
        bytes_fetched=m["bytes_fetched"], requests=m["requests"], images_read=0, neural_training=False,
        diagnostics_only=True, independent_confirmation=False, reserved_outcomes_opened=False,
        all_views_from_same_facility=True, risk_gate_not_evaluated=True))
    print(study.json.dumps(dict(flags=dict(flag_counts), totals=total), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("acquire", "freeze", "score"))
    phase = parser.parse_args().phase
    for p in (OUT, LOCAL):
        p.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if phase == "acquire":
            acquire()
        else:
            key, manifest = register()
            (freeze if phase == "freeze" else score)(key, manifest)


if __name__ == "__main__":
    main()
