"""First frozen 8-to-12 transfer to the Hannover shared-space release."""

import argparse
import csv
import fcntl
import hashlib
from io import StringIO
import os
from pathlib import Path
import platform
import re
import sys
import time
from zipfile import ZipFile

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_tumdot_matched_transfer_capacity as ceiling
from scripts import run_m3w_matched_path_choice as choice_study

transfer = ceiling.study
run, np, torch, source = transfer.run, transfer.np, transfer.torch, transfer.source
OUT = transfer.OUT.parent / "hannover_frozen_transfer_v1"
PRIVATE = transfer.PRIVATE.parent / OUT.name
ZIP = source.ROOT / "external_data/HannoverSharedSpace_source_audit/trajectory_data.zip"
INTAKE = OUT.parent / "station_sharedspace_intake_v1/analysis.json"
CLIPS = ("D1", "D2", "D3")
PREFIX = "trajectory_data/"
CONTROLS = ("cv", "ols4", "full_hindcast", "original_blend05")
ARMS = (*CONTROLS, *transfer.ARMS, "frozen_full_choice", "frozen_action_choice", "oracle_utility", "oracle_min_risk")


def publisher_contract(doc):
    scale = re.search(r"Ratio from pixel to meter\s+([\d.]+):1", doc)
    dt = re.search(r"Each time step lasts ([\d.]+) seconds", doc)
    if scale is None or dt is None:
        raise ValueError("Published scale or clock not found")
    return float(scale[1]), float(dt[1])


def register():
    parent, frozen, _ = choice_study.register()
    p = choice_study.OUT / "head_freeze.json"
    heads = run.read(p)
    if heads["registration"] != parent or heads["target_labels_used"]:
        raise ValueError("Require frozen OpenDD-only selectors")
    a = run.read(INTAKE)["hannover"]
    zip_binding = run.artifact(ZIP)
    if any(zip_binding[k] != a["asset"][k] for k in ("sha256", "bytes")):
        raise ValueError("Publisher archive differs from structural intake")
    with ZipFile(ZIP) as z:
        metadata = z.read(PREFIX+"readme.txt")
    scale, dt = publisher_contract(metadata.decode("utf-8-sig"))
    if dt != a["annotation_tick_seconds"] or not 1. < scale < 100.:
        raise ValueError("Unexpected published clock or scalar")
    reg = OUT / "registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), source_asset=zip_binding, metadata_sha256=hashlib.sha256(metadata).hexdigest(),
        intake=run.artifact(INTAKE), matched_path_heads=run.artifact(p), clips=CLIPS, source_groups=1,
        steps=[8, 12], native_stride=1, step_seconds=dt, pixel_per_publisher_unit=scale,
        forecasts="frozen OpenDD ensembles and OpenDD-only choice heads; no Hannover fit or selection",
        primary="half radius fixed tangent vs matched hindcast and CV on full grid",
        auxiliary="quarter radius, frozen choice heads, utility and risk hindsight ceiling",
        no_new_official_split=True, original_2pct=True, target_tuning=False, confirmation_access=False,
        known_label_utility_is_secondary=True, all_clips_required=True, seeds=choice_study.SEEDS)))
    return run.sha(reg), frozen, heads, a, scale, dt


def parse_csv(raw):
    rows = csv.reader(StringIO(raw.decode("utf-8-sig")))
    if next(rows) != ["time_step", "object_id", "x_coordinate", "y_coordinate", "user_type"]:
        raise ValueError("Unexpected Hannover trajectory fields")
    tracks, categories, low, high = {}, {}, np.inf, -np.inf
    for tick, key, x, y, kind in rows:
        t, k = int(tick), int(key)
        if kind not in ("1", "2", "3") or categories.setdefault(k, kind) != kind:
            raise ValueError("Road user changed class")
        low, high = min(low, t), max(high, t)
        if kind == "1":
            xy = float(x), float(y)
            a = tracks.setdefault(k, {})
            if t in a or not np.isfinite(xy).all():
                raise ValueError("Duplicate or nonfinite pedestrian sample")
            a[t] = xy
    if len(tracks) < 2:
        raise ValueError("Expected multi-agent pedestrian support")
    origin = int(low)
    grid = np.full((len(tracks), int(high)-origin+1, 2), np.nan)
    signatures = set()
    for j, k in enumerate(sorted(tracks)):
        t = sorted(tracks[k])
        x = np.array([tracks[k][i] for i in t])
        grid[j, np.asarray(t)-origin] = x
        h = hashlib.sha256(np.asarray(t, dtype="<i8").tobytes()+x.astype("<f8").tobytes()).hexdigest()
        signatures.add(h)
    return grid, tracks, signatures


def windows(grid, scale):
    h, y, ids = [], [], []
    for start, at in source.cohorts(np.isfinite(grid).all(2), 1):
        p = grid[at, start:start+20]
        h.append(p[:, :8]/scale)
        y.append((p[:, 8:]-p[:, 7:8])/scale)
        ids.append(start)
    if not ids:
        raise ValueError("No nonoverlapping shared query")
    return np.concatenate(h), np.concatenate(y), ids


def load_heads(heads):
    out = {}
    for seed in choice_study.SEEDS:
        out[seed] = {}
        for variant, meta in heads["fit"][str(seed)]["heads"].items():
            z = torch.load(run.checked(meta["checkpoint"]), weights_only=True, map_location="cpu")
            if run.stable(z["identity"]) != meta["identity"] or z["step"] != choice_study.STEPS:
                raise ValueError("Frozen selector lineage changed")
            net = choice_study.model.ActionValue()
            net.load_state_dict(z["head"])
            out[seed][variant] = net.eval()
    return out


@torch.inference_mode()
def predict(h, nets, selectors):
    cv, ols, base, _, delta = transfer.forecast(h, nets)
    _, _, hindcast, _ = source.motion_bank(h, transfer.motion)
    output = {}
    for part, fraction in transfer.FRACTIONS.items():
        paths, on = transfer.matched_paths(cv, base, delta, fraction)
        names = dict(cv=cv, ols4=ols, full_hindcast=hindcast, original_blend05=base, **paths)
        bank, x, _ = choice_study.prepare(h, cv, paths, on)
        tensor = torch.from_numpy(x)
        for arm, head in selectors.items():
            feat = tensor if arm == "full" else choice_study.action_only(tensor)
            idx = choice_study.model.select(head(feat).numpy(), choice_study.ANCHOR)["agreement"]
            a = cv.copy()
            a[on] = bank[np.flatnonzero(on), idx]
            np.testing.assert_array_equal((a != cv).any((1, 2)), on)
            np.testing.assert_allclose(np.linalg.norm(a-cv, axis=-1).max(1),
                                       np.linalg.norm(paths["hindcast"]-cv, axis=-1).max(1), atol=1e-13, rtol=1e-7)
            names[f"frozen_{arm.removesuffix('_only')}_choice"] = a
        output[part] = names
    return output


def score(clip, pred, y, valid, cuts, scale):
    cv, ols, match = (pred[k] for k in ("cv", "ols4", "hindcast"))
    options = [pred[k] for k in transfer.ARMS]
    # These two paths use future labels only as diagnostic ceilings.
    pred = dict(pred, oracle_utility=ceiling.best_utility(cv, options, y, valid),
                oracle_min_risk=ceiling.best_risk(cv, options, y, valid))
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    result = {}
    for k in ARMS:
        a = pred[k]
        row = source.score(a, cv, ols, y, valid, clip, cuts, match)
        row["lower_pp"]["full_hindcast"] = float(100*transfer.paired(pred["full_hindcast"], a, y, valid)[1].sum()/mass)
        row["available_ade"] *= scale
        row["available_fde"] *= scale
        row["known_pp_vs_cv"] = float(100*transfer.paired(cv, a, y, valid)[0].sum()/mass)
        result[k] = row
    return result


def execute(reg, frozen, fit, intake, scale, dt, beat, cache):
    nets = transfer.frozen_models(*frozen[:5])
    selectors = load_heads(fit)
    cuts = np.asarray(frozen[-1]["open_dd_positive_complete_cv_cuts"])
    results, overlap, prior_ids, prior_sigs = {}, {}, set(), set()
    with ZipFile(ZIP) as z:
        for clip in CLIPS:
            raw = z.read(PREFIX+f"{clip}_trajectory.csv")
            meta = intake["clips"][clip]
            if hashlib.sha256(raw).hexdigest() != meta["sha256"]:
                raise ValueError("Trajectory member hash changed")
            grid, ids, sigs = parse_csv(raw)
            if len(ids) != meta["tracks"]["1"] or sigs & prior_sigs:
                raise ValueError("Unexpected counts or duplicate pedestrian trajectory across files")
            overlap[clip] = len(ids.keys() & prior_ids)
            prior_ids.update(ids)
            prior_sigs.update(sigs)
            h, y, starts = windows(grid, scale)
            valid = np.isfinite(y).all(2)
            out = cache / f"{clip}.json"
            if not out.exists():
                scores = []
                for seed in choice_study.SEEDS:
                    pred = predict(h, nets[seed], selectors[seed])
                    stamp = run.kin.prior.source.array_hash
                    forecast = {part: {k: stamp(a) for k, a in v.items()} for part, v in pred.items()}
                    run.once(cache / f"{clip}_{seed}_forecasts.json", run.stable(dict(registration=reg, arrays=forecast)))
                    scores.append(dict(seed=seed, **{part: score(clip, paths, y, valid, cuts, scale) for part, paths in pred.items()}))
                run.once(out, run.stable(dict(registration=reg, clip=clip, queries=len(starts), targets=len(h),
                    complete12=int(valid.all(1).sum()), unknown=int((~valid.any(1)).sum()), scores=scores)))
            cell = run.read(out)
            if cell["registration"] != reg or cell["clip"] != clip:
                raise ValueError("Clip checkpoint has wrong registration")
            results[clip] = cell
            beat(clip=clip, complete=len(results))
    summary = {part: source.mean_summary([dict(arms=v[part]) for r in results.values() for v in r["scores"]])
               for part in transfer.FRACTIONS}
    run.once(OUT / "analysis.json", run.stable(dict(registration=reg, step_seconds=dt, source_groups=1, by_clip=results,
        summary=summary, repeated_numeric_ids=overlap, exact_cross_clip_track_duplicates=0,
        no_training=True, no_target_tuning=True, published_scalar_used_for_input=True, reported_errors="image pixels",
        original_2pct=True, formal_confirmation=False, confirmation_access=False, deployment=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("register", "run", "replay"))
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bound = register()
        if args.phase == "register":
            print(bound[0], flush=True)
            return

        def beat(**state):
            v = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **state)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", v)
            print(v, flush=True)

        if args.phase == "replay":
            from tempfile import TemporaryDirectory
            with TemporaryDirectory(dir=PRIVATE) as d:
                execute(*bound, beat, Path(d))
        else:
            execute(*bound, beat, PRIVATE)
        beat(phase="complete", replay=args.phase == "replay")


if __name__ == "__main__":
    main()
