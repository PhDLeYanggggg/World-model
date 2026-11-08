"""Test static-map path selection without changing the held-location forecast bank."""

import argparse
import fcntl
import hashlib
from io import BytesIO
import os
from pathlib import Path
import platform
import sys
import time
from zipfile import ZipFile

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use the native macOS arm64 .venv-pytorch interpreter")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from scripts import audit_m3w_prefix_path_assignment as shuffle
from scripts import run_m3w_prefix_path_replay as prior
from src.world_model import m3w_opendd_map_direction as method

run, np, torch, source, transfer = prior.run, prior.np, prior.torch, prior.source, prior.prior
OUT = prior.OUT.parent / "opendd_map_direction_v1"
PRIVATE = prior.PRIVATE.parent / OUT.name
RULES = ("map_persistence", "map_continuation")
REFERENCES = ("cv", "ols4", "hindcast", "tangent")


def map_files(site):
    with ZipFile(source.prior.ARCHIVES / f"opendd_v3-{site}.zip") as z:
        im, = (p for p in z.namelist() if Path(p).name == f"{site}_drivable_area.png")
        loc, = (p for p in z.namelist() if Path(p).stem == site and Path(p).suffix in (".pgw", ".tfw"))
        return {im: z.read(im), loc: z.read(loc)}


def map_binding(files):
    return {name: hashlib.sha256(b).hexdigest() for name, b in files.items()}


def register():
    parent, frozen, _ = prior.register()
    file = prior.OUT / "analysis.json"
    previous = run.read(file)
    if previous["registration"] != parent or previous["confirmation_access"]:
        raise ValueError("Require the exact exposed held-location roster")
    maps = {s: map_binding(map_files(s)) for s in source.SITES}
    f = OUT / "registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), method=run.artifact(Path(method.__file__)),
        same_count_control=run.artifact(Path(shuffle.__file__)), parent=run.artifact(file), static_assets=maps,
        primary="map_continuation_vs_matched_hindcast_tangent_and_same_count_shuffle_at_half",
        obs_pred=[8, 12], source="exposed OpenDD only", seeds=prior.SEEDS, fractions=transfer.FRACTIONS,
        map_is_vehicle_drivable_area_not_pedestrian_truth=True, same_recipients=True, same_radius=True,
        raster_threshold_uint8=128, original_2pct=True, no_training=True, no_future_inputs=True, confirmation_access=False)))
    return run.sha(f), frozen, previous, maps


def raster_for(site, expected):
    files = map_files(site)
    if map_binding(files) != expected:
        raise ValueError("Static map registration changed")
    pic, affine = files.values()
    with Image.open(BytesIO(pic)) as img:
        image = np.asarray(img)
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Expected an 8-bit grayscale drivable-area raster")
    fringe = float(np.mean((image != 0) & (image != 255)))
    if fringe >= .01:
        raise ValueError("Map is not a black-and-white mask with a small grayscale fringe")
    return method.RoadDistance(image >= 128, list(map(float, affine.split()))), fringe


def evaluate(h, y, valid, rec, cuts, pred, raster, part, hist):
    cv, ols, inc, _, deltas = pred
    paths, on = transfer.matched_paths(cv, inc, deltas, transfer.FRACTIONS[part])
    bank = np.stack([paths[a] for a in prior.ARMS], axis=1)
    choice, supported, distance = method.choose(h, bank, on, raster, prior.REFERENCE)
    values = dict(cv=cv, ols4=ols, hindcast=paths["hindcast"], tangent=paths["tangent"])
    for k, c in choice.items():
        a = bank[np.arange(len(bank)), c]
        np.testing.assert_array_equal((a != cv).any((1, 2)), on)
        np.testing.assert_allclose(np.linalg.norm(a-cv, axis=-1).max(1),
                                   np.linalg.norm(paths["hindcast"]-cv, axis=-1).max(1), atol=1e-13, rtol=1e-7)
        values[k] = a
    empty = {}
    result = prior.score(values, bank, on, empty, choice, y, valid, rec, cuts)
    for k in ("cv", "ols4", "hindcast", "tangent"):
        if result["arms"][k] != hist["arms"][k]:
            raise ValueError("Existing held-location control changed")
    support = dict(edited=int(on.sum()), fully_mapped=int(supported.sum()),
                   ends_on_road=int((supported & (distance[:, -1] < 0.)).sum()),
                   past_crosses_boundary=int((supported & (distance.min(1) < 0.) & (distance.max(1) > 0.)).sum()),
                   switch_from_matched_hindcast={k: int((on & (c != prior.REFERENCE)).sum()) for k, c in choice.items()})
    result["map_support"] = support
    result["assignment"] = {k: shuffle.cell(cv, bank, on, c, y, valid, rec) for k, c in choice.items()}
    return result


def execute(key, frozen, previous, maps, beat, cache):
    nets = transfer.frozen_models(*frozen[:5])
    data, errors = {}, {}
    for site in source.SITES:
        with np.load(source.prior.PRIVATE / f"{site}_cohort.npz", allow_pickle=False) as z:
            h, y, rec = z["history"], z["future"], np.repeat(z["recording"], np.diff(z["offsets"]))
        valid = np.isfinite(y).all(2)
        cv = transfer.motion.rollout(h, 12)[0]
        a = np.linalg.norm(cv[valid.all(1)]-y[valid.all(1)], axis=-1).mean(1)
        data[site], errors[site] = (h, y, valid, rec), a[a > 0.]
    results, geometry = {}, {}
    for j, site in enumerate(source.SITES):
        p = cache / f"{site}.json"
        if not p.exists():
            raster, fringe = raster_for(site, maps[site])
            h, y, valid, rec = data[site]
            cuts = np.quantile(np.concatenate([errors[s] for s in source.SITES if s != site]), [.25, .75])
            rows = []
            for i, seed in enumerate(prior.SEEDS):
                pred = transfer.forecast(h, [nets[seed][j]])
                old = previous["groups"]["opendd"][site][i]
                rows.append(dict(seed=seed, **{part: evaluate(h, y, valid, rec, cuts, pred, raster, part, old[part])
                                               for part in transfer.FRACTIONS}))
            run.once(p, run.stable(dict(registration=key, site=site, rows=rows, affine_axis_cosine=raster.axis_cosine,
                                        grayscale_fringe=fringe,
                                        raster_shape=list(raster.value.shape))))
        a = run.read(p)
        if a["registration"] != key or a["site"] != site:
            raise ValueError("Map diagnostic checkpoint mismatch")
        results[site], geometry[site] = a["rows"], {k: a[k] for k in ("affine_axis_cosine", "raster_shape", "grayscale_fringe")}
        beat(site=site, sites_done=len(results))
    summary = {}
    for part in transfer.FRACTIONS:
        draws = np.random.default_rng(817).integers(7, size=(3000, 7))

        def stat(get):
            a = np.array([np.mean([get(r[part]) for r in rows]) for rows in results.values()])
            return dict(point=float(a.mean()), interval95=np.quantile(a[draws].mean(1), [.025, .975]).tolist())

        flat = [r[part] for rows in results.values() for r in rows]
        summary[part] = dict(arms={k: dict(lower={b: stat(lambda r: r["arms"][k]["lower"][b]) for b in REFERENCES},
            risk={f: sum(r["arms"][k][f] for r in flat) for f in ("pass_views", "views", "undefined_easy")})
            for k in (*REFERENCES, *RULES)}, assignment={k: {f: stat(lambda r: r["assignment"][k][f])
            for f in flat[0]["assignment"][k]} for k in RULES},
            support={k: sum(r["map_support"][k] for r in flat) for k in ("edited", "fully_mapped", "ends_on_road", "past_crosses_boundary")},
            switches={k: sum(r["map_support"]["switch_from_matched_hindcast"][k] for r in flat) for k in RULES})
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, summary=summary, by_site=results, geometry=geometry,
        original_2pct=True, no_training=True, control_replay_exact=True, confirmation_access=False, deployment=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**state):
            a = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **state)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", a)
            print(a, flush=True)

        keys = register()
        if args.replay:
            from tempfile import TemporaryDirectory
            with TemporaryDirectory(dir=PRIVATE) as tmp:
                execute(*keys, beat, Path(tmp))
        else:
            execute(*keys, beat, PRIVATE)
        beat(phase="complete", replay=args.replay)


if __name__ == "__main__":
    main()
