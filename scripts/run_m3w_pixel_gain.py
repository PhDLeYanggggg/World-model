"""Extract past-only foreground cues and test them on held fitting sources."""

import argparse
import csv
import fcntl
import io
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

if sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 before importing numerical libraries")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_fit_past_pixels as pixels
from src.world_model import m3w_foreground_motion as foreground

prior, frames = pixels.conditional, pixels.frames
base, run, np, torch = prior.base, prior.run, prior.np, prior.torch
ade_from = prior.prior.prior.prior.ade
PUBLIC = pixels.PUBLIC.parent / "native_pixel_gain_v1"
PRIVATE = pixels.PRIVATE.parent / PUBLIC.name
REG = PUBLIC / "registration.json"
FILES = (Path(__file__), Path(foreground.__file__), ROOT / "tests/test_m3w_foreground_motion.py",
         PUBLIC / "protocol.md")
ARMS = ("no_image", "background", "foreground")
PAIRS = (("background", "no_image"), ("foreground", "background"), ("foreground", "no_image"),
         ("foreground", "ols4"), ("foreground", "support_only"), ("foreground", "minimum"))
STEPS = 500


def register():
    cut, jobs, old = prior.register()
    prior.prev.receipt(prior.PUBLIC, "replay")
    cfg, image = pixels.register()
    prepared, replay = (frames.read(pixels.PUBLIC / f"{s}_execution.json") for s in ("prepare", "verify"))
    if prepared["index"] != replay["index"] or replay["fresh_images"] or replay["range_bytes"]:
        raise ValueError("Expected complete independent local image verification")
    names = (*FILES, prior.REG, prior.PUBLIC / "replay_execution.json",
             pixels.REG, pixels.PUBLIC / "verify_execution.json")
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in names], arms=ARMS,
        pairs=PAIRS, steps=STEPS, seeds=prior.prev.SEEDS, same_support=True,
        prefix_images_only=True, sources="fixed_fitting_only", original_rule_unchanged=True)))
    return cut, jobs, old, cfg, image, frames.read(frames.checked(replay["index"]))


def read_raster(index, node, step):
    m = frames.read(frames.checked(index["images"][f"{node}:{step}"]))
    if m["row"]["annotation_frame"] != step or m["row"]["node"] != node:
        raise ValueError("Wrong observed raster")
    with np.load(frames.checked(m["cache"]), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def record_features(record, index, cfg):
    f = base.fit_source
    node = record["node"]
    raw = frames.checked(record["annotation"]).read_bytes()
    rec = f.prep.mot_record(raw, record["clip"], range(1, record["annotation_frames"] + 1))
    rows = np.asarray([[float(v) for v in row] for row in csv.reader(io.StringIO(raw.decode("utf-8-sig")))
                       if row and any(v.strip() for v in row)])
    boxes = np.zeros((*rec.valid.shape, 4))
    boxes[np.searchsorted(rec.agent_ids, rows[:, 1].astype(int)), rows[:, 0].astype(int)] = rows[:, 2:6]
    wanted = sorted({t for q, _ in record["queries"] for t in range(q-6, q+1)})
    measures, tracks = {}, {}
    a, last = None, -2
    for t in wanted:
        if last != t-1:
            a = read_raster(index, node, t-1)
        b = read_raster(index, node, t)
        m = frames.method.estimate(a, b, cfg["measurement"])
        visible = np.flatnonzero(rec.valid[:, t-1])
        values = np.zeros((len(boxes), 5))
        values[visible] = foreground.flow(a, b, boxes[visible, t-1])
        measures[t], tracks[t] = m, values
        a, last = b, t
    geometry, histories, ids, queries = [], [], [], []
    row = 0
    for q, expected in record["queries"]:
        p = f.prep.past_inputs(rec, q, f.prep.SPEC)
        g, at, _ = f.prep.encode_scene(p["positions"], p["history_mask"], p["agent_ids"])
        if len(g) != expected:
            raise ValueError("Observation eligibility changed")
        geometry.append(g)
        histories.append(p["positions"][at])
        ids.append(np.searchsorted(rec.agent_ids, p["agent_ids"][at]))
        queries.append(np.arange(row, row + len(g)))
        row += len(g)
    g = np.concatenate(geometry)
    _, _, _, paths, delta, _, _, _ = prior.prior.prior.model.inputs(g, queries)
    bg, fg = [], []
    for (q, _), ix, h, agents in zip(record["queries"], queries, histories, ids, strict=True):
        t = list(range(q-6, q+1))
        obs = np.stack([tracks[s][agents] for s in t], axis=1)
        bg.append(frames.method.contextual_features(h, paths["ols4"][ix]+delta[ix], q,
                                                   {s: measures[s] for s in t}))
        fg.append(foreground.features(h, obs))
    return g, np.concatenate(bg), np.concatenate(fg), dict(node=node, rows=row, pairs=len(wanted),
        background_pairs_supported=sum(m["supported"] for m in measures.values()))


def features(cfg, image, index, beat):
    g, bg, fg, records = [], [], [], []
    for r in image["roster"]["records"]:
        a, b, c, stat = record_features(r, index, cfg)
        g.append(a)
        bg.append(b)
        fg.append(c)
        records.append(stat)
        beat(state="past_pixels_measured", **stat)
    g, bg, fg = map(np.concatenate, (g, bg, fg))
    identity = base.fit_source.source.array_hash
    population = frames.read(base.fit_source.PUBLIC / "fit_population.json")
    if identity(g) != population["geometry_sha256"] or not all(np.isfinite(v).all() for v in (bg, fg)):
        raise ValueError("Features do not align with the original trajectory roster")
    path = PRIVATE / "features.npz"
    if not path.exists():
        prior.TRAINER.write_npz(path, background=bg, foreground=fg)
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["background"], bg)
        np.testing.assert_array_equal(z["foreground"], fg)
    report = PUBLIC / "feature_report.json"
    run.once(report, run.stable(dict(bank=run.artifact(path), geometry_hash=identity(g), rows=len(g),
        recordings=records, background_cols=bg.shape[1], foreground_cols=fg.shape[1],
        foreground_any=float(np.mean(fg[:, 0] > 0.)), foreground_last=float(fg[:, 1].mean()),
        foreground_mean_coverage=float(fg[:, 0].mean()), future_targets_opened=False)))
    return report


def train(jobs, old, beat, resume):
    d, qs, pack = prior.prior.prior.data(jobs, beat)
    x, _, scale, paths, delta, legal, _, _ = pack
    quality, _ = prior.prior.quality_inputs(jobs[0]["identity"], d, qs, beat)
    digest = base.fit_source.source.array_hash
    if digest(quality) != old["quality_hash"]:
        raise ValueError("Past observation features changed")
    feature_report = frames.read(PUBLIC / "feature_report.json")
    if digest(d["geometry"]) != feature_report["geometry_hash"]:
        raise ValueError("Past image features do not align")
    with np.load(run.checked(feature_report["bank"]), allow_pickle=False) as z:
        bg, fg = z["background"], z["foreground"]
    bg, fg = bg.clip(-16., 16.), fg.clip(-16., 16.)
    x = np.column_stack((x, quality))
    inputs = tuple(torch.from_numpy(np.column_stack((x, b, f))) for b, f in
        ((np.zeros_like(bg), np.zeros_like(fg)), (bg, np.zeros_like(fg)), (bg, fg)))
    sites = sorted(set(d["sites"]))
    mu = np.full((len(prior.prev.SEEDS), len(ARMS), len(x), 12), np.nan, np.float32)
    fits = []
    for held in sites:
        ir = np.flatnonzero(d["sites"] != held)
        iv = np.flatnonzero(d["sites"] == held)
        local = np.full(len(x), -1, np.int64)
        local[ir] = np.arange(len(ir))
        qlocal = [local[q] for q in qs if d["sites"][q[0]] != held]
        if any((q < 0).any() for q in qlocal):
            raise ValueError("Held rows appeared in training")
        debit, frac, _ = prior.prior.model.components(paths["ols4"][ir], delta[ir],
            d["target"][ir], d["valid"][ir], scale[ir])
        debit, frac, observed = map(torch.from_numpy, (debit, frac, d["valid"][ir]))
        for k, seed in enumerate(prior.prev.SEEDS):
            pairs, src = prior.pairs(qlocal, legal[ir], d["sites"][ir], seed, per_query=16)
            groups = [np.flatnonzero(src == s) for s in sorted(set(src))]
            states = []
            for j, arm in enumerate(ARMS):
                ident = dict(seed=seed, arm=arm, held=str(held), registration=run.sha(REG),
                    train_pairs=digest(d["ids"][ir][pairs]))
                torch.manual_seed(seed)
                net = prior.model.network(inputs[j].shape[-1])

                def loss(net, at):
                    ids = pairs[at]
                    return prior.model.loss(net(inputs[j][ir[ids]]), frac[ids], debit[ids],
                                            observed[ids], "conditional_point")

                net, ck, rng = prior.TRAINER.train_network(net, loss, groups,
                    PRIVATE / f"{held}_{seed}_{arm}.pt", ident, STEPS, beat, resume)
                states.append(rng)
                with torch.inference_mode():
                    mu[k, j, iv] = net(inputs[j][iv]).numpy()
                fits.append(dict(identity=ident, checkpoint=ck, rows=len(iv)))
            for rng in states[1:]:
                base.fit_source.source.exact(states[0], rng)
    if len(fits) != 81 or not np.isfinite(mu).all():
        raise ValueError("Incomplete vision fits")
    bank = prior.TRAINER.write_npz(PRIVATE / "mu.npz", ids=d["ids"], mu=mu)
    out = PRIVATE / "freeze.json"
    run.once(out, run.stable(dict(fits=fits, bank=bank, support=old["scores"],
        registration=run.sha(REG), past_only=True, source_held=True)))
    return out


def score(jobs, cut, old, freeze, beat):
    if freeze["registration"] != run.sha(REG) or freeze["support"] != old["scores"]:
        raise ValueError("Past-image rankers must use the fixed original support")
    d, qs, pack = prior.prior.prior.data(jobs, beat)
    _, _, scale, paths, delta, legal, magnitude, _ = pack
    y, v, floor = d["target"], d["valid"], paths["ols4"]
    known, end = v.any(1), v[:, -1]
    dist = np.linalg.norm(delta, axis=-1)
    candidate = floor + delta
    cv = ade_from(paths["cv"], y, v)[0]
    easy, hard = known & (cv > 0.) & (cv <= cut["easy"]), known & (cv >= cut["hard"])
    _, frac, _ = prior.prior.model.components(floor, delta, y, v, scale)
    sites, src = np.unique(d["sites"], return_inverse=True)
    w = np.empty(len(y))
    for q in qs:
        w[q] = 1. / len(q)

    def mass(z):
        return np.bincount(src, weights=w*z, minlength=len(sites))

    denom = mass(cv)
    with np.load(run.checked(freeze["bank"]), allow_pickle=False) as z, np.load(
            run.checked(old["scores"]), allow_pickle=False) as r:
        np.testing.assert_array_equal(z["ids"], d["ids"])
        np.testing.assert_array_equal(r["ids"], d["ids"])
        mu, support = z["mu"], r["support"][:, prior.prior.model.ARMS.index("factor_quality")]
    scores = prior.ORIGINAL.scores
    result, contrasts, risks, modified = {}, {}, {}, {}
    for k, seed in enumerate(prior.prev.SEEDS):
        p = support[k]
        zero = -((1.-p)*dist).mean(1)
        layout = prior.model.support_layout(p.mean(1), zero, legal, qs)
        rank = dict(support_only=zero, minimum=-magnitude, **{arm: prior.model.combined(mu[k, j], p, dist)
            for j, arm in enumerate(ARMS)})
        chosen = {a: prior.model.matched_choices(r, layout) for a, r in rank.items()}
        for m in chosen.values():
            np.testing.assert_array_equal([m[q].sum() for q in qs], [2]*len(qs))
            np.testing.assert_array_equal([m[b].sum() for b, _ in layout], [n for _, n in layout])
        for a, m in chosen.items():
            modified.setdefault(a, []).append(sum(not np.array_equal(m[q], chosen["no_image"][q]) for q in qs))
        chosen["ols4"] = np.zeros(len(y), bool)
        pred = {a: np.where(m[:, None, None], candidate, floor) for a, m in chosen.items()}
        for a, path in pred.items():
            ade, fde = ade_from(path, y, v)
            g = scores(path, floor, y, v)
            data = {name: 100.*mass(z)/denom for name, z in g.items()}
            data.update(ade=mass(ade)/mass(known), fde=mass(fde*end)/mass(end),
                        hard=100.*mass(hard*g["full_grid_lower"])/denom)
            if a in ARMS:
                m = dist * v * legal[:, None]
                data["conditional_mae"] = mass((np.abs(mu[k, ARMS.index(a)]-frac)*m).sum(1))/mass(m.sum(1))
            result.setdefault(a, []).append(data)
            risks.setdefault(a, []).append([float(100.*a/b) if b > 0 else None for a, b in zip(
                mass(easy*np.maximum(ade-cv, 0.)), mass(easy*cv), strict=True)])
        for a, b in PAIRS:
            g = scores(pred[a], pred[b], y, v)
            contrasts.setdefault(f"{a}_vs_{b}", []).append({name: 100.*mass(z)/denom for name, z in g.items()})
        beat(state="scored_seed", seed=seed, switches=2*len(qs))
    draws = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))
    interval = lambda r: prior.ORIGINAL.parent.old.metric.intervals(np.stack(r, 1), draws)
    summarize = lambda d: {a: {m: interval([r[m] for r in rows]) for m in rows[0]}
                           for a, rows in d.items()}
    return dict(utility=summarize(result), direct=summarize(contrasts), easy_risk=risks,
        changed_queries=modified, fits=len(freeze["fits"]), queries=len(qs), rows=len(y),
        sources=len(sites), switches_per_seed=2*len(qs), original_rule_unchanged=True,
        independent_confirmation=False, deployed=False, stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("register", "features", "feature_replay", "train", "score", "replay"))
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    cut, jobs, old, cfg, image, index = register()
    if a.phase == "register":
        return
    for f in (*FILES, REG):
        if subprocess.check_output(["git", "show", f"HEAD:{f.relative_to(ROOT)}"], cwd=ROOT) != f.read_bytes():
            raise ValueError("Commit registered experiment before reading outcomes")
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        frames.cv2.setNumThreads(1)
        frames.cv2.ocl.setUseOpenCL(False)

        def beat(**values):
            status = dict(pid=os.getpid(), phase=a.phase, seconds=time.monotonic()-start, **values)
            base.fit_source.source.atomic(PRIVATE / "heartbeat.json", status)
            if values.get("step", STEPS) == STEPS:
                print(status, flush=True)
            if shutil.disk_usage(ROOT).free < 2*1024**3 or status["seconds"] > 43200:
                raise OSError("Resource guard; checkpoints are preserved")

        receipt = PUBLIC / f"{a.phase}_execution.json"
        if receipt.exists():
            prior.prev.receipt(PUBLIC, a.phase)
            return
        if a.phase in ("features", "feature_replay"):
            if a.phase == "feature_replay":
                prior.prev.receipt(PUBLIC, "features")
            out = features(cfg, image, index, beat)
        elif a.phase == "train":
            prior.prev.receipt(PUBLIC, "feature_replay")
            out = train(jobs, old, beat, a.resume)
        else:
            if a.phase == "replay":
                prior.prev.receipt(PUBLIC, "score")
            out = PUBLIC / "analysis.json"
            run.once(out, run.stable(score(jobs, cut, old, prior.prev.receipt(PUBLIC, "train"), beat)))
        run.once(receipt, dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out)))


if __name__ == "__main__":
    main()
