"""Held-source fit and replay of box-quality and step-support signals."""

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
    raise SystemExit("Use the native arm64 .venv-pytorch interpreter.")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_recipient_top2 as prior
from src.world_model import m3w_observability as model
from sklearn.metrics import average_precision_score, roc_auc_score

run, base, prev, np, torch = prior.run, prior.base, prior.prev, prior.np, prior.torch
PUBLIC = prior.PUBLIC.parent / "native_observability_v1"
PRIVATE = prior.PRIVATE.parent / PUBLIC.name
REG = PUBLIC / "registration.json"
MANIFEST = prior.PUBLIC.parent / "movingdronecrowd_background_v2/image_manifest.json"
FILES = (Path(__file__), Path(model.__file__), ROOT / "tests/test_m3w_observability.py",
         PUBLIC / "protocol.md")
STEPS = 500
CONTRASTS = (("factor_quality", "factor_motion"), ("direct_quality", "direct_motion"),
             ("factor_quality", "ols4"), ("factor_quality", "minimum"),
             ("factor_quality", "direct_quality"), ("direct_quality", "ols4"))


def register():
    cut, jobs = prior.register()
    prev.receipt(prior.PUBLIC, "replay")
    bound = (*FILES, prior.REG, prior.PUBLIC / "replay_execution.json", MANIFEST,
             Path(prior.prior.trainer.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in bound], steps=STEPS,
        settings=prior.prior.trainer.CFG, seeds=prev.SEEDS, arms=model.ARMS, features=model.NAMES,
        comparisons=CONTRASTS, same_candidates=True, image_dimensions="verified_frame_zero_only",
        original_rule_unchanged=True, independent_confirmation=False)))
    return cut, jobs


def feature_at(box, mask, frame, dimensions):
    past = slice(frame - 7, frame + 1)
    target = np.flatnonzero(mask[:, past].all(1))
    return target, model.box_features(box[:, past], mask[:, past], target, dimensions)


def quality_inputs(identity, d, qs, beat):
    f = base.fit_source
    adm = f.prep.load_admission(ROOT, ROOT / f.read(f.source.CONFIG)["admission"])
    lookup = {c["clip"]: c for c in f.read(f.bank.old.HOME / "movingdronecrowd_intake_v1/analysis.json")["recordings"]}
    meta = {r["node"]: r for r in f.read(MANIFEST)["views"] if r["annotation_frame_zero_based"] == 0}
    blocks, ids, records = [], [], []
    offset, cursor, query_count = 0, 0, 0
    for node, entry in sorted(adm["recordings"].items()):
        clip = lookup[entry["clip"]]
        n = clip["window_support"]["obs8_pred12_native_stride1"]["history_complete_agent_instances"]
        start, offset = offset, offset + n
        if not n or entry["exclusion_group"] not in identity["normalizers"]:
            continue
        p = ROOT / entry["source_annotation_path"]
        f.prep.require_fit_record(adm, node, "fit", p)
        raw = p.read_bytes()
        rec = f.prep.mot_record(raw, entry["clip"], range(1, clip["image_frames"] + 1))
        size = meta[node]
        if size["annotation_sha256"] != rec.source_xml_sha256 or size["clip"] != entry["clip"]:
            raise ValueError("First-frame dimensions are not bound to this annotation")
        rows = np.asarray([[float(v) for v in r] for r in csv.reader(io.StringIO(raw.decode("utf-8-sig")))
                           if r and any(v.strip() for v in r)], dtype=np.float64)
        box = np.zeros((*rec.valid.shape, 4))
        a = np.searchsorted(rec.agent_ids, rows[:, 1].astype(np.int64))
        box[a, rows[:, 0].astype(np.int64)] = rows[:, 2:6]
        stop = cursor + n
        checks = 0
        frames = range(7, rec.valid.shape[1] - 12)
        for frame in frames:
            if not rec.valid[:, frame-7:frame+1].all(1).any():
                continue
            target, x = feature_at(box, rec.valid, frame, size["source_dimensions"])
            q = qs[query_count]
            np.testing.assert_array_equal(q, np.arange(cursor, cursor + len(target)))
            window = rec.positions[target, frame-7:frame+1]
            np.testing.assert_array_equal((window-window[:, -1:]).reshape(-1, 16).astype(np.float32),
                                          d["geometry"][q, :16])
            if frame in (frames.start, (frames.start + frames.stop)//2, frames.stop - 1):
                t, prefix = feature_at(box[:, :frame+1], rec.valid[:, :frame+1], frame, size["source_dimensions"])
                np.testing.assert_array_equal(target, t)
                np.testing.assert_array_equal(x, prefix)
                checks += 1
            blocks.append(x)
            query_count += 1
            cursor += len(target)
        if cursor != stop or not checks or not (d["sites"][stop-n:stop] == entry["exclusion_group"]).all():
            raise ValueError("Box features did not match admitted fit roster")
        ids.extend(range(start, start+n))
        records.append(dict(node=node, rows=n, prefix_checks=checks, dimensions=size["source_dimensions"]))
        beat(state="quality_source_checked", recording=node, rows=n)
    np.testing.assert_array_equal(ids, d["ids"])
    if query_count != len(qs):
        raise ValueError("Box features did not cover all queries")
    return np.concatenate(blocks), records


def train(jobs, beat, resume):
    d, qs, packed = prior.data(jobs, beat)
    motion, _, scale, paths, delta, eligible, _, _ = packed
    quality, records = quality_inputs(jobs[0]["identity"], d, qs, beat)
    x = [torch.from_numpy(np.column_stack((motion, q))) for q in (np.zeros_like(quality), quality)]
    d_eval = torch.from_numpy((np.linalg.norm(delta, axis=-1) / scale[:, None]).astype(np.float32))
    sites = sorted(set(d["sites"]))
    output = np.full((len(prev.SEEDS), len(model.ARMS), len(motion)), np.nan, np.float32)
    support = np.full((*output.shape, 12), np.nan, np.float32)
    base_probability = np.full((len(motion), 12), np.nan)
    fits = []
    for held in sites:
        ir = np.flatnonzero((d["sites"] != held) & eligible)
        iv = np.flatnonzero(d["sites"] == held)
        base_probability[iv] = d["valid"][ir].mean(0)
        groups = [np.flatnonzero(d["sites"][ir] == site) for site in sites if site != held]
        labels = tuple(map(torch.from_numpy, model.components(paths["ols4"][ir], delta[ir],
            d["target"][ir], d["valid"][ir], scale[ir])))
        observed = torch.from_numpy(d["valid"][ir])
        for k, seed in enumerate(prev.SEEDS):
            states = []
            for j, arm in enumerate(model.ARMS):
                inp = x[j % 2]
                ident = dict(seed=seed, held=str(held), arm=arm, registration=run.sha(REG),
                    train_ids=base.fit_source.source.array_hash(d["ids"][ir]))
                torch.manual_seed(seed)
                net = model.network(inp.shape[-1])

                def loss(net, at):
                    debit, frac, direct = (v[at] for v in labels)
                    return model.loss(net(inp[ir[at]]), debit, frac, observed[at], direct, arm)

                net, ck, rng = prior.prior.trainer.train_network(net, loss, groups,
                    PRIVATE / f"{seed}_{held}_{arm}.pt", ident, STEPS, beat, resume)
                with torch.inference_mode():
                    u, p = model.estimate(net(inp[iv]), d_eval[iv], arm)
                output[k, j, iv] = u.numpy() * scale[iv]
                support[k, j, iv] = p.numpy()
                states.append(rng)
                fits.append(dict(identity=ident, checkpoint=ck, held_rows=len(iv)))
            for rng in states[1:]:
                base.fit_source.source.exact(states[0], rng)
    if len(fits) != 108 or not all(np.isfinite(a).all() for a in (output, support, base_probability)):
        raise ValueError("Incomplete held-source predictions")
    artifact = prior.prior.trainer.write_npz(PRIVATE / "scores.npz", ids=d["ids"], scores=output,
        support=support, prevalence=base_probability)
    receipt = PRIVATE / "freeze.json"
    run.once(receipt, run.stable(dict(registration=run.sha(REG), fits=fits, scores=artifact,
        past_only=True, held_sources=True, matched_draws=True, quality_sources=records,
        quality_hash=base.fit_source.source.array_hash(quality),
        quality_std=quality.std(0).tolist())))
    return receipt


def score(jobs, cut, freeze, beat):
    if freeze["registration"] != run.sha(REG) or not all(
            freeze[k] for k in ("past_only", "held_sources", "matched_draws")):
        raise ValueError("Invalid frozen score bank")
    d, qs, packed = prior.data(jobs, beat)
    _, _, _, paths, delta, eligible, magnitude, _, = packed
    y, valid, floor = d["target"], d["valid"], paths["ols4"]
    known, endpoint = valid.any(1), valid[:, -1]
    cv = prior.prior.ade(paths["cv"], y, valid)[0]
    ols = prior.prior.ade(floor, y, valid)[0]
    easy = known & (cv > 0.) & (cv <= cut["easy"])
    hard = known & (cv >= cut["hard"])
    sites, site = np.unique(d["sites"], return_inverse=True)
    w = np.empty(len(y))
    for q in qs:
        w[q] = 1. / len(q)

    def mass(v):
        return np.bincount(site, weights=w*v, minlength=len(sites))

    def support_scores(p):
        s = []
        for i in range(len(sites)):
            at = eligible & (site == i)
            missing, pred = ~valid[at].flatten(), 1. - p[at].flatten()
            sw = np.repeat(w[at], valid.shape[1])
            s.append(dict(brier=float(np.average((missing-pred)**2, weights=sw)),
                          missing=float(np.average(missing, weights=sw)),
                          auroc=float(roc_auc_score(missing, pred, sample_weight=sw)) if np.unique(missing).size > 1 else None,
                          auprc=float(average_precision_score(missing, pred, sample_weight=sw))
                              if missing.any() else None))
        return s

    denom = mass(cv)
    if (denom <= 0).any():
        raise ValueError("Missing CV error denominator")
    candidate = floor + delta
    oracle_gain = prior.prior.prior.scores(candidate, floor, y, valid)["full_grid_lower"]
    count = [min(2, int(eligible[q].sum())) for q in qs]
    with np.load(run.checked(freeze["scores"]), allow_pickle=False) as z:
        np.testing.assert_array_equal(d["ids"], z["ids"])
        scores, p, prevalence = (z[k] for k in ("scores", "support", "prevalence"))
    utility, direct, risk = {}, {}, {}
    calibration = dict(prevalence=[support_scores(prevalence)] * len(prev.SEEDS))
    for k, seed in enumerate(prev.SEEDS):
        ranks = dict(zip(model.ARMS, scores[k], strict=True))
        ranks.update(oracle=oracle_gain, minimum=-magnitude)
        chosen = {a: prior.model.choose(v, eligible, qs) for a, v in ranks.items()}
        for a in chosen.values():
            np.testing.assert_array_equal([a[q].sum() for q in qs], count)
        chosen["ols4"] = np.zeros(len(y), bool)
        paths_out = {a: np.where(v[:, None, None], candidate, floor) for a, v in chosen.items()}
        for a, pred in paths_out.items():
            g = prior.prior.prior.scores(pred, floor, y, valid)
            ade, fde = prior.prior.ade(pred, y, valid)
            vals = {k: 100.*mass(z)/denom for k, z in g.items()}
            vals.update(ade=mass(ade)/mass(known), fde=mass(fde*endpoint)/mass(endpoint),
                        hard_contribution=100.*mass(hard*g["full_grid_lower"])/denom)
            if a != "ols4":
                selected_mass = mass(chosen[a])
                if (selected_mass <= 0).any():
                    raise ValueError("Missing active comparison support")
                vals["unknown_step_fraction"] = mass(chosen[a]*(~valid).mean(1)) / selected_mass
            utility.setdefault(a, []).append(vals)
            risk.setdefault(a, []).append({name: [float(100.*v/s) if s > 0 else None
                for v, s in zip(mass(easy*np.maximum(ade-ref, 0.)), mass(easy*ref), strict=True)]
                for name, ref in (("cv", cv), ("ols4", ols))})
        for a, b in CONTRASTS:
            g = prior.prior.prior.scores(paths_out[a], paths_out[b], y, valid)
            direct.setdefault(f"{a}_vs_{b}", []).append({m: 100.*mass(v)/denom for m, v in g.items()})
        for j, arm in enumerate(model.ARMS):
            calibration.setdefault(arm, []).append(support_scores(p[k, j]))
        beat(state="scored_seed", seed=seed, switches=sum(count))
    draws = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))
    interval = lambda v: prior.prior.prior.parent.old.metric.intervals(np.stack(v, 1), draws)

    def summary(d):
        return {a: {k: interval([r[k] for r in rows]) for k in rows[0]} for a, rows in d.items()}

    brier = {a: interval([np.array([v["brier"] for v in r]) for r in rows])
             for a, rows in calibration.items()}
    lift = {a: interval([np.array([b["brier"]-v["brier"] for b, v in zip(ref, r, strict=True)])
        for ref, r in zip(calibration["prevalence"], rows, strict=True)])
        for a, rows in calibration.items() if a != "prevalence"}
    regret = {a: interval([o["full_grid_lower"]-r["full_grid_lower"]
        for o, r in zip(utility["oracle"], vals, strict=True)]) for a, vals in utility.items()
        if a not in ("ols4", "oracle")}
    return dict(utility=summary(utility), direct=summary(direct), easy_by_source_seed=risk,
        support_by_source_seed=calibration, brier=brier, brier_improvement_vs_prevalence=lift,
        same_nomination_regret=regret, rows=len(y), queries=len(qs), switches_per_seed=sum(count),
        fits=len(freeze["fits"]), quality_sources=freeze["quality_sources"], quality_sha256=freeze["quality_hash"],
        quality_std=freeze["quality_std"], seeds=prev.SEEDS, sites=sites.tolist(),
        independent_confirmation=False, original_rule_unchanged=True, stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("register", "train", "score", "replay"))
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    cut, jobs = register()
    if a.phase == "register":
        return
    for f in (*FILES, REG):
        if subprocess.check_output(["git", "show", f"HEAD:{f.relative_to(ROOT)}"], cwd=ROOT) != f.read_bytes():
            raise ValueError("Commit registered code before running")
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)

        def beat(**data):
            msg = dict(pid=os.getpid(), phase=a.phase, seconds=time.monotonic()-start, **data)
            base.fit_source.source.atomic(PRIVATE / "heartbeat.json", msg)
            if data.get("step", STEPS) == STEPS:
                print(msg, flush=True)
            if shutil.disk_usage(ROOT).free < 2*1024**3 or msg["seconds"] > 43200:
                raise OSError("Resource limit; checkpoints preserved")

        mark = PUBLIC / f"{a.phase}_execution.json"
        if mark.exists():
            prev.receipt(PUBLIC, a.phase)
            return
        if a.phase == "train":
            out = train(jobs, beat, a.resume)
        else:
            if a.phase == "replay":
                prev.receipt(PUBLIC, "score")
            out = PUBLIC / "analysis.json"
            run.once(out, run.stable(score(jobs, cut, prev.receipt(PUBLIC, "train"), beat)))
        run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out)))


if __name__ == "__main__":
    main()
