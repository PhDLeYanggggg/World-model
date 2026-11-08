"""Train early reference admission at an unchanged gain clock."""

import argparse
from collections import Counter
from functools import partial
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_gain_wait as wait
from src.evaluation.m3w_selection_forecast_readout import policy
from src.world_model import m3w_opendd_early_reference as model
from src.world_model.m3w_opendd_record_support import support_targets

run, record, v, two, hard, prior, timing = wait.run, wait.record, wait.v, wait.two, wait.hard, wait.prior, wait.timing
OUT = wait.OUT.parent/"opendd_early_reference_v1"
PRIVATE = wait.PRIVATE.parent/OUT.name
ARMS = ("original", "incidence", "mass", "easy_matched", "early_oracle")
STEPS = 500


def register():
    key, *inputs = wait.register()
    a = timing.legacy.read(wait.OUT.name)
    timing.read_gain_wait_menu()
    if a["registration"] != key:
        raise ValueError("Require frozen gain-wait evidence")
    parents = [wait.OUT/f"{k}.json" for k in ("analysis", "head_freeze", "menu_registration", "menu_capacity")]
    f = OUT/"registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        trainer=run.artifact(Path(two.fit.__file__)), parents=[run.artifact(p) for p in parents],
        seeds=two.study.prior.SEEDS, steps=STEPS, threads=4, workers=0, native_steps=[8, 12],
        action="retain original gain exactly; original easy trigger has priority; one supplemental support before or at gain",
        target="local feasible easy-reference incidence and conditional supported reference mass",
        comparator="retrained incidence only; shadow easy control at identical support query, exact radius and count",
        lineage="outer and donor excluded upstream; cut uses neither site",
        oracle="retain original or move/add one support no later than original gain; original risk; original defined references preserved",
        primary="mass vs frozen original on all views, with same-clock exact-radius easy control",
        original_2pct=True, confirmation_access=False, no_threshold_search=True,
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    return run.sha(f), *inputs, a


@torch.inference_mode()
def prepare(d, cv, path, release, gain_head):
    (large, small), feat, blocks, gain, easy = wait.clock.prepare(d, cv, path, release, gain_head)
    radius = wait.shape.peak(small-cv)
    size = np.expm1(feat[1][:, -1]).clip(1e-8)
    x = np.c_[feat[1], np.log1p(radius/size)].astype(np.float32).clip(-16., 16.)
    return large, small, blocks, gain, easy, radius, x, size


def train_data(d, cv, cut, pack):
    _, small, _, _, _, radius, x, size = pack
    mom = two.moments(cv, small, d["y"], d["valid"], cut)
    # Conservative local supervision; the evaluation still uses the recording rule.
    safe = (mom["easy"] & (mom["all_excess"] <= 0.) & (mom["easy_excess"] <= 0.)
            & (mom["easy_signed"] <= .02*mom["ref_easy"]))
    legal = radius > 0.
    y = np.c_[safe, np.log1p(mom["ref_easy"]/size)].astype(np.float32)
    return (torch.from_numpy(x[legal]), torch.from_numpy(y[legal])), Counter(rows=int(legal.sum()), positive=int((safe & legal).sum()))


@torch.inference_mode()
def evaluate(d, cv, ols, cuts, pack, nets, ref):
    large, small, blocks, gain, easy, radius, x, _ = pack
    pred = {k: cv.copy() for k in ARMS}
    z = {k: n(torch.from_numpy(x)).numpy() for k, n in nets.items()}
    y, valid, rec, q = (d[k] for k in ("y", "valid", "rec", "query"))
    count = Counter()
    roster = {}
    threshold = wait.clock.base.model.EASY_THRESHOLD
    small_terms = record.contributions(cv, ols, small, y, valid, cuts[0])
    big_terms = record.contributions(cv, ols, large, y, valid, cuts[0])
    for r, seq in blocks.items():
        g, b, _ = wait.clock.base.model.choose((ix, gain[ix], easy[ix]) for ix in seq)
        roster[r] = g, b
        if g is not None:
            for p in pred.values():
                p[g] = large[g]
        if b is not None:
            pred["original"][b] = pred["early_oracle"][b] = small[b]
        for k, scores in z.items():
            a, j, c = model.choose(((ix, gain[ix], easy[ix], radius[ix], *scores[ix].T) for ix in seq),
                                    threshold, k)
            if a != g or (j is None and b is not None):
                raise ValueError("Early support changed gain or suppressed all support")
            if j is None:
                continue
            if j == a or (j != b and ((b is not None and q[j] >= q[b]) or (g is not None and q[j] > q[g]))):
                raise ValueError("Supplemental support violates chronological release")
            pred[k][j] = small[j]
            count.update({k+"_support": 1, k+"_advance": int(b is not None and j != b),
                          k+"_new": int(b is None), k+"_before_or_at_gain": int(g is not None and q[j] <= q[g])})
            if k == "mass":
                if c == a or q[j] != q[c] or radius[c] < radius[j]:
                    raise ValueError("Easy control does not match the intervention")
                pred["easy_matched"][c] = cv[c]+(small[c]-cv[c])*(radius[j]/radius[c])
                np.testing.assert_allclose(wait.shape.peak((pred["easy_matched"][c]-cv[c])[None])[0], radius[j], atol=1e-12, rtol=1e-10)
                count["easy_matched_recipient_changes"] += int(j != c)
        at = np.flatnonzero(rec == r)
        a = tuple(t[g] for t in big_terms[:4]) if g is not None else None
        old_defined = (g is not None and big_terms[2][g] > 0.) or (b is not None and small_terms[2][b] > 0.)
        allowed = (radius[at] > 0.) & (at != g)
        if g is not None:
            allowed &= q[at] <= q[g]
        if b is not None:
            allowed &= q[at] < q[b]
        bound = support_targets(*(t[at] for t in small_terms[:4]), small_terms[2][at].sum(),
                                a, small_terms[4][at].sum())[1]
        bound &= allowed
        if old_defined and (g is None or big_terms[2][g] <= 0.):
            bound &= small_terms[2][at] > 0.
        if bound.any():
            d0 = dict(y=y[at], valid=valid[at], rec=rec[at])
            original = v.parent.score(d0, cv[at], ols[at], pred["original"][at], cuts)
            if not original["dual_positive"]:
                if b is not None:
                    pred["early_oracle"][b] = cv[b]
                j = at[np.flatnonzero(bound)[0]]
                pred["early_oracle"][j] = small[j]
                exact = v.parent.score(d0, cv[at], ols[at], pred["early_oracle"][at], cuts)
                if exact["dual_positive"] != 1:
                    raise ValueError("Early-support witness failed original scoring")
                count["oracle_recovered_missing" if b is None else "oracle_advanced"] += 1
    stats = {k: v.parent.score(d, cv, ols, p, cuts) for k, p in pred.items()}
    if any(stats["original"][f] != ref["metrics"]["original"][f] for f in v.prior.FIELDS):
        raise ValueError("Frozen original no longer replays")
    if (stats["mass"]["edits"] != stats["easy_matched"]["edits"]
            or stats["early_oracle"]["dual_positive"] != stats["original"]["dual_positive"]
            + count["oracle_advanced"]+count["oracle_recovered_missing"]):
        raise ValueError("Role matching or oracle reconciliation failed")
    risk = {k: policy(p, cv, cv, y, valid, rec, easy_cut=cuts[0], hard_cut=cuts[1])["recordings"]
            for k, p in pred.items()}
    for k in ARMS[1:]:
        for r, (g, b) in roster.items():
            old = risk["original"][r]["completion_easy_risk_upper"] is not None
            new = risk[k][r]["completion_easy_risk_upper"] is not None
            count[k+"_lost_reference"] += int(old and not new)
            count[k+"_new_reference"] += int(not old and new)
    denom = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    pairs = [(k, "cv") for k in ARMS]+[(k, "original") for k in ARMS[1:]]
    pairs += [("mass", "incidence"), ("mass", "easy_matched")]
    effects = {f"{k}_vs_{b}": hard.paired_pp(cv if b == "cv" else pred[b], pred[k], y, valid, denom)
               for k, b in pairs}
    return dict(metrics=stats, effects=effects, diagnostics=count)


def execute(key, outer, old, source, gains, _rank, producers, releases, nested, _protected, ref, replay, beat):
    sites = {s: wait.study.c.forecast.load(s, source)[0] for s in v.parent.source.SITES}
    banks = {s: v.parent.prior.analytic(d) for s, d in sites.items()}
    refs = {(tuple(r["excluded"]), r["target"], r["seed"]): r for r in producers}
    cuts = lambda use: np.quantile(np.concatenate([sites[s]["cut_samples"] for s in use]), [.25, .75])
    hard_width = hard.model.features(next(iter(sites.values()))["h"][:1]).shape[1]
    result, freeze, updates = {}, {}, 0

    def prepare_site(s, path, r, f):
        d, cv = sites[s], banks[s][0]
        release = two.fit.load(r, lambda: hard.model.network(hard_width), hard.STEPS)
        n = prior.feature({k: d[k][:1] for k in ("h", "query")}, cv[:1], path[:1])[0].shape[1]
        first = two.fit.load(f, lambda: prior.model.network(n), prior.STEPS)
        return prepare(d, cv, path, release, first)

    for held, d in sites.items():
        use = [s for s in sites if s != held]
        result[held] = []
        for seed in two.study.prior.SEEDS:
            fit, donor = [], Counter()
            for s in use:
                pair = tuple(sorted((held, s)))
                r = releases["release_"+"_".join(pair)+f"_{seed}"]
                f = nested[f"first_{held}_{s}_{seed}"]
                if (set(r["identity"]["fit"]) & set(pair) or set(f["identity"]["excluded"]) != set(pair)
                        or set(f["identity"]["fit"]) & set(pair)):
                    raise ValueError("Nomination models expose held or donor labels")
                path = record.read_prediction(refs, pair, s, seed)
                p = prepare_site(s, path, r, f)
                items, info = train_data(sites[s], banks[s][0], cuts([t for t in use if t != s])[0], p)
                fit.append(items)
                donor.update(info)
            name = f"{held}_{seed}"
            n = fit[0][0].shape[1]
            nets = {}
            for k in ("mass", "incidence"):
                ident = dict(registration=key, held=held, fit=use, seed=seed, arm=k)
                tag = f"{name}_{k}"
                nets[k], freeze[tag], u = two.fit.fit(ident, tag, seed, lambda: model.network(n), STEPS,
                    two.batcher(fit), partial(model.loss, with_mass=k == "mass"), replay, beat)
                updates += u
            lookup = lambda a: next(z for z in a["by_site"][held] if z["seed"] == seed)
            with np.load(run.checked(lookup(old)["prediction"]), allow_pickle=False) as z:
                p = prepare_site(held, z["comoving_ego"], outer[name], gains[name])
            z = evaluate(d, *banks[held][:2], cuts(use), p, nets, lookup(ref))
            result[held].append(dict(seed=seed, donor=donor, **z))
            beat(phase="site", site=held, seed=seed, updates=updates, rows=donor["rows"], positives=donor["positive"])
    flat = [r for rows in result.values() for r in rows]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(k):
        x = np.array([np.mean([r["effects"][k] for r in rows]) for rows in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()))

    summary = dict(metrics={k: {f: sum(r["metrics"][k][f] for r in flat) for f in v.prior.FIELDS} for k in ARMS},
        effects={k: ci(k) for k in flat[0]["effects"]}, diagnostics=sum((r["diagnostics"] for r in flat), Counter()))
    f = OUT/"head_freeze.json"
    run.once(f, run.stable(dict(registration=key, heads=freeze)))
    run.once(OUT/"analysis.json", run.stable(dict(registration=key, heads=run.sha(f), by_site=result,
        summary=summary, original_2pct=True, confirmation_access=False, submission_ready=False)))
    return updates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**p):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **p)
            run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", z)
            if p.get("phase") != "fit":
                print(z, flush=True)

        two.fit.PRIVATE = PRIVATE
        u = execute(*register(), a.replay, beat)
        beat(complete=True, updates=u, replay=a.replay)


if __name__ == "__main__":
    main()
