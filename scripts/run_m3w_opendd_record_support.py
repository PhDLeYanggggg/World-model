"""Cross-fit whole-record support supervision with an observed-prefix control."""

import argparse
from collections import Counter
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
from scripts import run_m3w_opendd_slate_queue as study
from scripts import run_m3w_opendd_nested_support_bank as bank
from src.world_model import m3w_opendd_record_support as model

admit = bank.admit
v, run, two, hard, prior, base, stop = admit.v, admit.run, admit.two, admit.hard, admit.prior, admit.base, admit.stop
OUT, PRIVATE = admit.OUT.parent/"opendd_record_support_v1", admit.PRIVATE.parent/"opendd_record_support_v1"
STEPS = 500
HEADS = ("local_same_data", "record_no_prefix", "record_prefix")
ARMS = ("gain", "easy", *HEADS, "record_guard")


def register():
    key, outer, old, source, gain, prev, producers, releases, capacity = admit.register()
    bank_key, _ = bank.register()
    file = bank.OUT/"producer_freeze.json"
    b = run.read(file)
    old_path = admit.OUT/"analysis.json"
    if (b["registration"] != bank_key or run.read(old_path)["registration"] != key
            or not b["original_2pct"] or b["confirmation_access"]):
        raise ValueError("Require matched parent result and triple-excluded bank")
    if len(b["heads"]) != 105 or len(b["predictions"]) != 315:
        raise ValueError("Incomplete triple-excluded training bank")
    for z in b["heads"].values():
        ident = z["identity"]
        if (len(ident["excluded"]) != 3 or len(ident["fit"]) != 4
                or set(ident["excluded"]) & set(ident["fit"]) or ident["registration"] != bank_key):
            raise ValueError("Invalid nested producer lineage")
    p = OUT/"registration.json"
    run.once(p, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(x) for x in (old_path, file)], seeds=two.study.prior.SEEDS, steps=STEPS,
        native_steps=[8, 12], threads=4, workers=0, first_slot_steps=prior.STEPS,
        donor_first_slot="same model and threshold; fit excludes donor and outer site, all fitting forecasts triple excluded",
        features="current observed menu plus optional first edit state; first edit hidden until its query",
        target="whole-record risk and positive conservative gain against CV and OLS4; local arm uses same rows and features",
        guard="same budget; replace easy only for better joint logit and non-worse predicted risk and conservative gain",
        matching="keep both edit counts, first edit, second query, and both amplitudes frozen",
        primary="record_prefix versus local_same_data at equal edit budgets; record_guard versus frozen easy",
        no_threshold_search=True, original_2pct=True, confirmation_access=False,
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    return run.sha(p), outer, old, source, gain, prev, producers, releases, capacity, b


def read_prediction(refs, excluded, site, seed):
    r = refs[tuple(sorted(excluded)), site, seed]
    if r["target"] != site or site not in r["excluded"]:
        raise ValueError("Forecast receipt not bound to the excluded donor")
    return np.load(run.checked(r["prediction"]), allow_pickle=False)


def contributions(cv, ols, p, y, valid, cut):
    mom = two.moments(cv, p, y, valid, cut)
    offset = v.paired(ols, cv, y, valid)[1].sum(1)
    g = np.column_stack([v.paired(cv, p, y, valid)[1].sum(1), v.paired(ols, p, y, valid)[1].sum(1)-offset])
    risk = np.column_stack([mom[k] for k in ("all_excess", "easy_excess", "easy_signed")])
    return g, risk, mom["ref_easy"], mom["base"], offset


@torch.inference_mode()
def menus(d, cv, path, release, gain_head):
    p, x, scale, easy, blocks = stop.inputs(d, cv, path, release)
    large = cv+base.model.GAIN_SCALE*(path-cv)
    f, size = prior.feature(d, cv, large)
    gain = gain_head(torch.from_numpy(f)).numpy()[:, 0]*size
    q = d["query"]
    records = []
    for r, seq in blocks.items():
        first, anchor, _ = base.model.choose((rows, gain[rows], easy[rows]) for rows in seq)
        if anchor is None:
            records.append((r, first, None, None, None))
            continue
        rows = next(rows for rows in seq if len(rows) and q[rows[0]] == q[anchor])
        rows, replay = admit.model.menu(rows, easy, first)
        if replay != anchor:
            raise ValueError("Original support time or anchor changed")
        local = admit.model.features(x, easy, rows, anchor)
        pre = model.prefix(rows, q, first, f, gain, easy, size)
        records.append((r, first, anchor, rows, np.c_[local, np.broadcast_to(pre, (len(rows), len(pre)))]))
    return records, p, large, scale, f.shape[1]+5


def training(d, cv, ols, path, release, first_head, cuts):
    roster, p, large, scale, width = menus(d, cv, path, release, first_head)
    y, valid, rec = d["y"], d["valid"], d["rec"]
    g, risk, easy, err, offset = contributions(cv, ols, p, y, valid, cuts[0])
    big = contributions(cv, ols, large, y, valid, cuts[0])
    local, _ = admit.targets(d, cv, p, scale, cuts[0])
    x, z, counts = [], [], Counter()
    for r, first, anchor, rows, f in roster:
        if anchor is None:
            continue
        ix = np.flatnonzero(rec == r)
        g0 = tuple(a[first] for a in big[:4]) if first is not None else None
        safe, joint = model.support_targets(g[rows], risk[rows], easy[rows], err[rows],
                                            easy[ix].sum(), g0, offset[ix].sum())
        if joint.any():
            witness = int(rows[np.flatnonzero(joint)[0]])
            output = cv[ix].copy()
            if first is not None:
                output[np.searchsorted(ix, first)] = large[first]
            output[np.searchsorted(ix, witness)] = p[witness]
            scored = v.parent.score(dict(y=y[ix], valid=valid[ix], rec=rec[ix]), cv[ix], ols[ix], output, cuts)
            if scored["dual_positive"] != 1:
                raise ValueError("Record target does not reproduce the original scorer")
        x.append(f)
        z.append(np.c_[local[rows, 0], local[rows, 1], safe, joint].astype(np.float32))
        counts.update(records=1, rows=len(rows), positive=int(joint.sum()), menus_with_positive=int(joint.any()),
                      prior_first=int(first is not None and d["query"][first] <= d["query"][anchor]))
    if not x:
        return None, counts
    return (torch.from_numpy(np.concatenate(x)), torch.from_numpy(np.concatenate(z)), width), counts


def view(x, width, arm):
    if arm == "record_no_prefix":
        x = x.clone()
        x[:, -width:] = 0.
    return x


def fit_set(fit, arm):
    return [(view(x, width, arm), y[:, [0, 1, 1] if arm == "local_same_data" else [0, 2, 3]])
            for x, y, width in fit]


@torch.inference_mode()
def evaluate(d, baseline, path, release, first_head, nets, cut, ref, ceiling):
    cv, ols = baseline[:2]
    records, p, large, _, width = menus(d, cv, path, release, first_head)
    y, valid, rec, q = (d[k] for k in ("y", "valid", "rec", "query"))
    pred, choices = {k: cv.copy() for k in ARMS}, []
    for r, first, anchor, rows, f in records:
        if first is not None:
            for a in pred.values():
                a[first] = large[first]
        if anchor is None:
            continue
        score = {k: net(view(torch.from_numpy(f), width, k)).numpy() for k, net in nets.items()}
        picks = {k: model.choose(rows, anchor, score[k]) for k in HEADS}
        picks["record_guard"] = model.choose(rows, anchor, score["record_prefix"], guard=True)
        for k, j in dict(easy=anchor, **picks).items():
            if j == first or rec[j] != r or q[j] != q[anchor]:
                raise ValueError("Changing a recipient cannot alter the frozen schedule")
            pred[k][j] = p[j]
        choices.append((anchor, picks))
    metrics = {k: v.parent.score(d, cv, ols, a, cut) for k, a in pred.items()}
    for k, old in (("gain", "gain"), ("easy", "two_roles")):
        if any(metrics[k][f] != ref["metrics"][old][f] for f in v.prior.FIELDS):
            raise ValueError("Frozen edit policy did not replay")
    for k in ARMS[1:]:
        if (metrics[k]["edits"] != metrics["gain"]["edits"]+len(choices)
                or metrics[k]["dual_positive"] > ceiling["fixed"]["all_legal"]):
            raise ValueError("A policy did not match budget or menu")
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    pairs = [(k, "cv") for k in ARMS]+[(k, "easy") for k in ARMS[2:]]
    pairs += [("record_prefix", "local_same_data"), ("record_prefix", "record_no_prefix")]
    effect = {f"{a}_vs_{b}": hard.paired_pp(cv if b == "cv" else pred[b], pred[a], y, valid, mass)
              for a, b in pairs}
    changed = {k: sum(picks[k] != anchor for anchor, picks in choices) for k in ARMS[2:]}
    return dict(metrics=metrics, effects=effect, changed=changed)


def execute(key, outer, old, source, first_heads, prev, producers, releases, capacity, triple, replay, beat):
    sites = {s: study.c.forecast.load(s, source)[0] for s in v.parent.source.SITES}
    banks = {s: v.parent.prior.analytic(d) for s, d in sites.items()}
    index = lambda r: {(tuple(z["excluded"]), z["target"], z["seed"]): z for z in r}
    double, nested = index(producers), index(triple["predictions"])
    cuts = lambda use: np.quantile(np.concatenate([sites[s]["cut_samples"] for s in use]), [.25, .75])
    hard_width = hard.model.features(next(iter(sites.values()))["h"][:1]).shape[1]
    result, freeze, first_freeze, updates = {}, {}, {}, 0
    for held, d in sites.items():
        use = [s for s in sites if s != held]
        result[held] = []
        for seed in two.study.prior.SEEDS:
            fit, donors = [], Counter()
            for donor in use:
                others = [s for s in use if s != donor]
                z = [prior.training(sites[s], banks[s][0],
                        read_prediction(nested, (held, donor, s), s, seed),
                        cuts([a for a in others if a != s])[0]) for s in others]
                z = [x for x in z if x is not None]
                name = f"first_{held}_{donor}_{seed}"
                n = z[0][0].shape[1]
                identity = dict(registration=key, excluded=[held, donor], fit=others, seed=seed)
                first, first_freeze[name], u = two.fit.fit(identity, name, seed, lambda: prior.model.network(n),
                    prior.STEPS, two.batcher(z), prior.model.loss, replay, beat)
                updates += u
                pair = tuple(sorted((held, donor)))
                receipt = releases["release_"+"_".join(pair)+f"_{seed}"]
                if set(receipt["identity"]["fit"]) & set(pair):
                    raise ValueError("Donor release head used excluded data")
                release = two.fit.load(receipt, lambda: hard.model.network(hard_width), hard.STEPS)
                path = read_prediction(double, pair, donor, seed)
                a, b = training(sites[donor], *banks[donor][:2], path, release, first, cuts(others))
                donors.update(b)
                if a is not None:
                    fit.append(a)
            name = f"{held}_{seed}"
            nets = {}
            n = fit[0][0].shape[1]
            for k in HEADS:
                ident = dict(registration=key, held=held, fit=use, seed=seed, arm=k)
                nets[k], freeze[f"{name}_{k}"], u = two.fit.fit(ident, f"{name}_{k}", seed, lambda: model.network(n),
                    STEPS, two.batcher(fit_set(fit, k)), model.loss, replay, beat)
                updates += u
            release = two.fit.load(outer[name], lambda: hard.model.network(hard_width), hard.STEPS)
            lookup = lambda a: next(z for z in a["by_site"][held] if z["seed"] == seed)
            with np.load(run.checked(lookup(old)["prediction"]), allow_pickle=False) as z:
                path = z["comoving_ego"]
            cv = banks[held][0]
            n = prior.feature({k: d[k][:1] for k in ("h", "query")}, cv[:1], path[:1])[0].shape[1]
            first = two.fit.load(first_heads[name], lambda: prior.model.network(n), prior.STEPS)
            out = evaluate(d, banks[held], path, release, first, nets, cuts(use), lookup(prev), lookup(capacity)["counts"])
            result[held].append(dict(seed=seed, donors=donors, **out))
            beat(phase="site", site=held, seed=seed, updates=updates, records=donors["records"], positive=donors["positive"])
    flat = [r for rows in result.values() for r in rows]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(k):
        x = np.array([np.mean([r["effects"][k] for r in rows]) for rows in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()))

    summary = dict(metrics={k: {f: sum(r["metrics"][k][f] for r in flat) for f in v.prior.FIELDS} for k in ARMS},
        effects={k: ci(k) for k in flat[0]["effects"]},
        changed=sum((Counter(r["changed"]) for r in flat), Counter()))
    f = OUT/"head_freeze.json"
    run.once(f, run.stable(dict(registration=key, heads=freeze, first_heads=first_freeze)))
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
