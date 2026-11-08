"""Keep the causal release clock fixed and learn which matched slate to use."""

import argparse
from collections import Counter
import fcntl
from itertools import product
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
from scripts import run_m3w_opendd_dual_clock as clock
from src.world_model import m3w_opendd_budget_rank as model

record, base, run, v, two, hard, prior = clock.record, clock.base, clock.run, clock.v, clock.two, clock.hard, clock.prior
shape = clock.direction.shape.shape
OUT = clock.OUT.parent/"opendd_budget_rank_v1"
PRIVATE = clock.PRIVATE.parent/OUT.name
STEPS = 500
HEADS = ("point", "rank", "rank_no_geometry")
ARMS = ("original", *HEADS, "guard")


def register():
    key, outer, old, source, gains, ref, producers, releases, nested = clock.register()
    parents = [clock.OUT/f"{k}.json" for k in ("analysis", "head_freeze", "deadline_analysis", "serialization_recovery_v2")]
    a, h, _, r = map(run.read, parents)
    if (a["registration"] != key or h["registration"] != key or a["heads"] != run.sha(parents[1])
            or r["heads"]["sha256"] != a["heads"] or a["confirmation_access"] or not a["original_2pct"]):
        raise ValueError("Require the verified stopping study")
    f = OUT/"registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        trainer=run.artifact(Path(two.fit.__file__)), parents=[run.artifact(p) for p in parents],
        steps=STEPS, seeds=two.study.prior.SEEDS, threads=4, workers=0, native_steps=[8, 12],
        clock="frozen original online query role releases, exact count and radius per role",
        menu="anchor plus up to four current rows per role, neural/OLS4/past-prefix directions without amplification",
        training="double-excluded forecasts and release heads plus nested donor nomination; same examples and updates across all heads",
        target="direct paired conservative gain interval against same-query anchor, recording-risk deltas and joint feasibility",
        geometry="relative paths among current observed agents only; separate retrained masked-feature control",
        primary="ambiguity-aware ranking vs point regression; promotion also requires beating original with preserved risk",
        original_2pct=True, confirmation_access=False, no_threshold_search=True,
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    return run.sha(f), outer, old, source, gains, ref, producers, releases, nested


def paint(cv, rows, plan):
    p = cv[rows].copy()
    for _, j, path in plan:
        p[np.searchsorted(rows, j)] = path
    return p


def slate_menu(cv, rows, selected, paths, score, analytic, ols):
    variants = []
    for role, row in selected:
        anchor = paths[role][row]
        radius = shape.peak((anchor-cv[row])[None])[0]
        keep = [row] + [int(j) for j in rows[np.argsort(-score[role][rows], kind="stable")] if j != row][:3]
        options = [(role, row, anchor)]
        for j in keep:
            for k, path in enumerate((paths[role][j], ols[j], analytic[j])):
                if j == row and k == 0:
                    continue
                d = path-cv[j]
                limit = shape.peak(d[None])[0]
                if limit >= radius > 0.:
                    options.append((role, j, cv[j]+d*(radius/limit)))
        variants.append(options)
    return [p for p in product(*variants) if len({j for _, j, _ in p}) == len(p)]


def geometry(origin, anchor, proposed, unit):
    a, b = origin[:, None]+anchor, origin[:, None]+proposed
    i, j = np.triu_indices(len(a), 1)
    if not len(i):
        return np.zeros(model.GEOMETRY)
    r, s = a[i]-a[j], b[i]-b[j]
    delta = np.linalg.norm(s-r, axis=2)
    closest = np.linalg.norm(s, axis=2).min(1)-np.linalg.norm(r, axis=2).min(1)
    return np.array([delta.mean(), delta.max(), closest.min(), closest.mean()])/unit


@torch.inference_mode()
def packets(d, cv, ols, path, release, first, analytic):
    paths, features, blocks, gain, easy = clock.prepare(d, cv, path, release, first)
    q, rec, hist = (d[k] for k in ("query", "rec", "h"))
    scale = np.expm1(features[0][:, -1]).clip(1e-8)
    for r, seq in blocks.items():
        a, b, _ = base.model.choose((z, gain[z], easy[z]) for z in seq)
        fixed = tuple((k, j, paths[k][j]) for k, j in enumerate((a, b)) if j is not None)
        at = np.flatnonzero(rec == r)
        full = paint(cv, at, fixed)
        groups = []
        for t in sorted({q[j] for _, j, _ in fixed}):
            rows = at[q[at] == t]
            selected = [(k, j) for k, j, _ in fixed if q[j] == t]
            options = slate_menu(cv, rows, selected, paths, (gain, easy), analytic, ols)
            anchor = paint(cv, rows, options[0])
            unit = float(np.median(scale[rows]))
            feats, preds = [], []
            anchor_feat = []
            for role, j, p in options[0]:
                f, _ = study.c.model.features(hist[j:j+1], cv[j:j+1], p[None])
                anchor_feat.append(np.r_[f[0], easy[j], gain[j]/scale[j], role])
            for plan in options:
                p = paint(cv, rows, plan)
                z = [np.zeros(2*len(anchor_feat[0])+1) for _ in range(2)]
                for i, (role, j, u) in enumerate(plan):
                    f, _ = study.c.model.features(hist[j:j+1], cv[j:j+1], u[None])
                    f = np.r_[f[0], easy[j], gain[j]/scale[j], role]
                    z[role] = np.r_[f, f-anchor_feat[i], 1.]
                feats.append(np.r_[*z, np.log1p(len(rows)), geometry(hist[rows, -1], anchor, p, unit)])
                preds.append(p)
            groups.append((rows, options, np.asarray(feats, np.float32).clip(-16., 16.), preds, unit))
        yield at, full, groups


def targets(d, cv, ols, at, frozen, row, preds, unit, cut):
    pos = np.searchsorted(at, row)
    y, valid = d["y"][at], d["valid"][at]
    cv0, ols0 = cv[at], ols[at]
    floor = record.contributions(cv0, ols0, frozen, y, valid, cut)
    offset, reference = floor[4].sum(), floor[2].sum()
    before = floor[1][(frozen != cv0).any((1, 2))].sum(0)
    results = []
    for p in preds:
        x = frozen.copy()
        x[pos] = p
        obs, low = v.paired(frozen[pos], p, y[pos], valid[pos])
        g, z, easy, mass, _ = record.contributions(cv0, ols0, x, y, valid, cut)
        chosen = (x != cv0).any((1, 2))
        risk = z[chosen].sum(0)
        # Excess terms for unchanged rows are not part of the selected risk ratio.
        delta = risk-before
        label = model.risk_label(g.sum(0), risk, easy[chosen].sum(), mass[chosen].sum(), reference, offset)
        results.append(np.r_[(obs.sum()/12.)/unit, ((obs-low).sum()/12.)/unit, delta/unit, float(label)])
    return np.asarray(results, dtype=np.float32).clip(-8., 8.)


def training(d, cv, ols, path, release, first, analytic, cut):
    x, y, counts = [], [], Counter()
    for at, frozen, groups in packets(d, cv, ols, path, release, first, analytic):
        for row, _, feats, preds, unit in groups:
            z = targets(d, cv, ols, at, frozen, row, preds, unit, cut)
            x.append(feats)
            y.append(z)
            counts.update(queries=1, alternatives=len(z), decisive=int((np.abs(z[:, 0]) > z[:, 1]).sum()))
    if not x:
        return None, counts
    return tuple(torch.from_numpy(np.concatenate(z)) for z in (x, y)), counts


def view(x, k):
    if k == "rank_no_geometry":
        x = x.clone()
        x[:, -model.GEOMETRY:] = 0.
    return x


@torch.inference_mode()
def evaluate(d, bank, path, release, first, analytic, nets, cut, ref):
    cv, ols = bank[:2]
    y, valid, q = (d[k] for k in ("y", "valid", "query"))
    pred = {k: cv.copy() for k in ARMS}
    counts = Counter()
    for at, frozen, groups in packets(d, cv, ols, path, release, first, analytic):
        pred["original"][at] = frozen
        for row, options, feats, paths, unit in groups:
            radius = tuple(shape.peak((p-cv[j])[None])[0] for _, j, p in options[0])
            choices = {}
            for k, net in nets.items():
                z = net(view(torch.from_numpy(feats), k)).numpy()
                choices[k] = model.choose(z)
                if k == "rank":
                    choices["guard"] = model.choose(z, guarded=True)
            for k, ix in choices.items():
                plan = options[ix]
                np.testing.assert_allclose(tuple(shape.peak((p-cv[j])[None])[0] for _, j, p in plan),
                                           radius, rtol=1e-10, atol=1e-12)
                if not all(q[j] == q[row[0]] for _, j, _ in plan):
                    raise ValueError("Ranking changed intervention clock")
                pred[k][row] = paths[ix]
                counts[k+"_changed_queries"] += int(ix != 0)
                counts[k+"_recipient_changes"] += sum(j != a for (_, j, _), (_, a, _) in zip(plan, options[0], strict=True))
            z = targets(d, cv, ols, at, frozen, row, paths, unit, cut[0])
            counts.update(queries=1, multi_role_queries=int(len(options[0]) == 2),
                          queries_with_feasible_positive_swap=int(((z[:, 0] > z[:, 1]) & (z[:, 5] > .5)).any()))
    stats = {k: v.parent.score(d, cv, ols, p, cut) for k, p in pred.items()}
    for f in v.prior.FIELDS:
        if stats["original"][f] != ref["metrics"]["two_roles"][f]:
            raise ValueError("Original two-role predictions did not replay")
    count = lambda p: np.bincount(q[(p != cv).any((1, 2))], minlength=q.max()+1)
    for k in ARMS[1:]:
        np.testing.assert_array_equal(count(pred[k]), count(pred["original"]))
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    pairs = [(k, "cv") for k in ARMS]+[(k, "original") for k in ARMS[1:]]
    pairs += [("rank", "point"), ("rank", "rank_no_geometry"), ("guard", "rank")]
    effects = {f"{a}_vs_{b}": hard.paired_pp(cv if b == "cv" else pred[b], pred[a], y, valid, mass)
               for a, b in pairs}
    return dict(metrics=stats, effects=effects, diagnostics=counts,
                relative_rank_vs_point=study.c.relative_effect(d, pred["point"], pred["rank"]))


def execute(key, outer, old, source, gains, ref, producers, releases, nested, replay, beat):
    sites = {s: study.c.forecast.load(s, source)[0] for s in v.parent.source.SITES}
    banks = {s: v.parent.prior.analytic(d) for s, d in sites.items()}
    analytic = {s: shape.analytic(d["h"])[2] for s, d in sites.items()}
    refs = {(tuple(r["excluded"]), r["target"], r["seed"]): r for r in producers}
    cuts = lambda use: np.quantile(np.concatenate([sites[s]["cut_samples"] for s in use]), [.25, .75])
    hard_width = hard.model.features(next(iter(sites.values()))["h"][:1]).shape[1]
    result, freeze, updates = {}, {}, 0

    def gain_head(meta, d, cv, path):
        n = prior.feature({k: d[k][:1] for k in ("h", "query")}, cv[:1], path[:1])[0].shape[1]
        return two.fit.load(meta, lambda: prior.model.network(n), prior.STEPS)

    for held, d in sites.items():
        use = [s for s in sites if s != held]
        result[held] = []
        for seed in two.study.prior.SEEDS:
            fit, donor = [], Counter()
            for s in use:
                pair = tuple(sorted((held, s)))
                rr = releases["release_"+"_".join(pair)+f"_{seed}"]
                rr_first = nested[f"first_{held}_{s}_{seed}"]
                if (set(rr["identity"]["fit"]) & set(pair)
                        or set(rr_first["identity"]["excluded"]) != set(pair)
                        or set(rr_first["identity"]["fit"]) & set(pair)):
                    raise ValueError("Nomination models expose held or donor labels")
                release = two.fit.load(rr, lambda: hard.model.network(hard_width), hard.STEPS)
                path = record.read_prediction(refs, pair, s, seed)
                cv, ols = banks[s][:2]
                g = gain_head(rr_first, sites[s], cv, path)
                z, info = training(sites[s], cv, ols, path, release, g, analytic[s],
                                   cuts([j for j in use if j != s])[0])
                donor.update(info)
                if z is not None:
                    fit.append(z)
            n = fit[0][0].shape[1]
            name = f"{held}_{seed}"
            nets = {}
            for k in HEADS:
                tag = f"{name}_{k}"
                ident = dict(registration=key, held=held, fit=use, seed=seed, arm=k)
                batch = two.batcher([(view(x, k), y) for x, y in fit])
                loss = lambda m, x, y: model.loss(m, x, y, k != "point")
                nets[k], freeze[tag], u = two.fit.fit(ident, tag, seed, lambda: model.network(n),
                                                       STEPS, batch, loss, replay, beat)
                updates += u
            lookup = lambda a: next(z for z in a["by_site"][held] if z["seed"] == seed)
            release = two.fit.load(outer[name], lambda: hard.model.network(hard_width), hard.STEPS)
            with np.load(run.checked(lookup(old)["prediction"]), allow_pickle=False) as z:
                path = z["comoving_ego"]
            g = gain_head(gains[name], d, banks[held][0], path)
            out = evaluate(d, banks[held], path, release, g, analytic[held], nets, cuts(use), lookup(ref))
            result[held].append(dict(seed=seed, donor=donor, **out))
            beat(phase="site", site=held, seed=seed, updates=updates, alternatives=donor["alternatives"])
    flat = [r for rows in result.values() for r in rows]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(k):
        x = np.array([np.mean([r["effects"][k] for r in group]) for group in result.values()])
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
