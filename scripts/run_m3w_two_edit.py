"""Test capacity and causal choice at the same two-edit budget on OpenDD."""

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
from scripts import run_m3w_sustained_turn as prior
from scripts import run_m3w_recording_count as fit
from src.evaluation.m3w_recording_switch_ceiling import moments
from src.evaluation.m3w_selection_forecast_readout import policy
from src.world_model import m3w_two_edit as model

run, parent, paired = prior.run, prior.parent, prior.paired
study = prior.prior.study
OUT = prior.OUT.parent / "two_edit_v1"
PRIVATE = prior.PRIVATE.parent / OUT.name
ARMS = ("independent", "joint", "joint_geometry")
STEPS = 400


def register():
    key, a, ph, _, _, old, source = prior.register()
    turn = prior.OUT / "analysis.json"
    radius = prior.OUT / "radius_ceiling_v2" / "analysis.json"
    receipt = run.read(turn)
    if receipt["registration"] != key or receipt["confirmation_access"] or run.read(radius)["independent_confirmation"]:
        raise ValueError("Require the frozen one-edit evidence")
    f = OUT / "registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        sources=[run.artifact(p) for p in (turn, radius)], native_steps=[8, 12], seeds=study.prior.SEEDS,
        primary="paired full-grid lower gain, joint versus independent; same clock, bank, model and exactly two edits",
        clock="first legal query with two distinct eligible recipients and at least one unchanged row",
        diagnosis="exact finite-bank one and two edits with original query limits and recording risk; no chronological credit in ceiling",
        feature="frozen motion candidate features plus geometry of the actual emitted path",
        training="site-balanced batches, held location excluded from fitting and cut construction",
        thresholds="fixed penalties 2/4/1 for risk, .025 coverage and .10 normalized separation debit; no threshold search",
        steps=STEPS, batch=128, threads=4, workers=0, original_2pct=True, confirmation_access=False)))
    return run.sha(f), a, ph, old, source


def coefficients(cv, ols, y, valid, row, path, cut):
    m = moments(cv[row], path, y[row], valid[row], cut)
    offset = paired(ols, cv, y, valid)[1].sum(1)
    cv_gain = paired(cv[row], path, y[row], valid[row])[1].sum(1)
    ols_gain = paired(ols[row], path, y[row], valid[row])[1].sum(1)-offset[row]
    risk = np.stack([m[k] for k in ("all_excess", "easy_excess", "easy_signed")], axis=1)
    return np.column_stack((cv_gain, ols_gain)), risk, m["easy"], offset.sum()


def finite_capacity(cv, ols, y, valid, row, path, query, capacity, cuts):
    g, risk, easy, offset = coefficients(cv, ols, y, valid, row, path, cuts[0])
    lim = np.array([0., 0., .02*moments(cv, cv, y, valid, cuts[0])["ref_easy"].sum()])
    one = easy & (risk <= lim).all(1) & (g[:, 0] > 0.) & (g[:, 1]+offset > 0.)
    i, j = model.pairs(row, query[row], capacity)
    both = g[i]+g[j]
    ok = (easy[i] | easy[j]) & ((risk[i]+risk[j]) <= lim).all(1) & (both[:, 0] > 0.) & (both[:, 1]+offset > 0.)
    same = query[row[i]] == query[row[j]]
    mixed = easy[i] ^ easy[j]
    hard = moments(cv, cv, y, valid, cuts[1])["base"][row] > cuts[1]
    mixed_hard = (easy[i] & hard[j]) | (easy[j] & hard[i])
    evidence = dict(one=one, pair=ok, same_query=ok & same, separate_queries=ok & ~same,
                    easy_and_other=ok & mixed, easy_and_hard=ok & mixed_hard,
                    neither_single_wins=ok & ~one[i] & ~one[j])
    for k, hit in evidence.items():
        if not hit.any():
            continue
        k0 = np.flatnonzero(hit)[0]
        chosen = [k0] if k == "one" else [i[k0], j[k0]]
        p = cv.copy()
        p[row[chosen]] = path[chosen]
        score = parent.score(dict(y=y, valid=valid, rec=np.full(len(cv), "local")), cv, ols, p, cuts)
        if score["dual_positive"] != 1 or score["edits"] != len(chosen):
            raise ValueError("Two-edit witness failed the original scorer")
    return {k: bool(v.any()) for k, v in evidence.items()}


def prepare(d):
    bank = parent.prior.analytic(d)
    cv, ols, _, x, scale, legal = bank
    paths = prior.prior.emission_bank(d, bank)
    distances = [np.linalg.norm(paths-v[:, None], axis=-1)/scale[:, None, None] for v in (cv, ols)]
    extra = np.stack([f(v) for v in distances for f in (lambda z: z.mean(2), lambda z: z[:, :, -1])], -1)
    return bank, paths, np.concatenate((x, extra), -1).clip(-16., 16.).astype(np.float32)


def training(d, packet, cut):
    bank, paths, x = packet
    cv, ols, _, _, scale, legal = bank
    row, arm = np.nonzero(legal)
    g, r, easy, _ = coefficients(cv, ols, d["y"], d["valid"], row, paths[row, arm], cut)
    z = np.c_[g/12., r]/scale[row, None]
    return torch.from_numpy(x[row, arm]), torch.from_numpy(np.c_[z.clip(-8., 8.), easy].astype(np.float32))


def batcher(data):
    def batch(rng):
        n = rng.multinomial(128, np.full(len(data), 1./len(data)))
        use = [rng.integers(len(x), size=k) for (x, _), k in zip(data, n, strict=True)]
        return tuple(torch.cat([d[i][j] for d, j in zip(data, use, strict=True)]) for i in range(2))
    return batch


def select(d, packet, rows, net):
    bank, p, x = packet
    cv, _, _, _, scale, legal = bank
    i, arm = np.nonzero(legal[rows])
    r = rows[i]
    u, v = model.pairs(r, np.zeros(len(r), dtype=int), [2])
    if not len(u):
        return None
    q = p[r, arm]
    unit = np.median(scale[rows])
    y = net(torch.from_numpy(x[r, arm])).numpy()*np.c_[np.repeat(scale[r, None]/unit, 5, axis=1), np.ones(len(r))]
    base = np.linalg.norm((d["h"][r[u], -1, None]+cv[r[u]])-(d["h"][r[v], -1, None]+cv[r[v]]), axis=-1).min(1)
    edited = np.linalg.norm((d["h"][r[u], -1, None]+q[u])-(d["h"][r[v], -1, None]+q[v]), axis=-1).min(1)
    debit = (base-edited).clip(min=0.)/unit
    choice = model.rank(y, u, v, debit)
    return {k: (r[[u[t], v[t]]], q[[u[t], v[t]]]) for k, t in choice.items()}


def pair_diagnostic(d, cv, predictions, rows):
    i, j = np.triu_indices(len(rows), k=1)
    i, j = rows[i], rows[j]
    delta = d["h"][i, -1, None]-d["h"][j, -1, None]
    valid = d["valid"][i] & d["valid"][j]
    y = delta+d["y"][i]-d["y"][j]
    ref = delta+cv[i]-cv[j]
    f = {k: delta+p[i]-p[j] for k, p in predictions.items()}
    return dict(known=int(valid.sum()), mass=float(np.linalg.norm(ref-y, axis=-1)[valid].sum()),
                **{k: float(paired(f["independent"], f[k], y, valid)[1].sum()) for k in ARMS[1:]})


def forecast_diagnostic(d, cv, pred, cuts):
    y, valid = d["y"], d["valid"]
    e = np.linalg.norm(pred-y, axis=-1)
    full, end = valid.all(1), valid[:, -1]
    risk = policy(pred, cv, cv, y, valid, d["rec"], easy_cut=cuts[0], hard_cut=cuts[1])["recordings"]
    return dict(complete_rows=int(full.sum()), endpoint_rows=int(end.sum()),
                complete_ade=float(e[full].mean()), endpoint_fde=float(e[end, -1].mean()),
                p95_complete_ade=float(np.quantile(e[full].mean(1), .95)),
                worst_known_easy_net=float(max(v["known_easy_net_degradation"] for v in risk.values()
                                               if v["known_easy_net_degradation"] is not None)))


@torch.inference_mode()
def evaluate(d, packet, menu, rec, net, cuts):
    bank, paths, _ = packet
    cv, ols, _, _, _, legal = bank
    forecasts = {k: cv.copy() for k in ARMS}
    q, y, valid = d["query"], d["y"], d["valid"]
    names, inv = np.unique(rec, return_inverse=True)
    active = menu["legal"][:, 1]
    short = np.bincount(inv, weights=active) < 5
    short_rows = short[inv[q]]
    oracle = Counter(views=int(short.sum()))
    trigger = Counter()
    relative = Counter()
    for name, is_short in zip(names, short, strict=True):
        at = np.flatnonzero(d["rec"] == name)
        query, local = np.unique(q[at], return_inverse=True)
        sizes = np.bincount(local)
        if is_short:
            i, arm = np.nonzero(legal[at] & active[q[at], None])
            hit = finite_capacity(cv[at], ols[at], y[at], valid[at], i, paths[at[i], arm], local,
                                  np.minimum(2, sizes-1), cuts)
            oracle.update({k: int(v) for k, v in hit.items()})
            oracle["one_or_pair"] += int(hit["one"] or hit["pair"])
            oracle["new_mixed_hard"] += int(not hit["one"] and hit["easy_and_hard"])
        for i, s in enumerate(sizes):
            if s < 3 or not active[query[i]]:
                continue
            rows = at[local == i]
            choice = select(d, packet, rows, net)
            if choice is None:
                continue
            for k, (r, p) in choice.items():
                forecasts[k][r] = p
            trigger["short" if is_short else "long"] += 1
            relative.update(pair_diagnostic(d, cv, forecasts, rows))
            break
    full, short_metrics, diagnostic = {}, {}, {}
    for k, p in forecasts.items():
        full[k] = parent.score(d, cv, ols, p, cuts)
        if full[k]["edits"] != 2*sum(trigger.values()):
            raise ValueError("Causal two-edit arms must emit equal, nonzero edits")
        short_metrics[k] = parent.score({s: d[s][short_rows] for s in ("rec", "y", "valid")},
                                       cv[short_rows], ols[short_rows], p[short_rows], cuts) if short.any() else None
        diagnostic[k] = forecast_diagnostic(d, cv, p, cuts)
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    effects = {k: float(100.*paired(forecasts["independent"], forecasts[k], y, valid)[1].sum()/mass) for k in ARMS[1:]}
    return dict(full=full, short=short_metrics, oracle=oracle, effects=effects, trigger=trigger,
                relative=relative, diagnostic=diagnostic)


def execute(key, a, ph, old, source, replay, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    packets = {s: prepare(d) for s, d in sites.items()}
    cells, heads, updates = {}, {}, 0
    for s, d in sites.items():
        donors = [u for u in sites if u != s]
        cut = lambda names: np.quantile(np.concatenate([sites[u]["cut_samples"] for u in names]), [.25, .75])
        cuts = cut(donors)
        data = [training(sites[u], packets[u], cut([v for v in donors if v != u])[0]) for u in donors]
        sample = batcher(data)
        with np.load(source.PRIVATE / f"{s}_cohort.npz", allow_pickle=False) as z:
            rec = z["recording"]
        cells[s] = []
        for row in a["by_site"][s]:
            seed = row["seed"]
            name = f"{s}_{seed}"
            identity = dict(registration=key, held=s, fit=donors, seed=seed)
            net, heads[name], u = fit.fit(identity, name, seed, lambda: model.network(data[0][0].shape[1]),
                                           STEPS, sample, model.loss, replay, beat)
            updates += u
            with torch.inference_mode():
                rank = study.load(ph["heads"][f"{s}/{seed}/additive_rank"],
                                  lambda: study.prior.model.SwapValue(False), study.prior.STEPS)
                menu = study.menus(d, packets[s][0], rank)
                v = evaluate(d, packets[s], menu, rec, net, cuts)
            before = next(r for r in old["by_site"][s] if r["seed"] == seed)["one_edit_oracle"]["all_legal"]
            if (v["oracle"]["views"], v["oracle"]["one"]) != (before.get("views", 0), before.get("views_with_both", 0)):
                raise ValueError("One-edit counterfactual failed to reproduce")
            cells[s].append(dict(seed=seed, **v))
            beat(phase="readout", held=s, seed=seed)
    flat = [c for v in cells.values() for c in v]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(fn):
        x = np.array([np.mean([fn(r) for r in v]) for v in cells.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_location=float(x.min()))

    fields = ("edits", "views", "undefined", "risk_pass", "dual_positive", "unknown_edits")
    summary = {k: {a: {f: sum(c[k][a][f] for c in flat if c[k] is not None) for f in fields} for a in ARMS}
               for k in ("full", "short")}
    summary["oracle"] = sum((Counter(c["oracle"]) for c in flat), Counter())
    summary["trigger"] = sum((Counter(c["trigger"]) for c in flat), Counter())
    summary["paired_gain_vs_independent"] = {k: ci(lambda r: r["effects"][k]) for k in ARMS[1:]}
    summary["relative_path_pooled"] = {k: sum(c["relative"][k] for c in flat) for k in ("known", "mass", *ARMS[1:])}
    f = OUT / "head_freeze.json"
    run.once(f, run.stable(dict(registration=key, heads=heads)))
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, heads=run.sha(f), summary=summary,
        by_site=cells, original_2pct=True, confirmation_access=False, submission_ready=False)))
    return updates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    fit.PRIVATE = PRIVATE
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", z)
            if v["phase"] != "fit":
                print(z, flush=True)

        updates = execute(*register(), args.replay, beat)
        beat(phase="complete", updates=updates)


if __name__ == "__main__":
    main()
