"""Test donor motion shapes at the frozen OpenDD edit budget."""

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
from src.world_model import m3w_donor_motion_shape as model

run, parent, ledger, paired = prior.run, prior.parent, prior.ledger, prior.paired
OUT = prior.OUT.parent / "donor_motion_shape_v1"
PRIVATE = prior.PRIVATE.parent / OUT.name
ARMS = ("anchor", "forced_match", "forced_shuffle", "prefix_match", "prefix_shuffle")


def register():
    prior_key, a, ph, h, _, _, source = prior.register()
    old = run.read(prior.OUT / "analysis.json")
    if old["registration"] != prior_key or old["confirmation_access"] or not old["original_2pct"]:
        raise ValueError("Require the frozen radius-matched development reference")
    f = OUT / "registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parent=run.artifact(prior.OUT / "analysis.json"), cohort=run.artifact(source.PUBLIC / "registration.json"),
        original_2pct=True, confirmation_access=False, native_steps=[8, 12], k=model.K,
        null_seed=model.NULL_SEED, donor_rule="other six sites only; complete futures only for 8/12 library",
        null_rule="permute motion-frame offsets within donor site, same histories and nearest indices",
        primary="prefix match minus original; same nominees, query times and peak radii",
        secondary="forced match and prefix/forced shuffled controls; short-recording exact one-edit capacity",
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    return run.sha(f), a, ph, h, old, source


def choices(history, legal, offsets):
    paths = np.zeros((*legal.shape, len(ARMS), 12, 2))
    chosen = np.zeros((*legal.shape, 2), bool)
    for arm, amp in enumerate((.25, .5, 1.)):
        at = np.flatnonzero(legal[:, arm])
        for i in range(0, len(at), 512):
            r = at[i:i+512]
            _, anchor, candidates, selected, use = model.choose(
                history[r], np.full(len(r), amp), {k: v[r] for k, v in offsets.items()})
            paths[r, arm] = np.concatenate((anchor[:, None], candidates, selected), axis=1)
            chosen[r, arm] = use
    return paths, chosen


def score(d, bank, menu, count, rec, cuts, paths, chosen):
    cv, ols, _, _, _, legal = bank
    use = np.arange(2) < count[:, None]
    at, arm = menu["row"][use], menu["arm"][use]
    y, valid, query = d["y"], d["valid"], d["query"]
    forecasts = {}
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    for j, name in enumerate(ARMS):
        p = cv.copy()
        p[at] = paths[at, arm, j]
        np.testing.assert_array_equal(np.flatnonzero((p != cv).any((1, 2))), np.sort(at))
        np.testing.assert_allclose(model.peak(p[at]-cv[at]),
                                  model.peak(paths[at, arm, 0]-cv[at]), atol=1e-10)
        forecasts[name] = p
    effects = {}
    for name, ref in [(k, "anchor") for k in ARMS[1:]] + [
            ("forced_match", "forced_shuffle"), ("prefix_match", "prefix_shuffle")]:
        p, q = forecasts[name], forecasts[ref]
        effects[f"{name}_vs_{ref}"] = float(100.*paired(q, p, y, valid)[1].sum()/mass)
    results = {k: parent.score(d, cv, ols, p, cuts) for k, p in forecasts.items()}
    active = menu["legal"][:, 1]
    groups, inv = np.unique(rec, return_inverse=True)
    short = np.bincount(inv, weights=active) < 5
    capacity = Counter(views=int(short.sum()))
    for recording in groups[short]:
        r = np.flatnonzero(d["rec"] == recording)
        i, j = np.nonzero(legal[r] & active[query[r], None])
        args = cv[r], ols[r], y[r], valid[r], i
        old = prior.prior.one_edit(*args, paths[r[i], j, 0], cuts)["both"] > 0
        capacity["anchor"] += old
        for k, name in enumerate(ARMS[1:3], 1):
            new = prior.prior.one_edit(*args, paths[r[i], j, k], cuts)["both"] > 0
            capacity[name] += old or new
    return dict(policies=results, effects=effects, capacity=capacity,
                prefix_chose_extra=chosen[at, arm].sum(0).tolist())


def execute(key, a, ph, h, old, source, beat, replay):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    out = {}
    for s, d in sites.items():
        f = PRIVATE / f"{s}.json"
        fresh = not f.exists()
        if fresh or replay:
            donors = {k: v for k, v in sites.items() if k != s}
            libs = {k: model.Library.fit(donors, s, k) for k in (*model.PREFIXES, 8)}
            pred = {k: lib.predict(d["h"][:, :k]) for k, lib in libs.items()}
            bank = parent.prior.analytic(d)
            paths, chosen = choices(d["h"], bank[-1], {k: v[0] for k, v in pred.items()})
            with np.load(source.PRIVATE / f"{s}_cohort.npz", allow_pickle=False) as z:
                rec, starts = z["recording"], z["start"]
            cuts = np.quantile(np.concatenate([v["cut_samples"] for v in donors.values()]), [.25, .75])
            cells = []
            for row in a["by_site"][s]:
                seed = row["seed"]
                menu, count = ledger.frozen_count(d, bank, rec, starts,
                    ph["heads"][f"{s}/{seed}/additive_rank"], h["heads"][f"{s}_{seed}_motion"])
                x = score(d, bank, menu, count, rec, cuts, paths, chosen)
                ref = next(r for r in old["by_site"][s]["cells"] if r["seed"] == seed)
                for k, v in ref["policies"]["anchor"].items():
                    np.testing.assert_allclose(x["policies"]["anchor"][k], v, atol=1e-10, rtol=0.)
                if any(x["capacity"][k] != ref["oracle"].get(k, 0) for k in ("views", "anchor")):
                    raise ValueError("Frozen short-recording capacity changed")
                cells.append(dict(seed=seed, **x))
            run.once(f, run.stable(dict(registration=key, donors=sorted(donors), cells=cells, rows=len(d["h"]),
                full_donor_windows=libs[8].tree.n,
                kth_distance_quantiles=np.quantile(pred[8][1], [.25, .5, .75]).tolist())))
        out[s] = run.read(f)
        if out[s]["registration"] != key:
            raise ValueError("Cached score registration changed")
        beat(site=s, fresh=fresh, replay=replay)

    flat = [c for v in out.values() for c in v["cells"]]
    draw = np.random.default_rng(817).integers(len(out), size=(3000, len(out)))
    effects = {}
    for k in flat[0]["effects"]:
        x = np.asarray([np.mean([r["effects"][k] for r in z["cells"]]) for z in out.values()])
        effects[k] = dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                          worst_site=float(x.min()))
    totals = {k: {field: sum(c["policies"][k][field] for c in flat)
                 for field in ("edits", "views", "undefined", "risk_pass", "dual_positive")} for k in ARMS}
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, by_site=out, summary=dict(
        effects=effects, policies=totals, capacity=sum((Counter(c["capacity"]) for c in flat), Counter()),
        prefix_chose_extra=np.sum([c["prefix_chose_extra"] for c in flat], axis=0).tolist()),
        original_2pct=True, confirmation_access=False, no_neural_refit=True, submission_ready=False)))


def main():
    if sys.argv[1:] not in ([], ["--replay"]):
        raise SystemExit("Usage: run_m3w_donor_motion_shape.py [--replay]")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", z)
            print(z, flush=True)

        execute(*register(), beat, "--replay" in sys.argv)
        beat(complete=True)


if __name__ == "__main__":
    main()
