"""Read out sustained-turn candidates on exposed OpenDD, before any refit."""

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
from scripts import run_m3w_first_credit as prior
from src.world_model import m3w_sustained_turn as model

run, parent, ledger, paired = prior.run, prior.parent, prior.ledger, prior.paired
OUT = prior.OUT.parent / "sustained_turn_v2"
PRIVATE = prior.PRIVATE.parent / OUT.name


def register():
    key, a, ph, h, old = prior.register()
    p = prior.OUT / "analysis.json"
    prev = run.read(p)
    source = parent.prior.prior.old.source.prior
    registration = source.PUBLIC / "registration.json"
    if prev["registration"] != key or prev["confirmation_access"] or not prev["original_2pct"]:
        raise ValueError("Require verified original budget and first-credit audit")
    for s in parent.source.SITES:
        cache = source.PRIVATE / f"{s}_cohort.npz"
        receipt = run.read(cache.with_suffix(".json"))
        if receipt["registration"] != run.sha(registration) or receipt["sha256"] != run.sha(cache):
            raise ValueError("OpenDD cache changed")
    f = OUT / "registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parent=run.artifact(p), cohort=run.artifact(registration), native_steps=[8, 12],
        previous_aborted=run.artifact(OUT.parent / "sustained_turn_v1" / "registration.json"),
        retry="interpret missing historical Counter fields as zero, not a new roster",
        seeds=prior.study.prior.SEEDS, original_2pct=True, confirmation_access=False, new_training=False,
        raw="same initial speed and tangent; straight, decayed turn, constant turn; no refit",
        primary="legacy menu plus one candidate each; unchanged observed-prefix scores, rows and radii",
        ceiling="one-edit exact hindsight on prior zero-credit views, legacy emitted candidate plus one same-radius alternative",
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    return run.sha(f), a, ph, h, old, prev, source


def raw_metrics(d, cv, ols, paths):
    y, valid = d["y"], d["valid"]
    full, end = valid.all(1), valid[:, -1]
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    ref = dict(cv=cv, ols4=ols, **dict(zip(model.NAMES, paths.swapaxes(0, 1), strict=True)))
    out = {}
    for k, p in ref.items():
        e = np.linalg.norm(p-y, axis=-1)
        lower = lambda a: float(100.*paired(a, p, y, valid)[1].sum()/mass)
        out[k] = dict(complete_ade=float(e[full].mean()), endpoint_fde=float(e[end, -1].mean()),
                      p95_complete_ade=float(np.quantile(e[full].mean(1), .95)), lower_vs_ols4_pp=lower(ols),
                      lower_vs_straight_pp=lower(ref["straight"]))
    return dict(rows=len(cv), complete=int(full.sum()), endpoint=int(end.sum()), arms=out)


def choices(d, legal):
    n = len(legal)
    emission = np.zeros((n, 3, 12, 2))
    repl = {k: np.zeros_like(emission) for k in model.NAMES}
    extra = {k: np.zeros_like(emission) for k in model.NAMES}
    chosen = np.zeros((n, 3, len(model.NAMES)), bool)
    for arm, amp in enumerate((.25, .5, 1.)):
        at = np.flatnonzero(legal[:, arm])
        for i in range(0, len(at), 512):
            r = at[i:i+512]
            base, p, z, mask = model.choose(d["h"][r], np.full(len(r), amp))
            emission[r, arm], chosen[r, arm] = base, mask
            for j, k in enumerate(model.NAMES):
                repl[k][r, arm], extra[k][r, arm] = p[:, j], z[:, j]
    return emission, repl, extra, chosen


def assess(d, bank, menu, count, rec, cuts, path_bank):
    cv, ols, _, _, _, legal = bank
    emission, repl, extra, chosen = path_bank
    use = np.arange(2) < count[:, None]
    at, arm = menu["row"][use], menu["arm"][use]
    y, valid, query = d["y"], d["valid"], d["query"]
    forecasts = {}
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    for k, b in {"anchor": emission, **repl}.items():
        p = cv.copy()
        p[at] = b[at, arm]
        np.testing.assert_array_equal(np.flatnonzero((p != cv).any((1, 2))), np.sort(at))
        np.testing.assert_allclose(model.legacy.peak(p[at]-cv[at]),
                                  model.legacy.peak(emission[at, arm]-cv[at]), atol=1e-10)
        forecasts[k] = p
    results = {k: parent.score(d, cv, ols, p, cuts) for k, p in forecasts.items()}
    gain = lambda a, b: float(100.*paired(forecasts[b], forecasts[a], y, valid)[1].sum()/mass)
    effects = {f"{k}_vs_anchor": gain(k, "anchor") for k in model.NAMES}
    effects["sustained_vs_straight"] = gain("sustained", "straight")
    effects["sustained_vs_tapered"] = gain("sustained", "tapered")

    active = menu["legal"][:, 1]
    groups, inv = np.unique(rec, return_inverse=True)
    short = np.bincount(inv, weights=active) < 5
    oracle = Counter(views=int(short.sum()))
    for recording in groups[short]:
        r = np.flatnonzero(d["rec"] == recording)
        i, j = np.nonzero(legal[r] & active[query[r], None])
        base = emission[r[i], j]
        a = cv[r], ols[r], y[r], valid[r]
        old = prior.one_edit(*a, i, base, cuts)
        oracle["anchor"] += old["both"] > 0
        for k, b in extra.items():
            new = prior.one_edit(*a, i, b[r[i], j], cuts)
            oracle[k] += bool(old["both"] or new["both"])
    return dict(policies=results, effects=effects, oracle=oracle,
                accepted={k: int(chosen[at, arm, j].sum()) for j, k in enumerate(model.NAMES)})


def execute(key, a, ph, h, old, prev, source, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    out = {}
    for s, d in sites.items():
        f = PRIVATE / f"{s}.json"
        fresh = not f.exists()
        if fresh:
            bank = parent.prior.analytic(d)
            paths = choices(d, bank[-1])
            with np.load(source.PRIVATE / f"{s}_cohort.npz", allow_pickle=False) as z:
                rec, starts = z["recording"], z["start"]
            cuts = np.quantile(np.concatenate([v["cut_samples"] for k, v in sites.items() if k != s]), [.25, .75])
            cells = []
            for row in a["by_site"][s]:
                seed = row["seed"]
                m, count = ledger.frozen_count(d, bank, rec, starts,
                    ph["heads"][f"{s}/{seed}/additive_rank"], h["heads"][f"{s}_{seed}_motion"])
                x = assess(d, bank, m, count, rec, cuts, paths)
                reference = next(r for r in old["by_site"][s] if r["seed"] == seed)["metrics"]["anchor"]
                for k, v in reference.items():
                    np.testing.assert_allclose(x["policies"]["anchor"][k], v, atol=1e-10, rtol=0.)
                reference = next(r for r in prev["by_site"][s] if r["seed"] == seed)["one_edit_oracle"]["all_legal"]
                if (x["oracle"]["views"], x["oracle"]["anchor"]) != (
                        reference.get("views", 0), reference.get("views_with_both", 0)):
                    raise ValueError("Legacy capacity did not reproduce")
                cells.append(dict(seed=seed, **x))
            raw = raw_metrics(d, *bank[:2], model.paths(d["h"], 12))
            run.once(f, run.stable(dict(registration=key, cells=cells, raw=raw)))
        out[s] = run.read(f)
        if out[s]["registration"] != key:
            raise ValueError("Cached score registration changed")
        beat(site=s, fresh=fresh)

    flat = [c for v in out.values() for c in v["cells"]]
    draw = np.random.default_rng(817).integers(len(out), size=(3000, len(out)))

    def ci(x):
        z = np.asarray(x)
        return dict(point=float(z.mean()), interval95=np.quantile(z[draw].mean(1), [.025, .975]).tolist(),
                    worst_location=float(z.min()))

    arms = tuple(flat[0]["policies"])
    totals = {k: {field: sum(c["policies"][k][field] for c in flat)
                 for field in ("edits", "views", "undefined", "risk_pass", "dual_positive")} for k in arms}
    summary = dict(policies=totals, effects={k: ci([np.mean([r["effects"][k] for r in z["cells"]]) for z in out.values()])
        for k in flat[0]["effects"]}, oracle=sum((Counter(c["oracle"]) for c in flat), Counter()),
        accepted=sum((Counter(c["accepted"]) for c in flat), Counter()),
        raw_effects={k: {f: ci([v["raw"]["arms"][k][f] for v in out.values()]) for f in
                        ("lower_vs_ols4_pp", "lower_vs_straight_pp")} for k in ("cv", *model.NAMES)})
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, summary=summary, by_site=out,
        original_2pct=True, confirmation_access=False, no_training=True, submission_ready=False)))


def main():
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

        execute(*register(), beat)
        beat(complete=True)


if __name__ == "__main__":
    main()
