"""Match causal hindcast replacements within every current query."""

import fcntl
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_tumdot_peer_choice as study

run, torch, trainer, INC = study.run, study.torch, study.trainer, study.INC
parent, prior, ARMS = study.parent, study.prior, study.ARMS
OUT = study.OUT.parent/"tumdot_peer_budget_v1"
PRIVATE = study.PRIVATE.parent/OUT.name


def match(bank, scores, choices, offsets):
    row = np.arange(len(bank))
    change = {k: (bank[row, a] != bank[:, INC]).any((1, 2)) for k, a in choices.items()}
    selected = {k: np.full(len(bank), INC, dtype=int) for k in choices}
    for lo, hi in zip(offsets[:-1], offsets[1:], strict=True):
        count = min(int(mask[lo:hi].sum()) for mask in change.values())
        for k, a in choices.items():
            use = np.flatnonzero(change[k][lo:hi])+lo
            order = np.argsort(-scores[k][use, a[use], 0], kind="stable")
            ix = use[order[:count]]
            selected[k][ix] = a[ix]
            if int((bank[row[lo:hi], selected[k][lo:hi]] != bank[lo:hi, INC]).any((1, 2)).sum()) != count:
                raise ValueError("Changed realized query replacement count")
    return selected


@torch.inference_mode()
def score(rec, d, z, offsets, nets, cuts, old):
    cv, ols, bank, on = (z[k] for k in ("cv", "ols", "bank", "on"))
    y, valid = d["future"], d["valid"]
    scores = {k: parent.model.centered(v, torch.from_numpy(z["features"][k]), INC).numpy() for k, v in nets.items()}
    raw = {k: parent.model.choices(v, INC, True) for k, v in scores.items()}
    selected = match(bank, scores, raw, offsets)
    row = np.arange(len(bank))
    h = bank[:, INC]
    paths = dict(hindcast=h, **{a: bank[row, c] for a, c in selected.items()})
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    for k, c in raw.items():
        p = bank[row, c]
        n = int((p != h).any((1, 2)).sum())
        lower = 100*study.study.paired(h, p, y, valid)[1].sum()/mass
        np.testing.assert_allclose(lower, old["contrasts"][k+"_guard_vs_hindcast"]["lower"], atol=1e-12, rtol=0.)
        if n != old["metrics"][k+"_guard"]["changed_from_hindcast"]:
            raise ValueError("Frozen guard did not replay")
    metrics = {}
    for k, p in paths.items():
        np.testing.assert_array_equal((p != cv).any((1, 2)), on)
        np.testing.assert_allclose(np.linalg.norm(p-cv, axis=-1).max(1), z["radius"], rtol=1e-7, atol=1e-13)
        metrics[k] = study.study.source.score(p, cv, ols, y, valid, rec, cuts, h)
        metrics[k].update(edits=int(on.sum()), changed_from_hindcast=int((p != h).any((1, 2)).sum()))
    pairs = [(k, "hindcast") for k in ARMS]+[("aligned", k) for k in ("blind", "permuted")]
    contrasts = {}
    for a, b in pairs:
        known, lower = study.study.paired(paths[b], paths[a], y, valid)
        contrasts[a+"_vs_"+b] = dict(lower=float(100*lower.sum()/mass), observed=float(100*known.sum()/mass))
    return dict(metrics=metrics, contrasts=contrasts)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic()

        def beat(**p):
            run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", dict(pid=os.getpid(),
                seconds=time.monotonic()-start, **p))

        key, _ = study.register()
        paths = [study.OUT/f"{k}.json" for k in ("analysis", "head_freeze", "registration")]
        a, h = map(run.read, paths[:2])
        if (a["registration"] != key or h["registration"] != key or a["heads"] != run.sha(paths[1])
                or a["confirmation_access"] or not a["original_2pct"]):
            raise ValueError("Require original peer comparison")
        f = OUT/"registration.json"
        run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), parents=[run.artifact(p) for p in paths],
            original_2pct=True, confirmation_access=False, no_new_training=True,
            match="current-query minimum of the three original guarded counts; keep each arm's top predicted-gain changes",
            scope="outcome-blind matched comparison, not a new deployment rule", roster="same 27 repeated views")))
        k0, lineage, base, _ = prior.register()
        cache = prior.prepare(k0, lineage, base, beat)
        offsets = study.offsets_for(key, cache, beat)
        cuts = np.asarray(lineage[-1]["open_dd_positive_complete_cv_cuts"])
        rows = {}
        for seed in study.study.parent.job.SEEDS:
            for rec, d in cache.items():
                z = {p: study.task(d, seed, frac, offsets[rec]) for p, frac in study.study.FRACTIONS.items()}
                width = z["half"]["features"]["blind"].shape[-1]
                nets = {k: trainer.load(h["heads"][f"{rec}_{seed}_{k}"], lambda: prior.model.network(width),
                                        study.STEPS) for k in ARMS}
                old = next(v["scores"] for v in a["rows"][rec] if v["seed"] == seed)
                v = {k: score(rec, d, p, offsets[rec], nets, cuts, old[k]) for k, p in z.items()}
                rows.setdefault(rec, []).append(dict(seed=seed, scores=v))
                beat(phase="held", recording=rec, seed=seed)
        run.once(OUT/"analysis.json", run.stable(dict(registration=run.sha(f), rows=rows,
            summary=parent.aggregate(rows), original_2pct=True, confirmation_access=False, no_new_training=True,
            source_is_previously_exposed=True)))
        beat(complete=True)


if __name__ == "__main__":
    main()
