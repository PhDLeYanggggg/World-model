"""Replay the learned policies and attribute unmatched utility to expired edits."""

from collections import Counter
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_wait_peer_value as experiment

run, torch, parent, study = experiment.run, experiment.torch, experiment.parent, experiment.study
prior, model = experiment.prior, experiment.model
OUT = experiment.OUT / "expiry"


def reason(q, slots, proxy, release, cap, chosen):
    t = release[chosen]
    later = (slots == chosen) & (q > t)
    if t == len(cap)-1:
        return "recording_ended"
    if not later.any():
        return "no_legal_radius_recipient"
    if not (proxy[later] > 0.).any():
        return "no_positive_prefix_candidate"
    return "destination_blocked_by_reserve_cap_or_anchor"


@torch.inference_mode()
def readout(d, b, menu, count, rec, heads, ref):
    counts = {k: Counter() for k in model.MODES}
    loss = {k: 0. for k in model.MODES}
    worst = {k: 0. for k in model.MODES}
    y, valid, cv = d["y"], d["valid"], b[0]
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    for at, z, _, paths, args in experiment.packets(d, b, menu, count, rec):
        q, rows, slots, proxy, release, old, _, cap = args
        for i, k in enumerate(model.MODES):
            v = heads[k](torch.from_numpy(z[:, i].copy())).numpy()
            p = expit(v[:, :3])
            choice = model.causal_pick(release, p[:, 2], p[:, 0], v[:, 3])
            if choice is None:
                continue
            pick, _, _ = prior.model.walk(*args, lambda a, _: a == choice)
            if len(pick) == len(old):
                continue
            if len(pick) != len(old)-1 or not np.array_equal(pick, np.delete(old, choice)):
                raise ValueError("Expired decision changed another slot")
            counts[k][reason(q, slots, proxy, release, cap, choice)] += 1
            edge = old[choice]
            row = at[rows[edge]]
            delta = float(100.*prior.prior.paired(paths[edge:edge+1], cv[row:row+1],
                         y[row:row+1], valid[row:row+1])[1].sum()/mass)
            loss[k] += delta
            worst[k] = min(worst[k], delta)
    for k in model.MODES:
        if sum(counts[k].values()) != ref["decisions"][k].get("expired", 0):
            raise ValueError("Expiration replay diverged")
        expected = ref["direct"][f"{k}_vs_anchor"]-ref["direct"][f"{k}_vs_emission_match"]
        np.testing.assert_allclose(loss[k], expected, atol=1e-10, rtol=0.)
    return dict(counts=counts, expiry_gain_pp=loss, worst_single_expiry_pp=worst)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, a, ph, h, *_ = experiment.register()
    p, f = (experiment.OUT / f"{k}.json" for k in ("analysis", "head_freeze"))
    source, weights = run.read(p), run.read(f)
    if (any(x["registration"] != key for x in (source, weights)) or source["heads"] != run.sha(f)
            or source["confirmation_access"] or not source["original_2pct"]):
        raise ValueError("Require the frozen causal policy readout")
    OUT.mkdir(parents=True, exist_ok=True)
    reg = OUT / "registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), parents=[run.artifact(x) for x in (p, f)],
        attribution="single-toggle full minus emission-matched gain equals expired original edit contributions",
        no_training=True, confirmation_access=False, original_2pct=True)))
    cells = {}
    for s in parent.source.SITES:
        d = parent.prior.prior.old.load_site(s)
        b = parent.prior.analytic(d)
        cache = parent.prior.prior.old.source.prior.PRIVATE / f"{s}_cohort.npz"
        with np.load(cache, allow_pickle=False) as z:
            rec, start = z["recording"], z["start"]
        cells[s] = []
        for cell in a["by_site"][s]:
            seed = cell["seed"]
            name = f"{s}_{seed}"
            m, count = prior.frozen_count(d, b, rec, start,
                ph["heads"][f"{s}/{seed}/additive_rank"], h["heads"][name+"_motion"])
            first = next(experiment.packets(d, b, m, count, rec), None)
            if first is None:
                raise ValueError("Expected at least one released opportunity per fold")
            width = first[1].shape[-1]
            heads = {k: study.load(weights["heads"][f"{name}_{k}"], lambda: model.Value(width), experiment.STEPS)
                     for k in model.MODES}
            ref = next(r for r in source["by_site"][s] if r["seed"] == seed)
            cells[s].append(dict(seed=seed, **readout(d, b, m, count, rec, heads, ref)))
        print(s, flush=True)
    flat = [r for v in cells.values() for r in v]
    summary = {k: dict(expiries=sum((Counter(r["counts"][k]) for r in flat), Counter()),
        macro_gain_pp=float(np.mean([np.mean([r["expiry_gain_pp"][k] for r in v]) for v in cells.values()])),
        worst_single_expiry_pp=min(r["worst_single_expiry_pp"][k] for r in flat)) for k in model.MODES}
    run.once(OUT / "analysis.json", run.stable(dict(registration=run.sha(reg), summary=summary,
        by_site=cells, no_training=True, confirmation_access=False, original_2pct=True)))


if __name__ == "__main__":
    main()
