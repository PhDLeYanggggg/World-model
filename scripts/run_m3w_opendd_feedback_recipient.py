"""Test resolved past errors at the same OpenDD query counts and radii."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.special import expit

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_recipient_slot as prior
from scripts import run_m3w_ledger_wait as ledger
from src.world_model import m3w_opendd_feedback_recipient as feedback
from src.evaluation.m3w_recording_switch_ceiling import moments

run, parent, torch, study, model = prior.run, prior.parent, prior.torch, prior.study, prior.model
OUT = prior.OUT.parent / "opendd_feedback_recipient_v1"
PRIVATE = prior.PRIVATE.parent / OUT.name
ARMS = ("frozen_prefix", "archived_joint", *feedback.SCOPES)


def register():
    key, a, ph, h, _ = prior.register()
    p, f = [prior.OUT / f"{k}.json" for k in ("analysis", "head_freeze")]
    old, heads = [run.read(x) for x in (p, f)]
    if (old["registration"] != key or heads["registration"] != key or old["heads"] != run.sha(f)
            or old["confirmation_access"] or not old["original_2pct"] or not old["query_radii_exact"]):
        raise ValueError("Require the exact original recipient-slot experiment")
    source = parent.prior.prior.old.source.prior
    site_receipts = []
    for s in parent.source.SITES:
        cache = source.PRIVATE / f"{s}_cohort.npz"
        r = run.read(cache.with_suffix(".json"))
        if r["registration"] != run.sha(source.PUBLIC / "registration.json") or r["sha256"] != run.sha(cache):
            raise ValueError(f"Original OpenDD cache changed: {s}")
        site_receipts.append(run.artifact(cache.with_suffix(".json")))
    reg = OUT / "registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), feedback=run.artifact(Path(feedback.__file__)),
        parents=[run.artifact(x) for x in (p, f, Path(prior.__file__), Path(model.__file__))],
        cache_receipts=site_receipts, native_steps=[8, 12], observation_stride=source.STRIDE,
        history="extension: preceding disjoint completed 8/12 cohorts from the same recording",
        timing="last sampled target tick of previous cohort strictly before current observation start",
        training="held source excluded; three scopes, same architecture, fit data, seed and optimization",
        controls="availability-only and within-query cyclic wrong-agent error; preserve own mask, age and coverage",
        primary="aligned minus availability, fixed query counts and exact radius multiset; wrong-agent specificity",
        seeds=study.prior.SEEDS, arms=ARMS, steps=prior.STEPS, threads=4, workers=0,
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"),
        original_2pct=True, confirmation_access=False)))
    return run.sha(reg), a, ph, h, old, heads["heads"], source


def load(s, source):
    d = parent.prior.prior.old.load_site(s)
    with np.load(source.PRIVATE / f"{s}_cohort.npz", allow_pickle=False) as z:
        rec, starts, ids, offsets = [z[k] for k in ("recording", "start", "id", "offsets")]
    np.testing.assert_array_equal(d["rec"], np.repeat(rec, np.diff(offsets)))
    np.testing.assert_array_equal(d["query"], np.repeat(np.arange(len(rec)), np.diff(offsets)))

    def observe(at, last, now):
        if last >= now or not np.all(starts[d["query"][at]]+19*source.STRIDE == last):
            raise ValueError("Attempted read of unresolved measurements")
        return d["y"][at]

    trace, exchanged = feedback.traces(d["h"], ids, offsets, rec, starts, source.STRIDE, observe)
    return d, rec, starts, trace, dict(rows=len(ids), supported=int(trace["aligned"][:, 0].sum()),
                                       exchanged=exchanged)


def train_data(d, trace, cut):
    cv, ols, _, _ = model.shape.menu.rollout(d["h"], 12)
    on = model.shape.menu.peak(ols-cv) > 0.
    x, y = prior.prepare(d, cut)
    if len(x) != int(on.sum())*len(model.AMPS):
        raise ValueError("Feedback and original training rows differ")
    return {k: (torch.cat((x, torch.from_numpy(np.tile(trace[k][on], (len(model.AMPS), 1)))), dim=1), y)
            for k in feedback.SCOPES}


def assignments(rows, slots, blocks, rec, pred, scale):
    chosen = []
    remaining, last = 1., None
    for q, n, use, _, _ in blocks:
        if rec[q] != last:
            remaining, last = 1., rec[q]
        value = pred[use, :5]*scale[use, None]
        safe = value[:, :2].min(1)-value[:, 2:].clip(min=0.) @ np.array([2., 4., 1.])
        prob = expit(pred[use, 5])
        j = model.assign(rows[use], slots[use] % 2, safe, prob, n, .025*np.median(scale[use])*remaining)
        chosen.extend(use.start+j)
        remaining *= float(np.prod(1.-prob[j]))
    return np.asarray(chosen, dtype=int)


@torch.inference_mode()
def evaluate(d, b, menu, count, rec, trace, nets, cuts, ref):
    cv, ols = b[:2]
    rows, slots, amp, blocks = prior.edges(d, cv, ols, menu, count)
    x, _, _, path, scale = model.features(d["h"][rows], amp)
    predicted = {k: net(torch.from_numpy(x if k == "archived_joint" else np.c_[x, trace[k][rows]])).numpy()
                 for k, net in nets.items()}
    pick = {k: assignments(rows, slots, blocks, rec, v, scale) for k, v in predicted.items()}
    pick["frozen_prefix"] = np.concatenate([use.start+old for _, _, use, old, _ in blocks])
    y, valid = d["y"], d["valid"]
    forecasts, metrics = {}, {}
    # Each block stores the original chronological slots and radii.
    target_slots = slots[pick["frozen_prefix"]]
    target_radii = model.shape.menu.peak(path[pick["frozen_prefix"]]-cv[rows[pick["frozen_prefix"]]])
    easy = moments(cv, cv, y, valid, cuts[0])["easy"]
    for k, at in pick.items():
        np.testing.assert_array_equal(slots[at], target_slots)
        np.testing.assert_allclose(model.shape.menu.peak(path[at]-cv[rows[at]]), target_radii, atol=1e-10, rtol=1e-8)
        if len(np.unique(rows[at])) != count.sum():
            raise ValueError("Changed budget or duplicated recipient")
        f = cv.copy()
        f[rows[at]] = path[at]
        forecasts[k] = f
        v = parent.score(d, cv, ols, f, cuts)
        metrics[k] = dict(v, easy_edits=int(easy[rows[at]].sum()),
                          reassignments=int(np.sum(rows[at] != rows[pick["frozen_prefix"]])))
        if k in predicted:
            # One support score per legal candidate row, first eligible slot.
            _, unique = np.unique(rows, return_index=True)
            p = expit(predicted[k][unique, 5])
            metrics[k]["candidate_support_brier"] = float(np.mean(np.square(p-easy[rows[unique]])))
    for k, old in (("frozen_prefix", "frozen_prefix"), ("archived_joint", "joint_prefix")):
        for name, value in ref["metrics"][old].items():
            np.testing.assert_allclose(metrics[k][name], value, atol=1e-10, rtol=0.)
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    pairs = [("aligned", k) for k in ("availability", "wrong_agent", "archived_joint", "frozen_prefix")]
    effects = {f"{k}_vs_{r}": float(100.*parent.transfer.paired(forecasts[r], forecasts[k], y, valid)[1].sum()/mass)
               for k, r in pairs}
    return dict(metrics=metrics, effects=effects)


def execute(key, a, ph, h, old, archived, source, replay, beat):
    sites = {s: load(s, source) for s in parent.source.SITES}
    result, heads, updates = {}, {}, 0
    for held, (d, rec, starts, trace, _) in sites.items():
        donors = [s for s in sites if s != held]
        cut = np.quantile(np.concatenate([sites[s][0]["cut_samples"] for s in donors]), [.25, .75])
        train = [train_data(sites[s][0], sites[s][3], cut[0]) for s in donors]
        b = parent.prior.analytic(d)
        result[held] = []
        for cell in a["by_site"][held]:
            seed = cell["seed"]
            anchor = archived[f"{held}_{seed}"]
            if (anchor["identity"]["held"], anchor["identity"]["donors"], anchor["identity"]["seed"]) != (held, donors, seed):
                raise ValueError("Archived fit site roster changed")
            width = train[0]["aligned"][0].shape[1]
            nets = {"archived_joint": study.load(anchor, lambda: model.Value(width-7), prior.STEPS)}
            for k in feedback.SCOPES:
                name = f"{held}_{seed}_{k}"
                identity = dict(registration=key, held=held, donors=donors, seed=seed, scope=k)
                batch = prior.batcher([row[k] for row in train])
                nets[k], heads[name], u = study.fit(identity, name, seed, lambda: model.Value(width), prior.STEPS,
                                                     batch, model.objective, replay, beat)
                updates += u
            m, n = ledger.frozen_count(d, b, rec, starts, ph["heads"][f"{held}/{seed}/additive_rank"],
                                      h["heads"][f"{held}_{seed}_motion"])
            ref = next(r for r in old["by_site"][held] if r["seed"] == seed)
            result[held].append(dict(seed=seed, **evaluate(d, b, m, n, rec, trace, nets, cut, ref)))
        beat(phase="held", held=held, updates=updates, support=sites[held][4])

    flat = [r for v in result.values() for r in v]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def interval(fn):
        x = np.asarray([np.mean([fn(r) for r in v]) for v in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()))

    fields = ("edits", "views", "undefined", "risk_pass", "dual_positive", "easy_edits", "reassignments")
    table = {k: {f: sum(r["metrics"][k][f] for r in flat) for f in fields} for k in ARMS}
    for k in ARMS[1:]:
        table[k]["candidate_support_brier"] = interval(lambda r: r["metrics"][k]["candidate_support_brier"])
    summary = dict(policies=table, effects={k: interval(lambda r: r["effects"][k]) for k in flat[0]["effects"]})
    freeze = OUT / "head_freeze.json"
    run.once(freeze, run.stable(dict(registration=key, heads=heads)))
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, heads=run.sha(freeze),
        summary=summary, by_site=result, coverage={s: v[4] for s, v in sites.items()},
        original_2pct=True, confirmation_access=False, submission_ready=False)))
    return updates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    study.PRIVATE = PRIVATE
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            event = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", event)
            if v.get("phase") != "fit":
                print(event, flush=True)

        u = execute(*register(), a.replay, beat)
        beat(phase="complete", updates=u, replay=a.replay)


if __name__ == "__main__":
    main()
