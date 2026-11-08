"""Fit one-edit whole-recording targets on the exposed OpenDD development roster."""

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
from scripts import run_m3w_cross_query_edit as prior
from src.world_model import m3w_first_edit_target as model

run, parent, paired, two = prior.run, prior.parent, prior.paired, prior.prior
OUT = prior.OUT.parent / "first_edit_target_v1"
PRIVATE = prior.PRIVATE.parent / OUT.name
ARMS = ("cv", "frozen_moments", "refit_moments", "direct", "factorized")
STEPS = 500


def register():
    key, a, _, _, heads, head_key, source = prior.register()
    old = prior.OUT / "analysis.json"
    report = run.read(old)
    if report["registration"] != key or report["confirmation_access"] or not report["original_2pct"]:
        raise ValueError("Require the frozen cross-query result")
    reg = OUT / "registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parent=run.artifact(old), seeds=two.study.prior.SEEDS, native_steps=[8, 12], steps=STEPS,
        primary="direct whole-recording qualification vs refit moment readout from same head",
        control="CV and frozen moment head; every edited arm makes one edit in first eligible window",
        fitting="six donor sites; candidate sampling balanced by donor and recording; double-excluded easy cuts",
        factorization="P(easy) P(safe|easy) P(qualifies|safe), compared with direct target",
        features="past motion and candidate geometry only; recording-wide OLS offset is a label, never an input",
        threshold_search=False, workers=0, threads=4, original_2pct=True, confirmation_access=False)))
    return run.sha(reg), a, heads, head_key, source


def opportunities(d, packet):
    bank, _, _ = packet
    legal = bank[-1]
    q = d["query"]
    size = np.bincount(q)
    count = np.bincount(q, weights=legal.any(1))
    active = (size >= 2) & (count > 0.)
    for rec in np.unique(d["rec"]):
        at = np.flatnonzero(d["rec"] == rec)
        qs = np.unique(q[at])
        qs = qs[active[qs]]
        if len(qs):
            row, arm = np.nonzero(legal[at] & (q[at] == qs[0])[:, None])
            yield at, row, arm, int(len(qs)), np.flatnonzero(q == qs[0])


def target(d, packet, at, row, arm, cut):
    bank, paths, _ = packet
    cv, ols, _, _, scale, _ = bank
    local = at[row]
    y, valid = d["y"][at], d["valid"][at]
    p = paths[local, arm]
    g, risk, easy, offset = two.coefficients(cv[at], ols[at], y, valid, row, p, cut)
    mass = two.moments(cv[at], cv[at], y, valid, cut)["ref_easy"].sum()
    event = model.labels(g, risk, easy, offset, mass)
    z = np.c_[g/12., risk]/scale[local, None]
    return np.c_[z.clip(-8., 8.), event].astype(np.float32)


def training(d, packet, entries, cut):
    _, _, x = packet
    return [(torch.from_numpy(x[at[row], arm]), torch.from_numpy(target(d, packet, at, row, arm, cut)))
            for at, row, arm, _, _ in entries]


def batcher(sites):
    def sample(rng):
        src = rng.integers(len(sites), size=128)
        record = [sites[k][rng.integers(len(sites[k]))] for k in src]
        row = [rng.integers(len(x)) for x, _ in record]
        return tuple(torch.stack([v[j][i] for v, i in zip(record, row, strict=True)]) for j in range(2))
    return sample


def checked_oracle(d, packet, at, row, arm, cuts):
    bank, paths, _ = packet
    cv, ols = bank[:2]
    labels = target(d, packet, at, row, arm, cuts[0])[:, 5:].astype(bool)
    for j in (1, 2):
        hit = np.flatnonzero(labels[:, j])
        if not len(hit):
            continue
        k = hit[0]
        pred = cv[at].copy()
        pred[row[k]] = paths[at[row[k]], arm[k]]
        score = parent.score(dict(y=d["y"][at], valid=d["valid"][at], rec=np.full(len(at), "local")),
                             cv[at], ols[at], pred, cuts)
        if score[("risk_pass", "dual_positive")[j-1]] != 1 or score["edits"] != 1:
            raise ValueError("First-edit supervision failed original evaluator")
    return labels


@torch.inference_mode()
def evaluate(d, packet, entries, old_net, net, cuts):
    bank, paths, x = packet
    cv, ols, _, _, scale, legal = bank
    y, valid, q = d["y"], d["valid"], d["query"]
    size = np.bincount(q)
    predictions = {k: cv.copy() for k in ARMS}
    support, coverage = Counter(), {k: Counter() for k in ARMS[1:]}
    short_rows = np.zeros(len(cv), bool)
    choices_changed = Counter()
    for at, row, arm, n_queries, window in entries:
        local = at[row]
        values = x[local, arm]
        frozen, z = (m(torch.from_numpy(values)).numpy() for m in (old_net, net))
        unit = np.median(scale[window])
        pred = dict(frozen_moments=prior.local_score(frozen, scale[local], unit),
                    refit_moments=prior.local_score(z, scale[local], unit), **model.scores(z))
        take = {k: int(v.argmax()) for k, v in pred.items()}
        for k, j in take.items():
            predictions[k][local[j]] = paths[local[j], arm[j]]
        # Labels enter only after all policies have emitted their one edit.
        event = checked_oracle(d, packet, at, row, arm, cuts)
        n = len(event)
        short = n_queries < 5
        short_rows[at] = short
        support.update(records=1, short=int(short), candidates=n, positive_candidates=int(event[:, 2].sum()),
                       easy_possible=int(event[:, 0].any()), risk_possible=int(event[:, 1].any()),
                       qualifies_first_query=int(event[:, 2].any()), short_qualifies_first_query=int(short and event[:, 2].any()))
        allrow, allarm = np.nonzero(legal[at] & (size[q[at]] >= 2)[:, None])
        all_events = checked_oracle(d, packet, at, allrow, allarm, cuts)
        support.update(qualifies_any_query=int(all_events[:, 2].any()),
                       short_qualifies_any_query=int(short and all_events[:, 2].any()))
        for k, j in take.items():
            e, s, u = event[j]
            coverage[k].update(easy=int(e), safe=int(s), qualifies=int(u), short_qualifies=int(short and u))
            choices_changed[k] += int(j != take["refit_moments"])
    full, short = {}, {}
    for k, p in predictions.items():
        full[k] = parent.score(d, cv, ols, p, cuts)
        if k != "cv" and (full[k]["edits"], full[k]["risk_pass"], full[k]["dual_positive"]) != (
                len(entries), coverage[k]["safe"], coverage[k]["qualifies"]):
            raise ValueError("Whole-recording target and one-edit scorer differ")
        short[k] = (parent.score({v: d[v][short_rows] for v in ("rec", "y", "valid")}, cv[short_rows],
                                 ols[short_rows], p[short_rows], cuts) if short_rows.any() else None)
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    effects = {k: float(100.*paired(predictions["refit_moments"], predictions[k], y, valid)[1].sum()/mass)
               for k in ("direct", "factorized", "frozen_moments")}
    return dict(full=full, short=short, support=support, coverage=coverage, changed=choices_changed, effects=effects)


def execute(key, a, heads, head_key, source, replay, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    packets = {s: two.prepare(d) for s, d in sites.items()}
    entries = {s: list(opportunities(d, packets[s])) for s, d in sites.items()}
    result, frozen, updates = {}, {}, 0

    def cut(sites_in):
        return np.quantile(np.concatenate([sites[s]["cut_samples"] for s in sites_in]), [.25, .75])

    for held, d in sites.items():
        donors = [s for s in sites if s != held]
        data = [training(sites[s], packets[s], entries[s], cut([u for u in donors if u != s])[0]) for s in donors]
        fit_batch = batcher(data)
        result[held] = []
        with np.load(source.PRIVATE / f"{held}_cohort.npz", allow_pickle=False) as p:
            if len(np.unique(p["recording"])) != len(entries[held]):
                raise ValueError("One-edit roster differs from verified recordings")
        for cell in a["by_site"][held]:
            seed = cell["seed"]
            name = f"{held}_{seed}"
            info = heads[name]
            if info["identity"] != dict(registration=head_key, held=held, fit=donors, seed=seed):
                raise ValueError("Frozen head has incompatible source roles")
            width = packets[held][2].shape[-1]
            old = two.fit.load(info, lambda: two.model.network(width), two.STEPS)
            identity = dict(registration=key, held=held, fit=donors, seed=seed)
            net, frozen[name], u = two.fit.fit(identity, name, seed, lambda: model.network(width), STEPS,
                                              fit_batch, model.loss, replay, beat)
            updates += u
            r = evaluate(d, packets[held], entries[held], old, net, cut(donors))
            result[held].append(dict(seed=seed, **r))
            beat(phase="readout", site=held, seed=seed, support=r["support"])
    flat = [r for v in result.values() for r in v]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(k):
        x = np.array([np.mean([r["effects"][k] for r in v]) for v in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_location=float(x.min()))

    fields = ("edits", "views", "undefined", "risk_pass", "dual_positive", "unknown_edits")
    total = {k: {a: {f: sum(c[k][a][f] for c in flat if c[k][a] is not None) for f in fields} for a in ARMS}
             for k in ("full", "short")}
    total.update(support=sum((Counter(r["support"]) for r in flat), Counter()),
                 changed=sum((Counter(r["changed"]) for r in flat), Counter()),
                 effects_vs_refit_moments={k: ci(k) for k in ("direct", "factorized", "frozen_moments")})
    f = OUT / "head_freeze.json"
    run.once(f, run.stable(dict(registration=key, heads=frozen)))
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, heads=run.sha(f), summary=total, by_site=result,
        original_2pct=True, confirmation_access=False, submission_ready=False)))
    return updates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    two.fit.PRIVATE = PRIVATE
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
