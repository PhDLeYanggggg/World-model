"""Isolate edit timing under a fixed radius multiset and streaming budget."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_recipient_slot as prior
from src.evaluation import m3w_slot_timing_capacity as capacity
from src.evaluation.m3w_recording_switch_ceiling import moments
from src.world_model import m3w_slot_timing as model

base, parent, run, study, torch = prior.base, prior.parent, prior.run, prior.study, prior.torch
OUT, PRIVATE = prior.OUT.parent / "slot_timing_v1", prior.PRIVATE.parent / "slot_timing_v1"
METHODS = ("prefix", "learned")
POLICIES = ("anchor", *(f"{m}_{k}" for m in METHODS for k in ("now", "wait", "wait_match")))


def register():
    key, a, ph, h, _ = prior.register()
    p, f = (prior.OUT / f"{k}.json" for k in ("analysis", "head_freeze"))
    previous, heads = run.read(p), run.read(f)
    if (previous["registration"] != key or heads["registration"] != key or previous["heads"] != run.sha(f)
            or previous["confirmation_access"] or not previous["query_radii_exact"] or not previous["original_2pct"]):
        raise ValueError("Require the exact frozen recipient controls and original risk")
    reg = OUT / "registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        capacity=run.artifact(Path(capacity.__file__)), prior=run.artifact(p), heads=run.artifact(f),
        native_steps=[8, 12], oracle_windows=model.WINDOWS, oracle_label_aware_only=True, timeout=2.,
        matching="same recordings, number of original edit slots and each radius; timestamp index may change",
        streaming="original 0.2 credit per eligible query; cap 2; each slot costs 1",
        online="causal release by original count policy; greedy positive score now or until query age 8; expire if unspent",
        matched_online="for wait comparison only; replay slots that waited policy emits at original times and recipients",
        no_training=True, no_held_tuning=True, original_2pct=True, confirmation_access=False)))
    return run.sha(reg), a, ph, h, heads["heads"], previous


def local_plan(cv, ols, history, query, eligible, cap, initial, amplitude):
    native = model.shape.peak(ols-cv)
    release, radius = query[initial], amplitude*native[initial]
    rows, slots, amp = model.candidates(query, native, cap, release, radius, *model.WINDOWS["move8"])
    old = np.flatnonzero(rows == initial[slots])
    np.testing.assert_array_equal(slots[old], np.arange(len(initial)))
    amp[old] = amplitude
    c, paths, proxy = model.proposals(history[rows], amp)
    np.testing.assert_array_equal(cv[rows], c)
    np.testing.assert_allclose(model.shape.peak(paths-cv[rows]), radius[slots], atol=1e-10, rtol=1e-8)
    shift = query[rows]-release[slots]
    np.testing.assert_equal(capacity.respects_tokens(np.bincount(release, minlength=len(cap)), eligible), True)
    return rows, slots, amp, paths, proxy, old, release, shift


@torch.inference_mode()
def evaluate(key, name, d, bank, menu, count, rec, head, cuts, beat):
    cv, ols = bank[:2]
    y, valid, query = (d[k] for k in ("y", "valid", "query"))
    pred = {k: cv.copy() for k in POLICIES}
    oracle = {k: Counter() for k in model.WINDOWS}
    transitions = {k: Counter() for k in model.WINDOWS if k != "fixed"}
    reach, actions = Counter(), {k: Counter() for k in METHODS}
    for i, recid in enumerate(np.unique(rec)):
        at = np.flatnonzero(d["rec"] == recid)
        on = rec == recid
        qids = np.flatnonzero(on)
        q = query[at]-qids[0]
        np.testing.assert_array_equal(np.unique(query[at]), qids)
        cap = np.minimum(2, np.bincount(q)-1)
        eligible = menu["legal"][on, 1]
        use = np.arange(2) < count[on, None]
        initial = np.searchsorted(at, menu["row"][on][use])
        if not len(initial):
            easy = moments(cv[at], cv[at], y[at], valid[at], cuts[0])["easy"]
            legal = model.shape.peak(ols[at]-cv[at]) > 0.
            credit = np.cumsum(eligible)
            status = ("no_stream_quota" if credit[-1] < 5 else "easy_after_credit_reachable"
                      if (easy & legal & (credit[q] >= 5) & (cap[q] > 0)).any() else "no_credit_reachable_easy")
            reach[status] += 1
            results = dict.fromkeys(model.WINDOWS, "zero_edits")
        else:
            amp = np.array([.25, .5, 1.])[menu["arm"][on][use]]
            local = cv[at], ols[at], d["h"][at], q, eligible, cap
            rows, slots, a, paths, proxy, old, release, shift = local_plan(*local, initial, amp)
            file = PRIVATE / f"{name}_{i:04d}.json"
            if file.is_file():
                receipt = run.read(file)
                if receipt["identity"] != [key, name, i]:
                    raise ValueError("Timing capacity receipt moved")
                results = receipt["oracle"]
            else:
                results = {}
                for k, (lo, hi) in model.WINDOWS.items():
                    z = (shift >= lo) & (shift <= hi)
                    results[k] = capacity.assess(cv[at], ols[at], y[at], valid[at], q, eligible, cap,
                        rows[z], slots[z], len(initial), paths[z], cuts)
                run.once(file, run.stable(dict(identity=[key, name, i], oracle=results)))
            pred["anchor"][at[rows[old]]] = paths[old]
            forward = shift >= 0
            x, _, _, _, scale = prior.model.features(d["h"][at[rows[forward]]], a[forward])
            z = head(torch.from_numpy(x)).numpy()[:, :5]*scale[:, None]
            learned = z[:, :2].min(1)-z[:, 2:].clip(min=0.)@np.array([2., 4., 1.])
            for m, score in (("prefix", proxy[forward]), ("learned", learned)):
                er, es, ep = rows[forward], slots[forward], paths[forward]
                for label, age in (("now", 0), ("wait", 8)):
                    pick = model.schedule(q[er], er, es, score, release, eligible, cap, age)
                    np.testing.assert_equal(capacity.respects_tokens(np.bincount(q[er[pick]], minlength=len(cap)), eligible), True)
                    pred[f"{m}_{label}"][at[er[pick]]] = ep[pick]
                    actions[m][label] += len(pick)
                    if label == "wait":
                        match = old[es[pick]]
                        actions[m]["delayed"] += int((q[er[pick]] > release[es[pick]]).sum())
                        actions[m]["expired_or_end"] += len(old)-len(pick)
                        pred[f"{m}_wait_match"][at[rows[match]]] = paths[match]
        for k, status in results.items():
            oracle[k][status] += 1
            if k != "fixed":
                transitions[k][f'{results["fixed"]} -> {status}'] += 1
        beat(phase="recording", name=name, recording=i)
    metrics = {k: parent.score(d, cv, ols, p, cuts) for k, p in pred.items()}
    for m in METHODS:
        if metrics[f"{m}_wait"]["edits"] != metrics[f"{m}_wait_match"]["edits"]:
            raise ValueError("Emitted-slot match lost an edit")
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    comparisons = [(f"{m}_wait", ref) for m in METHODS for ref in ("anchor", f"{m}_wait_match", f"{m}_now")]
    direct = {f"{k}_vs_{ref}": float(100.*parent.transfer.paired(pred[ref], pred[k], y, valid)[1].sum()/mass)
              for k, ref in comparisons}
    return dict(metrics=metrics, direct=direct, oracle=oracle, zero_edit_reachability=reach, actions=actions,
                transitions=transitions)


def execute(key, a, ph, h, heads, previous, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    rows = {}
    for held, d in sites.items():
        donors = [s for s in sites if s != held]
        cuts = np.quantile(np.concatenate([sites[s]["cut_samples"] for s in donors]), [.25, .75])
        b = parent.prior.analytic(d)
        file = parent.prior.prior.old.source.prior.PRIVATE / f"{held}_cohort.npz"
        with np.load(file, allow_pickle=False) as z:
            rec, starts = z["recording"], z["start"]
        rows[held] = []
        width = prior.model.features(d["h"][:1], np.array([.25]))[0].shape[1]
        for cell in a["by_site"][held]:
            seed = cell["seed"]
            name = f"{held}_{seed}"
            meta = heads[name]
            head = study.load(meta, lambda: prior.model.Value(width), prior.STEPS)
            rk = study.load(ph["heads"][f"{held}/{seed}/additive_rank"], lambda: study.prior.model.SwapValue(False), study.prior.STEPS)
            m = study.menus(d, b, rk)
            x, _, _ = base.features(d, b, m)
            net = study.load(h["heads"][f"{held}_{seed}_motion"], lambda: base.model.Value(x["motion"].shape[1]), base.STEPS)
            with torch.inference_mode():
                p = net(torch.from_numpy(x["motion"])).numpy()
            count = study.model.streaming_count(base.model.score(p, risk=False), m["legal"], rec, starts)
            v = evaluate(key, name, d, b, m, count, rec, head, cuts, beat)
            old = next(r for r in previous["by_site"][held] if r["seed"] == seed)
            for k, x in v["metrics"]["anchor"].items():
                np.testing.assert_allclose(x, old["metrics"]["frozen_prefix"][k], atol=1e-10, rtol=0.)
            if dict(v["oracle"]["fixed"]) != old["oracle"]:
                raise ValueError("Fixed-timing oracle did not reproduce")
            rows[held].append(dict(seed=seed, **v))
            beat(phase="held", held=held, seed=seed, witnesses={k: z[capacity.POS] for k, z in v["oracle"].items()})
    flat = [r for v in rows.values() for r in v]
    draw = np.random.default_rng(817).integers(len(rows), size=(3000, len(rows)))

    def ci(fn):
        x = np.array([np.mean([fn(r) for r in v]) for v in rows.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())

    totals = ("edits", "views", "undefined", "risk_pass", "dual_positive", "unknown_edits")
    summary = dict(arms={k: dict(vs_cv=ci(lambda r: r["metrics"][k]["vs_cv"]),
        **{f: sum(r["metrics"][k][f] for r in flat) for f in totals}) for k in POLICIES},
        direct={k: ci(lambda r: r["direct"][k]) for k in flat[0]["direct"]},
        oracle={k: sum((Counter(r["oracle"][k]) for r in flat), Counter()) for k in model.WINDOWS},
        transitions={k: sum((Counter(r["transitions"][k]) for r in flat), Counter()) for k in flat[0]["transitions"]},
        zero_edit_reachability=sum((Counter(r["zero_edit_reachability"]) for r in flat), Counter()),
        actions={k: sum((Counter(r["actions"][k]) for r in flat), Counter()) for k in METHODS})
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, summary=summary,
        by_site=rows, original_2pct=True, confirmation_access=False, new_training=False)))


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            row = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", row)
            if v["phase"] != "recording":
                print(row, flush=True)

        execute(*register(), beat)
        beat(phase="complete")


if __name__ == "__main__":
    main()
