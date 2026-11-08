"""Test first-credit access on the frozen OpenDD controller and original gate."""

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
from scripts import run_m3w_ledger_wait as ledger
from src.evaluation.m3w_recording_switch_ceiling import moments
from src.world_model import m3w_first_credit as model

run, parent, study = ledger.run, ledger.parent, ledger.study
paired = ledger.prior.paired
OUT = ledger.OUT.parent / "first_credit_v2"
PRIVATE = ledger.PRIVATE.parent / OUT.name
ARMS = ("anchor", "advance", "advance_prefix")


def register():
    key, a, ph, h, *_ = ledger.register()
    path = ledger.OUT / "analysis.json"
    prior = run.read(path)
    if prior["registration"] != key or prior["confirmation_access"] or not prior["original_2pct"]:
        raise ValueError("Require the frozen OpenDD timing anchor")
    reg = OUT / "registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        anchor=run.artifact(path), anchor_binding=key, arms=ARMS,
        previous_aborted=run.artifact(OUT.parent / "first_credit_v1" / "registration.json"),
        correction="emit the original frozen prefix-selected shapes, not the analytic straight nomination paths",
        primary="first-credit advance versus exact frozen controller; both use identical menu and scores",
        credit="one edit can use credit before accrued; repay before further spending; at most one unpaid terminal edit",
        prefix_gate="require positive existing past-prefix proxy only for the early edit",
        oracle="exact at most one nonzero edit in original zero-credit recordings, fixed nominee versus all legal rows/amplitudes",
        native_steps=[8, 12], seeds=study.prior.SEEDS, bootstrap_draws=3000, bootstrap_seed=817,
        original_2pct=True, confirmation_access=False, no_held_tuning=True, new_training=False)))
    return run.sha(reg), a, ph, h, prior


def one_edit(cv, ols, y, valid, candidate_rows, paths, cuts):
    """Exhaustively score the one-edit menu without emitting hindsight choices."""
    n = len(candidate_rows)
    if n == 0:
        return dict(candidates=0, easy=0, risk=0, cv=0, both=0)
    i = candidate_rows
    m = moments(cv[i], paths, y[i], valid[i], cuts[0])
    cv_gain = paired(cv[i], paths, y[i], valid[i])[1].sum(1)
    ols_offset = paired(ols, cv, y, valid)[1].sum(1)
    ols_gain = ols_offset.sum()-ols_offset[i]+paired(ols[i], paths, y[i], valid[i])[1].sum(1)
    full_easy = moments(cv, cv, y, valid, cuts[0])["ref_easy"].sum()
    # A single edited row has to supply both strictly positive references.
    easy = (m["radius"] > 0.) & (m["ref_easy"] > 0.)
    safe = easy & (m["all_excess"] <= 0.) & (m["easy_excess"] <= 0.) & (m["easy_signed"] <= .02*full_easy)
    wins = safe & (cv_gain > 0.) & (ols_gain > 0.)
    if wins.any():
        k = np.flatnonzero(wins)[0]
        p = cv.copy()
        p[i[k]] = paths[k]
        s = parent.score(dict(y=y, valid=valid, rec=np.full(len(cv), "recording")), cv, ols, p, cuts)
        if s["dual_positive"] != 1:
            raise ValueError("One-edit witness did not pass the original evaluator")
    return dict(candidates=n, easy=int(easy.sum()), risk=int(safe.sum()),
                cv=int((safe & (cv_gain > 0.)).sum()), both=int(wins.sum()))


def emission_bank(d, bank):
    cv, _, straight, _, _, legal = bank
    shaped = straight.copy()
    for arm, amp in enumerate((.25, .5, 1.)):
        at = np.flatnonzero(legal[:, arm])
        for part in np.array_split(at, max(1, (len(at)+511)//512)):
            if len(part):
                baseline, path, _ = ledger.timing.model.proposals(d["h"][part], np.full(len(part), amp))
                np.testing.assert_array_equal(baseline, cv[part])
                shaped[part, arm] = path
    return shaped


@torch.inference_mode()
def evaluate(d, bank, paths, rec, starts, rank_meta, count_meta, cuts):
    left, right = rank_meta["identity"], count_meta["identity"]
    if any(left[k] != right[k] for k in ("held", "seed", "donors")) or left["held"] in left["donors"]:
        raise ValueError("Incompatible or exposed nomination/count fitting roster")
    cv, ols, _, _, _, legal_bank = bank
    rank = study.load(rank_meta, lambda: study.prior.model.SwapValue(False), study.prior.STEPS)
    menu = study.menus(d, bank, rank)
    x, _, proxy = ledger.base.features(d, bank, menu)
    net = study.load(count_meta, lambda: ledger.base.model.Value(x["motion"].shape[1]), ledger.base.STEPS)
    scores = ledger.base.model.score(net(torch.from_numpy(x["motion"])).numpy(), risk=False)
    legal = menu["legal"]
    q = d["query"]
    y, valid = d["y"], d["valid"]
    active = legal[:, 1]
    counts = dict(anchor=study.model.streaming_count(scores, legal, rec, starts),
                  advance=model.decode(scores, legal, rec, starts, early_allowed=np.ones(len(rec), bool)),
                  advance_prefix=model.decode(scores, legal, rec, starts, early_allowed=proxy[:, 1] > 0.))
    np.testing.assert_array_equal(counts["anchor"], model.decode(scores, legal, rec, starts, early_allowed=np.zeros(len(rec), bool)))
    outputs, events, short_metrics = {}, {}, {}
    groups, inv = np.unique(rec, return_inverse=True)
    opportunities = np.bincount(inv, weights=active)
    short = opportunities < 5
    short_rows = short[inv[q]]
    for name, count in counts.items():
        take = np.arange(2) < count[:, None]
        choice = np.full(len(cv), -1, dtype=int)
        choice[menu["row"][take]] = menu["arm"][take]
        np.testing.assert_array_equal(np.bincount(q[choice >= 0], minlength=len(rec)), count)
        outputs[name] = parent.selected(cv, paths, choice)
        total = np.bincount(inv, weights=count)
        if (total > np.maximum(1, opportunities//5)).any() or (total[opportunities == 0] > 0).any():
            raise ValueError("First credit breached its terminal bound")
        events[name] = dict(short_edits=int(total[short].sum()), long_edits=int(total[~short].sum()))
        short_metrics[name] = parent.score({k: d[k][short_rows] for k in ("rec", "y", "valid")},
                                            cv[short_rows], ols[short_rows], outputs[name][short_rows], cuts) if short.any() else None
    oracle = {k: Counter() for k in ("nominee", "all_legal")}
    for recording in groups[short]:
        at = np.flatnonzero(d["rec"] == recording)
        qs = np.flatnonzero((rec == recording) & active)
        local = np.searchsorted(at, menu["row"][qs, 0])
        cases = {"nominee": (local, paths[menu["row"][qs, 0], menu["arm"][qs, 0]])}
        i, arm = np.nonzero(legal_bank[at] & active[q[at], None])
        cases["all_legal"] = i, paths[at[i], arm]
        for k, (i, p) in cases.items():
            v = one_edit(cv[at], ols[at], y[at], valid[at], i, p, cuts)
            oracle[k].update(views=1, **{f"views_with_{key}": int(value > 0) for key, value in v.items()})
    metrics = {k: parent.score(d, cv, ols, v, cuts) for k, v in outputs.items()}
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    direct = {k: float(100.*paired(outputs["anchor"], outputs[k], y, valid)[1].sum()/mass) for k in ARMS[1:]}
    return dict(metrics=metrics, direct=direct, events=events, short_metrics=short_metrics, one_edit_oracle=oracle)


def execute(key, a, ph, h, old, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    by_site = {}
    for s, d in sites.items():
        bank = parent.prior.analytic(d)
        paths = emission_bank(d, bank)
        cuts = np.quantile(np.concatenate([v["cut_samples"] for k, v in sites.items() if k != s]), [.25, .75])
        f = parent.prior.prior.old.source.prior.PRIVATE / f"{s}_cohort.npz"
        with np.load(f, allow_pickle=False) as z:
            rec, starts = z["recording"], z["start"]
        by_site[s] = []
        for cell in a["by_site"][s]:
            seed = cell["seed"]
            p = PRIVATE / f"{s}_{seed}.json"
            identity = [key, s, seed]
            fresh = not p.exists()
            if fresh:
                ref = next(r for r in old["by_site"][s] if r["seed"] == seed)["metrics"]["anchor"]
                rank_meta = ph["heads"][f"{s}/{seed}/additive_rank"]
                count_meta = h["heads"][f"{s}_{seed}_motion"]
                value = evaluate(d, bank, paths, rec, starts, rank_meta, count_meta, cuts)
                for k, y in value["metrics"]["anchor"].items():
                    np.testing.assert_allclose(y, ref[k], rtol=0., atol=1e-10)
                run.once(p, run.stable(dict(identity=identity, value=value)))
            entry = run.read(p)
            if entry["identity"] != identity:
                raise ValueError("Cached first-credit result does not match registration")
            by_site[s].append(dict(seed=seed, **entry["value"]))
            beat(phase="readout", site=s, seed=seed, fresh=fresh)
    flat = [r for v in by_site.values() for r in v]
    fields = ("edits", "views", "undefined", "risk_pass", "dual_positive", "unknown_edits")
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))
    ci = {}
    for arm in ARMS[1:]:
        x = np.array([np.mean([r["direct"][arm] for r in v]) for v in by_site.values()])
        ci[arm] = dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                       worst_location=float(x.min()))
    summary = dict(arms={a: {f: sum(r["metrics"][a][f] for r in flat) for f in fields} for a in ARMS},
        short={a: {f: sum(r["short_metrics"][a][f] for r in flat if r["short_metrics"][a] is not None)
                   for f in fields} for a in ARMS},
        one_edit_oracle={a: sum((Counter(r["one_edit_oracle"][a]) for r in flat), Counter())
                         for a in ("nominee", "all_legal")},
        advance_minus_anchor_pp=ci)
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, by_site=by_site, summary=summary,
        original_2pct=True, confirmation_access=False, no_training=True, submission_ready=False)))


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**values):
            row = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **values)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", row)
            print(row, flush=True)

        execute(*register(), beat)
        beat(phase="complete")


if __name__ == "__main__":
    main()
