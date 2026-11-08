"""Test hindsight capacity of optional recipients and a stronger reference bridge."""

import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import sys
import time
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_hannover_frozen_transfer as previous
from src.evaluation.m3w_optional_dual_capacity import coefficients, solve
from src.evaluation.m3w_selection_forecast_readout import policy

run, np, torch, transfer, source = previous.run, previous.np, previous.torch, previous.transfer, previous.source
OUT = previous.OUT.parent / "optional_reference_bridge_v1"
PRIVATE = previous.PRIVATE.parent / OUT.name
MIX = (.25, .5, 1.)
MENUS = ("frozen_optional", "ols_bridge_only", "union")
POSITIVE_PP = 1e-6


def register():
    parent, frozen, _, intake, scale, _ = previous.register()
    parent_path = previous.OUT / "analysis.json"
    a = run.read(parent_path)
    if a["registration"] != parent or a["confirmation_access"] or a["formal_confirmation"]:
        raise ValueError("Require the frozen Hannover negative result")
    f = OUT / "registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), solver=run.artifact(Path(solve.__code__.co_filename)),
        parent=run.artifact(parent_path), data="exposed OpenDD and already scored Hannover only",
        objective="maximize minimum full-grid lower utility versus CV and OLS4 under original CV risk",
        primary="optional neural or neural plus OLS4 bridges", radius="half realized original neural peak",
        bridge_cv_to_ols_fractions=MIX, optional_recipients_from_same_eligible_universe=True, at_most_one_action=True,
        no_redefinition_of_empty_or_unsupported=True, allowed_seconds=2., positive_tolerance_pp=POSITIVE_PP,
        seeds=previous.choice_study.SEEDS, no_training=True, labels_hindsight_only=True, confirmation_access=False)))
    return run.sha(f), frozen, intake, scale


def menus(cv, ols, bank, eligible):
    # OLS4 is a candidate, not a replacement of the original CV risk floor.
    b = cv[:, None]+(ols-cv)[:, None]*np.asarray(MIX)[None, :, None, None]
    b[:, -1] = ols
    b[~eligible] = cv[~eligible, None]
    return dict(frozen_optional=bank, ols_bridge_only=b, union=np.concatenate((bank, b), axis=1))


def assess(cv, ols, bank, y, valid, name, cuts):
    m = coefficients(cv, ols, bank, y, valid, cuts[0])
    take, solver = solve(m)
    s = dict(solver, classification=solver["status"], incumbent=None)
    if take is None:
        return s
    row, arm = m["row"][take], m["arm"][take]
    if len(np.unique(row)) != len(row):
        raise ValueError("Solver chose more than one action per row")
    p = cv.copy()
    p[row] = bank[row, arm]
    np.testing.assert_array_equal(np.flatnonzero((p != cv).any((1, 2))), row)
    risk = policy(p, cv, cv, y, valid, np.full(len(y), name), easy_cut=cuts[0], hard_cut=cuts[1])["recordings"][name]
    gain = 100*np.array([transfer.paired(ref, p, y, valid)[1].sum() for ref in (cv, ols)])/m["cv_mass"]
    np.testing.assert_allclose(gain, 100*(m["offset"]+m["gain"][:, take].sum(1))/m["cv_mass"], atol=1e-10, rtol=1e-8)
    q = float(gain.min())
    if solver["upper_pp"] is not None and q > solver["upper_pp"]+1e-5:
        raise ValueError("Verified incumbent exceeds its upper bound")
    s["incumbent"] = dict(gains_pp=gain.tolist(), risk=risk, edits=len(row), action_counts=np.bincount(arm, minlength=bank.shape[1]).tolist())
    if risk["supported"] and q > POSITIVE_PP:
        s["classification"] = "verified_positive_both"
    elif solver["upper_pp"] is not None and solver["upper_pp"] <= POSITIVE_PP:
        s["classification"] = "no_strict_positive_bound"
    else:
        s["classification"] = "unresolved"
    return s


def source_data(intake, scale):
    d = {"opendd": {}, "hannover": {}}
    for s in source.SITES:
        with np.load(source.prior.PRIVATE / f"{s}_cohort.npz", allow_pickle=False) as z:
            h, y, rec = z["history"], z["future"], np.repeat(z["recording"], np.diff(z["offsets"]))
        d["opendd"][s] = h, y, rec
    with ZipFile(previous.ZIP) as z:
        for s in previous.CLIPS:
            raw = z.read(previous.PREFIX+f"{s}_trajectory.csv")
            if hashlib.sha256(raw).hexdigest() != intake["clips"][s]["sha256"]:
                raise ValueError("Registered clip hash changed")
            h, y, _ = previous.windows(previous.parse_csv(raw)[0], scale)
            d["hannover"][s] = h, y, np.full(len(h), s)
    return d


def summarize(groups):
    records = [rec for cells in groups.values() for c in cells for rec in c["recordings"].values()]
    names = ("absent_easy", "no_legal_easy", "risk_infeasible", "no_strict_positive_bound", "verified_positive_both", "unresolved")
    out = {}
    for menu in MENUS:
        r = [v[menu] for v in records]
        positive = [v["incumbent"] for v in r if v["classification"] == "verified_positive_both"]
        counts = {k: sum(v["classification"] == k for v in r) for k in names}
        if sum(counts.values()) != len(records):
            raise ValueError("Unclassified recording/seed view")
        out[menu] = dict(counts=counts, views=len(records),
            verified_supported_incumbents=sum(v["incumbent"] is not None and v["incumbent"]["risk"]["supported"] for v in r),
            mean_edits_positive=float(np.mean([v["edits"] for v in positive])) if positive else None)
    union = out["union"]["counts"]
    if union["verified_positive_both"]+union["unresolved"] < max(out[k]["counts"]["verified_positive_both"] for k in MENUS[:2]):
        raise ValueError("Larger menu lost feasible witness count")
    return out


def execute(key, frozen, intake, scale, beat, cache):
    nets = transfer.frozen_models(*frozen[:5])
    data = source_data(intake, scale)
    outputs = {}
    cuts = {}
    for s, (h, y, _) in data["opendd"].items():
        cv = transfer.motion.rollout(h, 12)[0]
        complete = np.isfinite(y).all((1, 2))
        a = np.linalg.norm(cv[complete]-y[complete], axis=-1).mean(1)
        cuts[s] = a[a > 0]
    for d, groups in data.items():
        outputs[d] = {}
        for j, (s, (h, y, rec)) in enumerate(groups.items()):
            outputs[d][s] = []
            valid = np.isfinite(y).all(2)
            cut = np.quantile(np.concatenate([v for name, v in cuts.items() if name != s]), [.25, .75])
            for seed in previous.choice_study.SEEDS:
                p = cache / f"{d}_{s}_{seed}.json"
                if not p.exists():
                    model = [nets[seed][j]] if d == "opendd" else nets[seed]
                    cv, ols, inc, _, delta = transfer.forecast(h, model)
                    paths, on = transfer.matched_paths(cv, inc, delta, .5)
                    bank = np.stack([paths[k] for k in transfer.ARMS], axis=1)
                    arms = menus(cv, ols, bank, on)
                    results = {}
                    for name in sorted(set(rec)):
                        m = rec == name
                        results[name] = {k: assess(cv[m], ols[m], b[m], y[m], valid[m], name, cut) for k, b in arms.items()}
                    run.once(p, run.stable(dict(registration=key, source=d, site=s, seed=seed, recordings=results)))
                a = run.read(p)
                if (a["registration"], a["source"], a["site"], a["seed"]) != (key, d, s, seed):
                    raise ValueError("Wrong source or solver checkpoint")
                outputs[d][s].append(a)
                beat(domain=d, site=s, seed=seed, recordings=len(a["recordings"]))
    summary = {d: summarize(groups) for d, groups in outputs.items()}
    if any(summary[d]["union"]["views"] != n for d, n in (("opendd", 606), ("hannover", 9))):
        raise ValueError("Incomplete required recording/seed roster")
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, summary=summary, groups=outputs,
        no_training=True, original_2pct=True, official_split_changed=False, confirmation_access=False, deployment=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**state):
            a = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **state)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", a)
            print(a, flush=True)

        reg = register()
        if args.replay:
            from tempfile import TemporaryDirectory
            with TemporaryDirectory(dir=PRIVATE) as t:
                execute(*reg, beat, Path(t))
        else:
            execute(*reg, beat, PRIVATE)
        beat(phase="complete", replay=args.replay)


if __name__ == "__main__":
    main()
