"""Measure matched, low-radius support-exchange capacity without held tuning."""

import argparse
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
from scripts import run_m3w_incumbent_direction as parent
from src.evaluation import m3w_small_support_exchange as model

run, h, torch, readout = parent.run, parent.h, parent.torch, parent.readout
OUT = parent.OUT.parent / "small_support_exchange_v1"
PRIVATE = parent.PRIVATE.parent / OUT.name
MODES = {"risk_one": (True, False), "joint_one": (True, True),
         "risk_query": (False, False), "joint_query": (False, True)}
SECONDS = 3.


def audit(m, rec, ix, groups, initial, y, valid, cut):
    cv, bank = m.floor[ix], m.bank[ix]
    ols, y, valid = bank[:, run.NAMES.index(run.parent.OLS)], y[ix], valid[ix]
    p = model.menu(cv, bank, ols, initial, groups, y, valid, cut["easy"])
    paths = {run.parent.CV: cv, "ols": ols, "incumbent": p["original"]}
    choices, outcomes = {}, {}
    for name, (single, joint) in MODES.items():
        take, status = model.solve(p, single=single, joint=joint, seconds=SECONDS)
        outcomes[name] = dict(status=status)
        if take is not None:
            paths[name], new = model.paths(p, take, initial, cv, bank)
            for q in groups:
                old, after = initial[q] >= 0, new[q] >= 0
                if old.sum() != after.sum() or (old & ~after).sum() > 1:
                    raise ValueError("Changes to per-query count or exchange cardinality")
            choices[name] = take
    pairs = {f"{a}_{b}": (b, a, run.parent.CV)
             for a in ["incumbent", *choices] for b in (run.parent.CV, "ols", "incumbent")}
    z = readout.score_recording(paths, pairs, y, valid, m.d["queries"][ix], rec,
                                easy_cut=cut["easy"], hard_cut=cut["hard"])
    before = z["metrics"]["incumbent"]["risk"]
    strong_cmp = readout.comparison_masses(ols, cv, cv, y, valid, p["weights"])
    strong_ols = strong_cmp["known_delta"]+strong_cmp["unknown_radius"] < 0.
    for name, take in choices.items():
        risk = z["metrics"][name]["risk"]
        ms = [z["comparisons"][f"{name}_{b}"] for b in ("incumbent", run.parent.CV, "ols")]
        lower = np.array([a["known_delta"]-a["unknown_radius"] for a in ms])
        expected = np.r_[0., p["old_lower"]] + p["terms"][take, 4:].sum(0)
        np.testing.assert_allclose(lower, expected, atol=1e-6, rtol=1e-9)
        threshold = max(1., p["reference"])*model.POS
        verified = risk["supported"] and (not MODES[name][1] or np.all(lower >= threshold-1e-8))
        if not verified:
            outcomes[name] = dict(status="postcheck_unresolved")
            continue
        outcomes[name].update(exchanges=len(take), lower=lower.tolist(),
                              observed=ms[0]["known_delta"], disagreement=ms[0]["unknown_radius"],
                              strictly_joint=bool(np.all(lower >= threshold)),
                              risk_rescue=not before["supported"])
    return dict(edits=before["selected"], undefined=before["completion_easy_risk_upper"] is None,
                support_count=p["support"], full_easy=p["full_easy"], legal_options=len(p["choice"]),
                supported=before["supported"], strong_ols=bool(strong_ols), reference=p["reference"],
                outcomes=outcomes)


def receipt(reg):
    paths = [parent.OUT/f"{k}.json" for k in ("registration", "analysis", "head_freeze")]
    previous = run.read(paths[1])
    if (previous["registration"] != run.sha(paths[0]) or previous["heads"] != run.sha(paths[2])
            or not previous["original_2pct"] or previous["confirmation_access"]):
        raise ValueError("Require the verified fixed-recipient experiment")
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(x) for x in paths], original_2pct=True, confirmation_access=False,
        no_new_training=True, outcome_aware_only=True, exact_query_edit_count=True, modes=MODES,
        eligibility="same query; incoming labeled easy row; positive radius no larger than removed radius",
        joint="original risk and strictly positive paired lower gain versus each of incumbent, CV, OLS4",
        margin=model.POS, solver_seconds=SECONDS, bootstrap=dict(draws=3000, seed=73107))))
    return previous


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay", action="store_true", help="Recompute frozen source cells without reusing scratch")
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic()
        reg = OUT/"registration.json"
        prev = receipt(reg)
        key = run.sha(reg)
        _, _, weights = parent.study.register()
        d, refs = run.population(), run.register()
        lab = run.old.prior.arrays(("target_eval", "valid", "frames"))
        records = h.previous.prior.blocks(d, lab["frames"])
        y, valid = lab["target_eval"]-d["origin"][:, None], lab["valid"]
        cuts = run.cuts_by_source()
        parents, controls, ranks = h.cost.banks()
        trainer = parent.study.old.study.trainer
        rows = {s: [] for s in prev["by_site"]}
        for cell in parents:
            m = h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
            for s in rows:
                name = f"{m.seed}_{s}"
                scratch = PRIVATE/f"{name}.json"
                if scratch.exists() and not args.replay:
                    z = run.read(scratch)
                    if z["registration"] != key:
                        raise ValueError("Changed source audit definition")
                    records_out = z["records"]
                else:
                    frozen = trainer.load(weights[name], lambda: parent.study.old.model.network(m.x.shape[-1]),
                                          parent.study.old.STEPS)
                    records_out = []
                    for rec, (ix, groups) in records.items():
                        if str(d["sites"][ix[0]]) == s:
                            a, *_ = parent.menu(m, ix, groups, frozen)
                            records_out.append(audit(m, rec, ix, groups, a, y, valid, cuts[s]))
                    run.once(scratch, run.stable(dict(registration=key, records=records_out)))
                ref = next(r["metrics"]["incumbent"] for r in prev["by_site"][s] if r["seed"] == m.seed)
                if (sum(r["edits"] for r in records_out), sum(r["undefined"] for r in records_out),
                    sum(r["supported"] for r in records_out)) != (ref["edits"], ref["undefined"], ref["risk_pass"]):
                    raise ValueError("Incumbent readout changed")
                rows[s].append(dict(seed=m.seed, records=records_out))
                run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", dict(pid=os.getpid(),
                    cells=sum(map(len, rows.values())), expected=42, complete=False, seconds=time.monotonic()-start))
            print(f"Verified support exchange witnesses for seed {m.seed}", flush=True)
            del m
        records_all = [r for rs in rows.values() for z in rs for r in z["records"]]
        if len(records_all) != 102:
            raise ValueError("Require all 102 development views")
        from collections import Counter
        counts = dict(views=102, undefined=sum(r["undefined"] for r in records_all),
            structural=sum(not r["full_easy"] for r in records_all),
            undefined_with_options=sum(r["undefined"] and r["legal_options"] > 0 for r in records_all),
            options=sum(r["legal_options"] for r in records_all), strong_ols=sum(r["strong_ols"] for r in records_all))
        for k in MODES:
            take = [(r, r["outcomes"][k]) for r in records_all]
            counts[k] = dict(status=Counter(o["status"] for _, o in take),
                witnesses=sum("exchanges" in o for _, o in take),
                exchanged=sum(o.get("exchanges", 0) for _, o in take),
                undefined_rescues=sum(r["undefined"] and o.get("risk_rescue", False) for r, o in take),
                risk_rescues=sum(o.get("risk_rescue", False) for _, o in take),
                strictly_joint=sum(o.get("strictly_joint", False) for _, o in take),
                strong_joint=sum(r["strong_ols"] and o.get("strictly_joint", False) for r, o in take))
        draw = np.random.default_rng(73107).integers(len(rows), size=(3000, len(rows)))
        intervals = {}
        for k in MODES:
            per_source = []
            for ss in rows.values():
                values = []
                for z in ss:
                    rr = z["records"]
                    # Oracle witnesses only; other recordings retain the incumbent.
                    gain = sum(r["outcomes"][k].get("lower", [0.])[0] for r in rr)
                    values.append(100*gain/sum(r["reference"] for r in rr))
                per_source.append(np.mean(values))
            v = np.asarray(per_source)
            intervals[k] = dict(point=float(v.mean()), interval95=np.quantile(v[draw].mean(1), [.025, .975]).tolist())
        run.once(OUT/"analysis.json", run.stable(dict(registration=key, counts=counts, intervals=intervals,
            by_site=rows, original_2pct=True, confirmation_access=False, no_new_training=True)))
        run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", dict(pid=os.getpid(), cells=42, complete=True,
            replay=args.replay, seconds=time.monotonic()-start))
        print(run.stable(dict(counts=counts, intervals=intervals)), flush=True)


if __name__ == "__main__":
    main()
