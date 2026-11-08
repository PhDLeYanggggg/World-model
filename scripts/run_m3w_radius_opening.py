"""Compare learning when to open at an identical one-edit radius budget."""

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
from scripts import audit_m3w_small_support_exchange as parent
from src.evaluation.m3w_recording_switch_ceiling import moments
from src.world_model import m3w_radius_opening as model

run, h, torch, readout = parent.run, parent.h, parent.torch, parent.readout
OUT = parent.OUT.parent/"radius_opening_v1"
PRIVATE = parent.PRIVATE.parent/OUT.name
ARMS = ("incumbent", "frozen_first", "learned_first", "always_wait", "learned_stop", "first_count_matched")


def labeling(a, original, cv, ols, y, valid, selected, w, cut):
    base = moments(cv, original, y, valid, cut)
    full_easy = base["ref_easy"].sum()
    old_ols = readout.comparison_masses(ols, original, cv, y, valid, w)
    old_cv = readout.comparison_masses(cv, original, cv, y, valid, w)
    old_lower = lambda z: z["known_delta"]-z["unknown_radius"]
    risk = np.array([base[k][selected].sum() for k in ("all_excess", "easy_excess", "easy_signed")])
    easy_mass = base["ref_easy"][selected].sum()
    old_safe = easy_mass > 0. and np.all(risk[:2] <= 0.) and risk[2] <= .02*full_easy
    for r in a:
        i, path = r["row"], r["path"]
        z = moments(cv[i], path, y[i], valid[i], cut)
        u = w[i]*z["utility"]
        old_diff = np.linalg.norm(cv[i]-ols[i], axis=-1).max(1)
        new_diff = np.linalg.norm(path-ols[i], axis=-1).max(1)
        ols_delta = w[i]*np.where(z["known"], z["diff"], old_diff-new_diff)
        support = z["ref_easy"]+easy_mass > 0.
        support &= (z["all_excess"]+risk[0] <= 0.) & (z["easy_excess"]+risk[1] <= 0.)
        support &= z["easy_signed"]+risk[2] <= .02*full_easy
        joint = support & (u > 0.) & (old_lower(old_cv)+u > 0.) & (old_lower(old_ols)+ols_delta > 0.)
        r.update(joint=joint, support=support,
                 quality=2.*joint+support-float(old_safe)+.25*z["utility"]/r["radius"])


@torch.inference_mode()
def prepare(m, ix, groups, frozen):
    cv, bank = m.floor[ix], m.bank[ix]
    initial, *_ = parent.parent.menu(m, ix, groups, frozen)
    ols = bank[:, run.NAMES.index(run.parent.OLS)]
    original = run.model.selected_paths(initial, bank, cv)
    weights = 1./m.d["crowd"][ix]
    offers = []
    for row, _, path, radius in model.offers(cv, bank, groups, initial):
        x, s = run.model.descriptors(m.d["history"][ix[row]], path[:, None], cv[row], m.d["crowd"][ix[row]])
        x = x[:, 0]
        score = frozen(torch.from_numpy(x.astype(np.float32))).numpy()[:, 0]*s
        offers.append(dict(x=x, row=row, path=path, radius=radius, frozen=int(score.argmax())))
    return dict(initial=initial, cv=cv, ols=ols, original=original, offers=offers, weights=weights)


def add_labels(p, y, valid, cut):
    labeling(p["offers"], p["original"], p["cv"], p["ols"], y, valid,
             p["initial"] >= 0, p["weights"], cut["easy"])
    return p["offers"]


def score(p, coeff, wait, target, valid, q, rec, cut):
    offers = p["offers"]
    first, stop, delayed = model.choose(offers, coeff, wait)
    op = dict(incumbent=None, frozen_first=(0, offers[0]["frozen"]) if offers else None,
              learned_first=first, always_wait=(1, model.rank(offers[1], coeff)) if len(offers) > 1 else None,
              learned_stop=stop, first_count_matched=first if stop is not None else None)
    paths = {run.parent.CV: p["cv"], "ols": p["ols"]}
    for name, action in op.items():
        pred = p["original"].astype(float)
        if action is not None:
            j, i = action
            z = offers[j]
            row = z["row"][i]
            pred[row] = z["path"][i]
            np.testing.assert_allclose(np.linalg.norm(pred[row]-p["cv"][row], axis=-1).max(), z["radius"],
                                       rtol=1e-6, atol=1e-7)
        paths[name] = pred
    pairs = {f"{a}_{r}": (r, a, run.parent.CV) for a in ARMS for r in (run.parent.CV, "ols", "incumbent")}
    pairs["stop_vs_first_matched"] = ("first_count_matched", "learned_stop", run.parent.CV)
    pairs["stop_vs_first_full"] = ("learned_first", "learned_stop", run.parent.CV)
    pairs["first_vs_frozen"] = ("frozen_first", "learned_first", run.parent.CV)
    z = readout.score_recording(paths, pairs, target, valid, q, rec, easy_cut=cut["easy"], hard_cut=cut["hard"])
    add_labels(p, target, valid, cut)
    counts = {}
    for a, action in op.items():
        risk = z["metrics"][a]["risk"]
        def gain(r):
            a0 = z["comparisons"][f"{a}_{r}"]
            return a0["known_delta"]-a0["unknown_radius"]
        joint = risk["supported"] and all(gain(r) > 0. for r in ("incumbent", "ols", run.parent.CV))
        if action is not None and joint != offers[action[0]]["joint"][action[1]]:
            raise ValueError("Action label must match original scoring")
        counts[a] = dict(views=1, edits=risk["selected"], opened=int(action is not None),
                         late=int(action is not None and action[0] == 1),
                         undefined=int(risk["completion_easy_risk_upper"] is None),
                         risk=int(risk["supported"]), joint=int(joint),
                         zero_inc_opened=int(action is not None and not (p["initial"] >= 0).any()))
    oracle = dict(views=1, no_incumbent=int(not (p["initial"] >= 0).any()), first=int(bool(offers)),
                  second=int(len(offers) > 1), expired=int(delayed and stop is None),
                  first_joint=int(bool(offers) and offers[0]["joint"].any()),
                  two_joint=int(any(r["joint"].any() for r in offers)),
                  top_one_either=int(any(r["joint"][model.rank(r, coeff)] for r in offers)))
    return counts, z["comparisons"], oracle


def execute(replay, beat):
    a, reg = (parent.OUT/f"{k}.json" for k in ("analysis", "registration"))
    if run.read(a)["registration"] != run.sha(reg) or run.read(a)["confirmation_access"]:
        raise ValueError("Require frozen exchange-capacity study")
    key, _, weights = parent.parent.study.register()
    freeze = OUT/"registration.json"
    run.once(freeze, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(x) for x in (a, reg, parent.parent.study.OUT/"head_freeze.json")],
        original_2pct=True, confirmation_access=False, maximum_openings=1, maximum_offers=2,
        radius="quarter median positive action radius at first eligible query; same at next",
        fit="7 fitting sources, 6 reserved, 1 held; leave-one-donor-out ranking for timing labels",
        decision="marginal whole-recording quality target; learned wait is positive predicted continuation advantage",
        utility_control="earliest choice on same recordings where learned opens; offline matching only",
        no_held_tuning=True, cap_matches_not_a_deployment_claim=True, seeds=run.kin.prior.SEEDS)))
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(("target_eval", "valid", "frames"))
    records = h.previous.prior.blocks(d, lab["frames"])
    y, valid = lab["target_eval"]-d["origin"][:, None], lab["valid"]
    cuts = run.cuts_by_source()
    parents, controls, ranks = h.cost.banks()
    trainer = parent.parent.study.old.study.trainer
    rows = {s: [] for s in run.read(a)["by_site"]}
    for cell in parents:
        m = h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        for c in controls[m.seed]:
            role = c["identity"]
            s, fit, cal = role["held"], role["fit"], role["cal_a"]+role["cal_b"]
            if (len(fit), len(cal), len(set(fit+cal+[s]))) != (7, 6, 14) or set(fit+cal+[s]) != set(rows):
                raise ValueError("Changed source roles")
            name = f"{m.seed}_{s}"
            scratch = PRIVATE/f"{name}.json"
            if scratch.exists() and not replay:
                z = run.read(scratch)
                if z["registration"] != run.sha(freeze):
                    raise ValueError("Changed opening study")
            else:
                width = m.x.shape[-1]
                old = trainer.load(weights[name], lambda: parent.parent.study.old.model.network(width),
                                   parent.parent.study.old.STEPS)
                donors, held = {k: [] for k in fit}, []
                for rec, (ix, groups) in records.items():
                    src = str(d["sites"][ix[0]])
                    if src == s:
                        held.append((rec, ix, groups))
                    elif src in donors:
                        donors[src].append(add_labels(prepare(m, ix, groups, old), y[ix], valid[ix], cuts[src]))
                coeff, wait = model.fit(donors, width)
                data = [score(prepare(m, ix, groups, old), coeff, wait, y[ix], valid[ix],
                              d["queries"][ix], rec, cuts[s]) for rec, ix, groups in held]
                stats = {k: {j: sum(r[0][k][j] for r in data) for j in data[0][0][k]} for k in ARMS}
                effects = {k: readout.merge_comparisons([r[1][k] for r in data])["completion_lower_pp"]
                           for k in data[0][1]}
                oracle = {k: sum(r[2][k] for r in data) for k in data[0][2]}
                z = dict(registration=run.sha(freeze), seed=m.seed, metrics=stats, effects=effects, oracle=oracle,
                         coeff=coeff.tolist(), wait=wait.tolist(), fit_records=sum(map(len, donors.values())))
                run.once(scratch, run.stable(z))
            rows[s].append({k: v for k, v in z.items() if k not in ("coeff", "wait", "registration")})
            beat(cells=sum(map(len, rows.values())), seed=m.seed)
        del m
    flat = [r for rs in rows.values() for r in rs]
    metrics = {k: {f: sum(r["metrics"][k][f] for r in flat) for f in flat[0]["metrics"][k]} for k in ARMS}
    oracle = {k: sum(r["oracle"][k] for r in flat) for k in flat[0]["oracle"]}
    if len(flat) != 42 or metrics["incumbent"]["views"] != 102 or metrics["incumbent"]["edits"] != 12235:
        raise ValueError("Incomplete fixed cohort")
    if metrics["first_count_matched"]["edits"] != metrics["learned_stop"]["edits"]:
        raise ValueError("Matched count failed")
    draw = np.random.default_rng(73107).integers(len(rows), size=(3000, len(rows)))
    effects = {}
    for k in flat[0]["effects"]:
        v = np.array([np.mean([r["effects"][k] for r in rs]) for rs in rows.values()])
        effects[k] = dict(point=float(v.mean()), interval95=np.quantile(v[draw].mean(1), [.025, .975]).tolist())
    run.once(OUT/"analysis.json", run.stable(dict(registration=run.sha(freeze), by_site=rows, metrics=metrics,
        oracle=oracle, effects=effects, original_2pct=True, confirmation_access=False)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def beat(**p):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **p)
            run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", z)
            if p.get("cells", 0) % 14 == 0:
                print(z, flush=True)
        execute(a.replay, beat)
        beat(cells=42, complete=True, replay=a.replay)


if __name__ == "__main__":
    main()
