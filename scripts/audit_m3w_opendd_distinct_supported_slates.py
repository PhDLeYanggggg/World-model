"""Count distinct risk-feasible joint improvements at frozen OpenDD slots."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys
from time import time

from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csr_array, vstack

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_supported_acceleration as run
from src.evaluation import m3w_interaction_frontier as frontier
from src.evaluation.m3w_exact_slot_headroom import lower
from src.evaluation.m3w_selection_forecast_readout import errors

np, torch, study, base = run.np, run.torch, run.study, run.base
OUT = run.OUT.parent/"opendd_distinct_supported_slates_v1"
LOCAL = run.LOCAL.parent/OUT.name
PAIR_EPS, GAIN_EPS, ACC_TOL, SECONDS = 1e-10, 1e-8, 1e-8, 2.


def beat(**info):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(pid=os.getpid(), epoch=time(), **info))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def register():
    key, cohorts, turn, _, _, _ = run.register()
    p = run.OUT/"readout.json"
    readout = study.read(p)
    f = study.read(study.checked(readout["freeze"]))
    if (f["registration"] != key or readout["registration"] != key or readout["reserved_outcomes_opened"]
            or f["held_future_used_in_policy"] or not f["complete_before_scoring"]):
        raise ValueError("Need the sealed location-excluded forecasts")
    path = OUT/"registration.json"
    base.immutable(path, dict(parent=study.artifact(p), parent_freeze=readout["freeze"], cohorts=cohorts,
        code=[study.artifact(Path(x)) for x in (__file__, frontier.__file__, run.__file__)],
        variants=run.GROUPS, seeds=run.SEEDS, sites=run.SITES,
        fixed="exact gate-released query, slot count and peak offset; retain frozen additive singletons",
        eligibility="gain over CV and OLS4, noninferior to frozen additive, three original 2 percent risks",
        objective="full-grid paired relative-position improvement with missing-step debit",
        second="exclude all locally equivalent first paths (1e-9 atol/rtol), not just candidate indices",
        seconds_per_solve=SECONDS, pair_epsilon=PAIR_EPS, gain_epsilon_pp=GAIN_EPS,
        arithmetic_accuracy_tolerance_pp=ACC_TOL, original_2pct=True,
        hindsight_only=True, independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(path), cohorts, turn, f, readout


def local_choices(c, a, b, radii):
    """Complete menus for two slots; the singleton is frozen to additive."""
    if len(radii) < 2:
        return dict(fixed=a, count=1)
    paths, row, col, opts, group = run.prev.options(c, b, radii)
    opts = opts[group == 2]
    if not len(opts):
        raise ValueError("Empty exact-slot menu")
    return dict(c=c, paths=paths, row=row, col=col, opts=opts, count=len(opts))


def render(local, take):
    if "fixed" in local:
        return np.broadcast_to(local["fixed"], (len(take), *local["fixed"].shape))
    return run.slate.assemble(local["c"], local["paths"], local["row"], local["col"], local["opts"][take])


def terms(paths, cv, ols, additive, y, valid, cut, weight, mass, pair_count):
    """Each candidate uses only original scoring terms; future is diagnostic only."""
    n = valid.sum(1)
    c, _ = errors(cv, y, valid)
    easy = valid.any(1) & (c > 0.) & (c <= cut)
    err = np.linalg.norm(paths-y, axis=-1)
    ade = (err*valid).sum(2)/n.clip(1)
    radius = run.turn.turn.peak(paths-cv)
    edited = radius > 0.
    unknown = (radius*(~valid.any(1))).sum(1)
    harm = np.maximum(ade-c, 0.)*edited
    all_excess = harm.sum(1)+unknown-.02*(edited@c)
    easy_excess = harm[:, easy].sum(1)+unknown-.02*(edited[:, easy]@c[easy])
    signed = (ade[:, easy]-c[easy]).sum(1)
    gains = []
    for anchor in (cv, ols, additive):
        e = np.linalg.norm(anchor-y, axis=-1)
        debit = np.linalg.norm(paths-anchor, axis=-1)
        gains.append(np.where(valid, e-err, -debit).sum((1, 2))*100.*weight/mass)
    pair = frontier.pair_reduction(paths, additive, y, valid)[:, 1]/pair_count
    return np.column_stack((pair, *gains, all_excess, easy_excess, signed, edited[:, easy].sum(1)))


def design(cv, ols, additive, joint, bank, y, valid, q, cut):
    radius = run.turn.turn.peak(joint-cv)
    a_radius = run.turn.turn.peak(additive-cv)
    double = 0
    locals_, matrix = [], []
    weights = 1./np.bincount(q)[q]
    mass = float(weights@np.where(valid, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
    den = frontier.denominators(q, valid)
    if not (radius > 0.).any():
        return None, "no_edits"
    if not den.all() or mass <= 0.:
        return None, "no_reference"
    c, _ = errors(cv, y, valid)
    easy = valid.any(1) & (c > 0.) & (c <= cut)
    if not easy.any():
        return None, "no_record_easy_reference"
    fixed = np.zeros(8)
    for ids in (np.flatnonzero(q == i) for i in np.unique(q)):
        rr = radius[ids]
        np.testing.assert_allclose(np.sort(a_radius[ids]), np.sort(rr), rtol=1e-8, atol=1e-8)
        args = (*[x[ids] for x in (cv, ols, additive, y, valid)], cut, weights[ids[0]], mass, den[1])
        loc = local_choices(cv[ids], additive[ids], bank[ids], rr[rr > 0.])
        if "fixed" in loc:
            fixed += terms(loc["fixed"][None], *args)[0]
            continue
        double += 1
        rows = []
        for first in range(0, loc["count"], 128):
            paths = render(loc, np.arange(first, min(first+128, loc["count"])))
            actual = np.sort(run.turn.turn.peak(paths-cv[ids]), axis=1)[:, -2:]
            np.testing.assert_allclose(actual, np.broadcast_to(np.sort(rr[rr > 0.]), actual.shape), rtol=1e-8, atol=1e-8)
            rows.append(terms(paths, *args))
        loc["ids"] = ids
        locals_.append(loc)
        matrix.append(np.concatenate(rows))
    if not double:
        return None, "singletons_only"
    upper = fixed+np.array([z.max(0) for z in matrix]).sum(0)
    if upper[-1] < 1:
        return None, "no_edit_easy_reference"
    if min(upper[1:3]) < GAIN_EPS or upper[3] < -ACC_TOL:
        return None, "accuracy_relaxation"
    if upper[0] <= PAIR_EPS:
        return None, "pair_relaxation"
    return dict(locals=locals_, values=np.concatenate(matrix), fixed=fixed,
                cap=.02*float(c[easy].sum()), weights=weights, mass=mass, den=den, upper=upper[0]), None


def solve(d, forbidden=None):
    values, fixed = d["values"], d["fixed"]
    counts = [v["count"] for v in d["locals"]]
    group = np.repeat(np.arange(len(counts)), counts)
    by_query = coo_array((np.ones(len(group)), (group, np.arange(len(group)))), shape=(len(counts), len(group)))
    # Scale each constraint for numerical resolution at the original margins.
    a = np.column_stack((-values[:, 1:4], values[:, 4:7], -values[:, 7]))
    b = np.r_[fixed[1:4]-[GAIN_EPS, GAIN_EPS, -ACC_TOL], -fixed[4:7]+[0., 0., d["cap"]], fixed[7]-1.]
    scale = np.maximum(np.abs(a).max(0), np.maximum(np.abs(b), 1e-9))
    blocks = [by_query, csr_array((a/scale).T)]
    lb = np.r_[np.ones(len(counts)), np.full(len(b), -np.inf)]
    ub = np.r_[np.ones(len(counts)), b/scale]
    if forbidden is not None:
        blocks.append(csr_array(forbidden[None].astype(float)))
        lb, ub = np.r_[lb, -np.inf], np.r_[ub, len(counts)-1]
    cost = values[:, 0]
    objective_scale = max(float(np.abs(cost).max()), 1e-12)
    r = milp(-cost/objective_scale, integrality=np.ones(len(group)), bounds=Bounds(0., 1.),
             constraints=LinearConstraint(vstack(blocks, format="csc"), lb, ub),
             options={"time_limit": SECONDS, "mip_rel_gap": 0.})
    take = np.flatnonzero(r.x > .5) if r.x is not None else None
    if take is not None and not np.array_equal(group[take], np.arange(len(counts))):
        raise ValueError("Mixed-integer witness violates one-slate-per-query rule")
    if take is not None and forbidden is not None and forbidden[take].all():
        raise ValueError("Second witness repeats the first")
    upper = getattr(r, "mip_dual_bound", None)
    upper = fixed[0]-objective_scale*upper if upper is not None and np.isfinite(upper) else None
    return take, dict(status={0: "optimal", 1: "limited", 2: "infeasible"}.get(r.status, "unresolved"),
                      upper=float(upper) if upper is not None else None)


def reconstruct(d, take, additive):
    path = additive.copy()
    offset = 0
    for loc, j in zip(d["locals"], take, strict=True):
        path[loc["ids"]] = render(loc, np.array([j-offset]))[0]
        offset += loc["count"]
    return path


def equivalent(d, path):
    result = []
    for loc in d["locals"]:
        expected = path[loc["ids"]]
        for first in range(0, loc["count"], 128):
            sample = render(loc, np.arange(first, min(first+128, loc["count"])))
            result.extend(np.isclose(sample, expected, rtol=1e-9, atol=1e-9).all((1, 2, 3)))
    return np.asarray(result, dtype=bool)


def assess(cv, ols, add, joint, bank, y, v, q, cut):
    edited = run.turn.turn.peak(joint-cv) > 0.
    edits = int(edited.sum())
    doubles = int(np.count_nonzero(np.bincount(q, weights=edited) == 2))
    d, reason = design(cv, ols, add, joint, bank, y, v, q, cut[0])
    if d is None:
        return dict(state=reason, edits=edits, double_queries=doubles, witnesses=[])
    witnesses, logs, forbidden, first = [], [], None, None
    for _ in range(2):
        take, log = solve(d, forbidden)
        logs.append(log)
        if take is None:
            break
        path = reconstruct(d, take, add)
        pair = float(frontier.aggregate_pair(path, add, y, v, q, d["den"])[1])
        direct = np.array([100.*float(d["weights"]@lower(ref, path, y, v))/d["mass"] for ref in (cv, ols, add)])
        achieved = np.r_[pair, direct]
        np.testing.assert_allclose(achieved, (d["fixed"]+d["values"][take].sum(0))[:4], rtol=1e-8, atol=1e-8)
        if pair <= PAIR_EPS:
            log["nonpositive_pair"] = True
            break
        r = study.policy(path, cv, cv, y, v, ["r"]*len(y), easy_cut=cut[0], hard_cut=cut[1])["recordings"]["r"]
        if not r["supported"] or r["selected"] != edits or min(direct[:2]) <= 0. or direct[2] < -ACC_TOL:
            log["status"] = "numerical_boundary"
            break
        if first is not None and np.allclose(first, path, rtol=1e-9, atol=1e-9):
            raise ValueError("Second witness is not geometrically distinct")
        witnesses.append(dict(pair_native=pair, gain_pp=direct.tolist(), risk=r))
        if first is None:
            first = path
            forbidden = equivalent(d, first)
            if not forbidden[take].all():
                raise ValueError("Exclusion mask does not contain the first witness")
    last = logs[-1]
    proved_exhausted = (last["status"] == "infeasible"
                        or last["status"] == "optimal" and last.get("nonpositive_pair")
                        or last["upper"] is not None and last["upper"] <= PAIR_EPS)
    state = "two_verified" if len(witnesses) == 2 else "proved_exhausted" if proved_exhausted else "unresolved"
    return dict(state=state, edits=edits, witnesses=witnesses, logs=logs,
                double_queries=len(d["locals"]), relaxed_pair_upper=float(d["upper"]))


def tally(rows, group):
    edited = [r for r in rows if r[group]["edits"]]
    return dict(edited=len(edited), edits=sum(r[group]["edits"] for r in edited),
        with_double=sum(r[group].get("double_queries", 0) > 0 for r in edited),
        one_or_more=sum(bool(r[group]["witnesses"]) for r in edited),
        two_or_more=sum(len(r[group]["witnesses"]) == 2 for r in edited),
        unique_with_two=len({r["record"] for r in edited if len(r[group]["witnesses"]) == 2}),
        state_counts=dict(Counter(r[group]["state"] for r in edited)))


def run_audit(key, cohorts, turn, freeze, before):
    data = run.turn.exp.loader(cohorts)
    rows, sites, receipts = [], {}, {}
    for site in run.SITES:
        d = data[site]
        cut = np.quantile(np.concatenate([v["e"] for s, v in data.items() if s != site]), [.25, .75])
        local = []
        for seed in run.SEEDS:
            name = f"{site}_{seed}"
            binding = dict(registration=key, forecast=freeze["forecasts"][name], menu=turn["banks"][name])
            p = LOCAL/f"{name}.json"
            if not p.exists():
                with (np.load(study.checked(binding["forecast"]), allow_pickle=False) as f,
                      np.load(study.checked(binding["menu"]), allow_pickle=False) as m):
                    for k, expected in (("q", d["q"]), ("cv", d["cv"]), ("ols4", d["ols"])):
                        np.testing.assert_array_equal(f[k], expected)
                        np.testing.assert_array_equal(m[k], expected)
                    bank = run.with_acceleration(d["h"], d["cv"], m["bank"])
                    paths = {k: f[k] for k in run.ARMS}
                part = []
                for rec in np.unique(d["rec"]):
                    at = d["rec"] == rec
                    q = np.unique(d["q"][at], return_inverse=True)[1]
                    cv, ols, y, v = (d[k][at] for k in ("cv", "ols", "y", "v"))
                    arm = {}
                    for g in run.GROUPS:
                        a, j = (paths[f"{g}_{arm}"][at] for arm in ("add", "joint"))
                        arm[g] = assess(cv, ols, a, j, bank[at], y, v, q, cut)
                    part.append(dict(record=f"{site}:{rec}", seed=seed, **arm))
                base.immutable(p, dict(binding=binding, rows=part))
            item = study.read(p)
            if item["binding"] != binding:
                raise ValueError("Stale solver checkpoint")
            local.extend(item["rows"])
            receipts[name] = study.artifact(p)
            beat(phase="solve", fold=name, done=len(receipts), total=len(run.SITES)*len(run.SEEDS))
        sites[site] = {g: tally(local, g) for g in run.GROUPS}
        rows.extend(local)
    total = {g: tally(rows, g) for g in run.GROUPS}
    for g in run.GROUPS:
        b = before["totals"][f"{g}_joint"]
        if (total[g]["edited"], total[g]["edits"]) != (b["edited_views"], b["edits"]):
            raise ValueError("Fixed recording budget did not replay")
    base.immutable(OUT/"summary.json", dict(registration=key, result_source="fresh_run", input_source="cached_verified",
        record_views=len(rows), unique_records=len({r["record"] for r in rows}), totals=total, sites=sites,
        checkpoints=receipts, original_2pct=True, hindsight_only=True, no_new_fit=True,
        reserved_outcomes_opened=False, independent_confirmation=False, deployment_changed=False))
    print(study.json.dumps(total, indent=2), flush=True)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for p in (OUT, LOCAL):
        p.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_audit(*register())
        beat(phase="complete")


if __name__ == "__main__":
    main()
