"""Evaluate abstention with a matched release and a verified parent-subset ceiling."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_opendd_risk_moments as risk

np, torch, study, base, turn, slate, shared = risk.np, risk.torch, risk.study, risk.base, risk.turn, risk.slate, risk.shared
SITES, SEEDS = risk.SITES, risk.SEEDS
OUT = risk.OUT.parent/"opendd_optional_edits_v1"
LOCAL = risk.LOCAL.parent/OUT.name
ARMS = ("cv", "parent", "local", "prefix_additive", "prefix_joint")
EPS = 1e-10


def register():
    key, cohorts, parent, prior, _, _ = risk.register()
    paths = [risk.OUT/f"{n}.json" for n in ("registration", "freeze", "readout")]
    frozen, readout = map(study.read, paths[1:])
    if (frozen["registration"] != key or readout["freeze"] != study.artifact(paths[1])
            or not frozen["no_future_in_policy"] or readout["reserved_outcomes_opened"]):
        raise ValueError("Require the verified harm-moment parent")
    p = OUT/"registration.json"
    base.immutable(p, dict(parents=[study.artifact(x) for x in paths],
        code=[study.artifact(Path(x)) for x in (__file__, risk.model.__file__, slate.__file__)],
        roles=ARMS, sites=SITES, seeds=SEEDS, observed_steps=8, forecast_steps=12,
        release="positive min of two predicted gain increments; three predicted excesses <= 0",
        budget="keep any subset of original peak slots; zero edits are allowed and not a safety pass",
        comparison="prefix releases by joint-individual utility; compare risk-eligible add and pair at same radii",
        scope="exposed development; no new training or threshold search; no realized feedback",
        oracle="subset of frozen parent trajectories; maximize min lower gain over CV/OLS4 under original risk",
        original_2pct=True, independent_confirmation=False))
    return base.sha(p), cohorts, parent, prior, frozen


def options(c, bank, radii):
    paths, slots = shared.slot_menu(c, bank, radii)
    rows = np.concatenate([s[0] for s in slots])
    cols = np.concatenate([s[1]+k*bank.shape[1] for k, s in enumerate(slots)])
    count = np.cumsum([0, *[len(s[0]) for s in slots]])
    single = np.column_stack((np.arange(len(rows)), np.full(len(rows), -1)))
    double = (shared.legal_assignments(slots[0][0], slots[1][0]) if len(radii) == 2
              else np.empty((0, 2), dtype=int))
    opts = np.concatenate((single, double))
    group = np.r_[np.repeat(np.arange(len(slots)), np.diff(count)), np.full(len(double), 2)]
    return paths, rows, cols, opts, group


def paint(cv, paths, rows, cols, opt):
    use = opt[opt >= 0]
    y = cv.copy()
    y[rows[use]] = paths[rows[use], cols[use]]
    return y


def release(gain, moments, full_easy, ledger, past_easy):
    slack, d = risk.model.excess(moments+ledger, full_easy+past_easy)
    possible = (gain.min(1) > EPS) & (d > EPS).all(1) & (slack <= EPS).all(1)
    return np.flatnonzero(possible)


@torch.inference_mode()
def infer(net, cost, relative, h, z, p, rec):
    q, cv, ols, bank, native = (z[k] for k in ("q", "cv", "ols4", "bank", "native"))
    out = dict(cv=cv, parent=p["full_grid_joint"], **{k: cv.copy() for k in ARMS[2:]})
    ledger, easy_by_record, counts = {}, {}, Counter()
    x, unit, _, _ = shared.costs.features(h, cv, ols, cv[:, None])
    easy = risk.moments(net, x[:, 0], np.zeros(len(h)), unit)[:, 4]
    radii = turn.turn.peak(native-cv)
    for ix in slate.chunks(q):
        name = rec[int(q[ix][0])]
        past = easy_by_record.get(name, 0.)
        full_easy = float(easy[ix].sum())
        easy_by_record[name] = past+full_easy
        r = radii[ix][radii[ix] > 0.]
        if not len(r):
            continue
        hist, c, o, b = (a[ix] for a in (h, cv, ols, bank))
        paths, row, arm, opts, group = options(c, b, r)
        x, scale, legal, _ = shared.costs.features(hist, c, o, paths)
        if not legal[row, arm].all():
            raise ValueError("An edit radius lost its causal candidate")
        xx = x[row, arm]
        mu = risk.moments(net, xx, turn.turn.peak(paths[row, arm]-c[row]), scale[row])
        pred = cost(torch.from_numpy(xx)).numpy()[:, :2]*scale[row, None]
        active = opts >= 0
        g = np.where(active[..., None], pred[opts], 0.).sum(1)
        m = np.where(active[..., None], mu[opts], 0.).sum(1)
        pair = g.min(1)
        additive = np.where(active, pred.min(1)[opts], 0.).sum(1)
        possible = release(g, m, full_easy, 0., 0.)
        if len(possible):
            j = possible[np.argmax(pair[possible])]
            out["local"][ix] = paint(c, paths, row, arm, opts[j])
        counts["released_queries_considered"] += 1
        credit = ledger.get(name, np.zeros(6))
        live = release(g, m, full_easy, credit, past)
        if not len(live):
            counts["prefix_abstain"] += 1
            continue
        j = live[np.argmax(pair[live])]
        same = live[group[live] == group[j]]
        add = same[np.argmax(additive[same])]
        a = paint(c, paths, row, arm, opts[add])
        selected = add
        if group[j] == 2 and len(same) > 1:
            best = (-np.inf, add)
            for start in range(0, len(same), 128):
                take = same[start:start+128]
                trial = slate.assemble(c, paths, row, arm, opts[take])
                value = pair[take]+slate.WEIGHT*slate.gain(relative, hist, c, o, trial, a)[:, 2]
                k = int(value.argmax())
                if value[k] > best[0]:
                    best = (value[k], take[k])
            selected = best[1]
        out["prefix_additive"][ix] = a
        out["prefix_joint"][ix] = paint(c, paths, row, arm, opts[selected])
        ledger[name] = credit+m[selected]
        counts[f"prefix_release_{int(active[selected].sum())}"] += 1
        counts["joint_changes"] += int(not np.allclose(out["prefix_joint"][ix], a, atol=1e-9, rtol=1e-9))
    validate(q, cv, radii, out)
    return dict(q=q, cv=cv, ols4=ols, **{k: out[k] for k in ARMS if k != "cv"}), dict(counts)


def validate(q, cv, radii, out):
    for ix in slate.chunks(q):
        budget = list(sorted(radii[ix][radii[ix] > 0.]))
        got = {}
        for arm in ARMS[1:]:
            rr = turn.turn.peak(out[arm][ix]-cv[ix])
            got[arm] = sorted(rr[rr > 0.])
            leftover = budget.copy()
            for r in got[arm]:
                matches = np.flatnonzero(np.isclose(leftover, r, atol=1e-9, rtol=1e-9))
                if not len(matches):
                    raise ValueError("Invented or amplified a peak slot")
                leftover.pop(int(matches[0]))
        np.testing.assert_allclose(got["prefix_joint"], got["prefix_additive"], atol=1e-9, rtol=1e-9)


def freeze(key, cohorts, parent, prior, moment):
    receipts, counts = {}, {}
    for site in SITES:
        h, q, cv, ols, _ = turn.menu.parent.source_inputs(cohorts[site])
        rec = risk.time_order(cohorts[site], q)
        for seed in SEEDS:
            name = f"{site}_{seed}"
            if set(moment["fits"][name]["identity"]["source"]) != set(SITES)-{site}:
                raise ValueError("Frozen harm model included its evaluation site")
            net, cost, relative = risk.model.MomentHead().eval(), turn.refit.model.CostHead().eval(), slate.model.RelativeGain().eval()
            for model, source, steps in ((net, moment, risk.STEPS), (cost, parent, turn.refit.STEPS),
                                          (relative, prior, slate.STEPS)):
                model.load_state_dict(risk.state(source["fits"][name], site, steps))
            with (np.load(study.checked(parent["banks"][name]), allow_pickle=False) as z,
                  np.load(study.checked(prior["forecasts"][name]), allow_pickle=False) as p):
                risk.gather(z, p, dict(q=q, cv=cv, ols=ols), name)
                forecasts, counts[name] = infer(net, cost, relative, h, z, p, rec)
            receipts[name] = turn.exp.previous.save_bank(LOCAL/f"{name}.npz", forecasts)
            beat(phase="freeze", fold=name, complete=len(receipts))
            print(name, counts[name], flush=True)
    base.immutable(OUT/"freeze.json", dict(registration=key, forecasts=receipts, counts=counts,
        loaded_future_in_inference=False, outcome_scoring_not_run=True))


def beat(**fields):
    tmp = LOCAL/"heartbeat.pending"
    tmp.write_text(study.json.dumps(dict(pid=os.getpid(), **fields))+"\n")
    tmp.replace(LOCAL/"heartbeat.json")


def oracle(cv, ols, parent, y, v, q, cut):
    chosen = np.flatnonzero(turn.turn.peak(parent-cv) > 0.)
    if not len(chosen):
        return dict(state="no_parent_edits")
    w = 1./np.bincount(q)[q]
    total = float(w @ np.where(v, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
    moment = risk.model.target(cv, parent[:, None], y, v, cut[0])[:, 0]
    m = moment[chosen]
    if total <= 0. or not np.any(m[:, 4] > 0.):
        return dict(state="no_reference")
    lower = turn.exact.headroom.lower
    b = 100.*(w @ lower(ols, cv, y, v))/total
    gains = np.stack([w*lower(cv, parent, y, v), w*(lower(ols, parent, y, v)-lower(ols, cv, y, v))])
    gains = gains[:, chosen]*100./total
    n = len(chosen)
    slack, _ = risk.model.excess(m, 0.)
    a = np.vstack((np.column_stack((slack.T, np.zeros(3))),
                   np.r_[-(m[:, 4] > 0.).astype(float), 0.],
                   np.column_stack((-gains, np.ones(2)))))
    ub = np.r_[0., 0., .02*moment[:, 4].sum(), -1., 0., b]
    result = milp(np.r_[np.zeros(n), -1.], integrality=np.r_[np.ones(n), 0.],
                  bounds=Bounds(np.r_[np.zeros(n), -np.inf], np.r_[np.ones(n), np.inf]),
                  constraints=LinearConstraint(a, -np.inf, ub), options=dict(time_limit=2., mip_rel_gap=0.))
    label = {0: "optimal", 1: "limited", 2: "infeasible"}.get(result.status, "unresolved")
    if result.x is None:
        return dict(state=label)
    path = cv.copy()
    use = chosen[result.x[:-1] > .5]
    path[use] = parent[use]
    verdict = study.policy(path, cv, cv, y, v, ["r"]*len(cv), easy_cut=cut[0], hard_cut=cut[1])["recordings"]["r"]
    actual = float(100.*min(w@lower(k, path, y, v) for k in (cv, ols))/total)
    np.testing.assert_allclose(actual, result.x[-1], atol=1e-7, rtol=1e-7)
    if not verdict["supported"]:
        return dict(state="numerical_boundary", value_pp=actual)
    return dict(state=label, verified=True, value_pp=actual, edits=len(use), positive=actual > 1e-8)


def fold_score(d, values, cut):
    rows = []
    for r in np.unique(d["rec"]):
        at = d["rec"] == r
        cv, ols, y, v = (d[k][at] for k in ("cv", "ols", "y", "v"))
        q = np.unique(d["q"][at], return_inverse=True)[1]
        paths = dict(cv=cv, **{k: values[k][at] for k in ARMS[1:]})
        assessments = {}
        w = 1./np.bincount(q)[q]
        mass = float(w @ np.where(v, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
        for name, p in paths.items():
            s = study.policy(p, cv, cv, y, v, ["r"]*len(y), easy_cut=cut[0], hard_cut=cut[1])
            gain = [100.*float(w@turn.exact.headroom.lower(k, p, y, v))/mass for k in (cv, ols)] if mass > 0. else None
            assessments[name] = dict(risk=s["recordings"]["r"], gain_pp=gain, ade=s["known_ADE_pixel"],
                fde=s["available_endpoint_FDE_diagnostic_pixel"], tail=s["p95_known_ADE_pixel"],
                hard=s["subsets"]["hard"]["gain_percent"])
        c = turn.expanded.actual_contrast(cv, ols, paths["prefix_joint"], paths["prefix_additive"], y, v, q, cut)
        rows.append(dict(record=str(r), arms=assessments, matched=c,
                         oracle=oracle(cv, ols, paths["parent"], y, v, q, cut)))
    return rows


def summary(rows):
    result = {}
    for name in ARMS:
        a = [r["arms"][name] for r in rows]
        selected = [x for x in a if x["risk"]["selected"]]
        result[name] = dict(views=len(a), edited_views=len(selected), edits=sum(x["risk"]["selected"] for x in a),
            risk_pass=sum(x["risk"]["supported"] for x in a),
            easy_undefined_edited=sum(x["risk"]["completion_easy_risk_upper"] is None for x in selected),
            easy_undefined_all=sum(x["risk"]["completion_easy_risk_upper"] is None for x in a),
            signed_easy_exceeds=sum((x["risk"]["known_easy_net_degradation"] or 0.) > .02 for x in a),
            worst_signed_easy=max((x["risk"]["known_easy_net_degradation"] for x in a
                                   if x["risk"]["known_easy_net_degradation"] is not None), default=None),
            strict_dual_positive=sum(x["risk"]["supported"] and x["gain_pp"] is not None and min(x["gain_pp"]) > 1e-8 for x in a))
        for m in ("ade", "fde", "tail", "hard"):
            result[name][m] = float(np.mean(z)) if (z := [x[m] for x in a if x[m] is not None]) else None
        result[name]["by_record_gain_pp"] = turn.effects.site_means(
            [(r["record"], r["arms"][name]["gain_pp"]) for r in rows if r["arms"][name]["gain_pp"] is not None])
    m = [(r["record"], r["matched"]) for r in rows if r["matched"] is not None]
    result["matched"] = turn.effects.summary([dict(record=r, effect=x["pair_native"], qualified=x["qualified"]) for r, x in m])
    o = [r["oracle"] for r in rows]
    result["oracle"] = dict(states=dict(Counter(x["state"] for x in o)), verified=sum(x.get("verified", False) for x in o),
        verified_positive=sum(x.get("positive", False) for x in o),
        positive_edits=sum(x["edits"] for x in o if x.get("positive", False)),
        unique_positive_records=len({r["record"] for r in rows if r["oracle"].get("positive", False)}))
    return result


def score(key, cohorts):
    path = OUT/"freeze.json"
    f = study.read(path)
    expected = {f"{s}_{seed}" for s in SITES for seed in SEEDS}
    if f["registration"] != key or not f["outcome_scoring_not_run"] or set(f["forecasts"]) != expected:
        raise ValueError("Freeze every fold before opening scoring labels")
    data, by_site, all_rows, receipts = turn.exp.loader(cohorts), {}, [], {}
    for site in SITES:
        d = data[site]
        cut = np.quantile(np.concatenate([z["e"] for s, z in data.items() if s != site]), [.25, .75])
        rows = []
        for seed in SEEDS:
            name = f"{site}_{seed}"
            binding = dict(registration=key, forecast=f["forecasts"][name])
            cache = LOCAL/f"{name}_score.json"
            if not cache.exists():
                with np.load(study.checked(binding["forecast"]), allow_pickle=False) as p:
                    for k, x in (("q", d["q"]), ("cv", d["cv"]), ("ols4", d["ols"])):
                        np.testing.assert_array_equal(p[k], x)
                    values = {k: p[k] for k in ARMS[1:]}
                base.immutable(cache, dict(binding=binding, rows=fold_score(d, values, cut)))
            saved = study.read(cache)
            if saved["binding"] != binding:
                raise ValueError("Score input changed")
            rows.extend(saved["rows"])
            receipts[name] = study.artifact(cache)
            beat(phase="score", fold=name, complete=len(receipts))
            print(name, len(receipts), flush=True)
        by_site[site] = summary(rows)
        all_rows.extend({**r, "record": f"{site}:{r['record']}"} for r in rows)
    total = summary(all_rows)
    for k in ARMS:
        total[k].pop("by_record_gain_pp")
        total[k]["location_bootstrap_gain_pp"] = turn.effects.interval([s[k]["by_record_gain_pp"] for s in by_site.values()])
    total["matched"].pop("site_mean_native")
    total["matched"]["location_bootstrap_native"] = turn.effects.interval(
        [s["matched"]["site_mean_native"] for s in by_site.values()])
    if (total["parent"]["edits"], total["parent"]["risk_pass"]) != (3690, 47) or (
            total["prefix_joint"]["edits"] != total["prefix_additive"]["edits"]):
        raise ValueError("Parent or matched intervention count did not replay")
    base.immutable(OUT/"readout.json", dict(registration=key, freeze=study.artifact(path),
        result_source="fresh_run", fits_reused=21, new_fits=0, sites=by_site, totals=total, checkpoints=receipts,
        decisions=dict(sum((Counter(c) for c in f["counts"].values()), Counter())),
        original_2pct=True, reserved_outcomes_opened=False, independent_confirmation=False, deployment_changed=False))
    print(study.json.dumps(total, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "score"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for d in (OUT, LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, cohorts, parent, prior, moments = register()
        if phase == "freeze":
            freeze(key, cohorts, parent, prior, moments)
        else:
            score(key, cohorts)


if __name__ == "__main__":
    main()
