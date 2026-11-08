"""Test when a frozen OpenDD EqMotion budget should be spent."""

from collections import Counter
from pathlib import Path
import platform
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.special import expit

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_candidate_advantage as prior
from src.world_model import m3w_opendd_release_credit as model

run, parent, paired, two = prior.run, prior.parent, prior.paired, prior.two
OUT = prior.OUT.parent / "opendd_release_credit_v1"
DELAYS = (0, 2, 8)
FIELDS = ("edits", "views", "undefined", "risk_pass", "dual_positive")


def register():
    key, a, old_hard, old_heads, old, source = prior.register()
    p, h, c, cr = [prior.OUT/f"{k}.json" for k in ("analysis", "head_freeze", "capacity", "capacity_registration")]
    report, heads, cap, r = [run.read(x) for x in (p, h, c, cr)]
    if (report["registration"] != key or heads["registration"] != key or r["parent_registration"] != key
            or report["heads"] != run.sha(h) or cap["registration"] != run.sha(cr)
            or any(v["confirmation_access"] or not v["original_2pct"] for v in (report, cap, r))):
        raise ValueError("Require the exact original matched EqMotion readout")
    reg = OUT/"registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(x) for x in (p, h, c, cr)], native_steps=[8, 12], seeds=two.study.prior.SEEDS,
        exact_delay=DELAYS, causal_delay=2, policies=model.POLICIES, original_2pct=True, confirmation_access=False,
        no_training=True, no_tuning=True, release="previous hardness count at each query; no advance or cross-record borrowing",
        local_cap="current releases plus at most one extra edit, not exceeding legal rows",
        accuracy="direct paired lower gain versus immediate on surviving credits; also full policies",
        original_matched_intervention="original raw bank, equal realized recording total in hindsight; no radius match",
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    return run.sha(reg), a, old_heads, old, source, report, heads["heads"], cap


def solve(groups, release, capacity, gain, risk, support, base_ols, easy_mass, delay):
    """Exact prefix constraints are equivalent to FIFO matching of unit credits."""
    if not release.sum():
        return "no_edits", None
    cum = release.cumsum()
    due = np.r_[np.zeros(delay, dtype=int), cum][:len(release)].copy()
    available = cum-np.r_[0, due[:-1]]
    due[-1] = cum[-1]
    active = (available[groups] > 0) & (capacity[groups] > 0)
    if not (active & support).any():
        return "no_eligible_easy", None
    n = len(groups)
    q = np.arange(len(release))[:, None]
    local = q == groups[None]
    prefix = q >= groups[None]
    matrix = np.r_[np.c_[local, np.zeros(len(q))], np.c_[prefix, np.zeros(len(q))],
                   np.c_[risk.T, np.zeros(3)], np.c_[-gain.T, np.ones(2)], np.r_[support, 0.][None]]
    low = np.r_[np.zeros(len(q)), due, [-np.inf]*5, 1.]
    high = np.r_[capacity, cum, 0., 0., .02*easy_mass, 0., base_ols, np.inf]
    scale = np.maximum(np.max(np.abs(matrix), axis=1), 1.)
    result = milp(np.r_[np.zeros(n), -1.], integrality=np.r_[np.ones(n), 0.],
        bounds=Bounds(np.r_[np.zeros(n), -np.inf], np.r_[active.astype(float), np.inf]),
        constraints=LinearConstraint(matrix/scale[:, None], low/scale, high/scale),
        options=dict(time_limit=5., mip_rel_gap=0.))
    if result.status == 2:
        return "risk_infeasible", None
    if result.x is not None:
        on = result.x[:-1] > .5
        spend = np.bincount(groups[on], minlength=len(q))
        margin = (gain[on].sum(0)+[0., base_ols]).min()
        if (margin > 0. and support[on].any() and (risk[on].sum(0) <= [0., 0., .02*easy_mass]).all()
                and model.legal_schedule(release, spend, capacity, delay, True)):
            return "qualified", on
    bound = getattr(result, "mip_dual_bound", None)
    if bound is not None and np.isfinite(bound) and -bound < -1e-8:
        return "not_dual_positive", None
    return "unresolved", None


def assess(d, cv, ols, ego, releases, legal, cuts):
    y, valid, rec, query = (d[k] for k in ("y", "valid", "rec", "query"))
    m = two.moments(cv, ego, y, valid, cuts[0])
    risk = np.column_stack([m[k] for k in ("all_excess", "easy_excess", "easy_signed")])
    base = paired(ols, cv, y, valid)[1].sum(1)
    g = np.column_stack((paired(cv, ego, y, valid)[1].sum(1), paired(ols, ego, y, valid)[1].sum(1)-base))
    out = {str(k): Counter() for k in DELAYS}
    for r in np.unique(rec):
        full = np.flatnonzero(rec == r)
        qs = np.unique(query[full])
        at = full[legal[full]]
        group = np.searchsorted(qs, query[at])
        count = releases[qs]
        cap = np.minimum(count+1, np.bincount(group, minlength=len(qs)))
        for delay in DELAYS:
            status, choose = solve(group, count, cap, g[at], risk[at], m["easy"][at],
                                   base[full].sum(), m["ref_easy"][full].sum(), delay)
            if status == "qualified":
                p = cv[full].copy()
                p[np.searchsorted(full, at[choose])] = ego[at[choose]]
                exact = parent.score(dict(y=y[full], valid=valid[full], rec=rec[full]), cv[full], ols[full], p, cuts)
                if exact["dual_positive"] != 1 or exact["edits"] != int(count.sum()):
                    raise ValueError("Timing witness does not reproduce in original scorer")
            out[str(delay)][status] += 1
    return out


@torch.inference_mode()
def evaluate(d, starts, cv, ols, ego, cut, risk_net, hard_net, old, old_cap):
    q, rec, y, valid = (d[k] for k in ("query", "rec", "y", "valid"))
    x, scale = prior.model.features(d["h"], cv, ego)
    z = risk_net(torch.from_numpy(x)).numpy()
    _, risk = prior.model.scores(z, scale)
    p_easy = expit(z[:, 4])
    hard = hard_net(torch.from_numpy(prior.prior.model.features(d["h"]))).softmax(1).numpy()
    legal = (cv != ego).any((1, 2))
    release = prior.prior.model.decide(hard, legal)
    releases = np.bincount(q[release], minlength=q.max()+1)
    metrics, effects, logs = {}, {}, {}
    choices = {k: [] for k in model.POLICIES}
    expired = Counter()
    for r in np.unique(rec):
        qs = np.unique(q[rec == r])
        if not (np.diff(starts[qs]) > 0).all():
            raise ValueError("Original query clock is not strictly chronological")
        group = [np.flatnonzero((q == j) & legal) for j in qs]
        packets = [(rows, int(releases[j]), risk[rows], p_easy[rows]) for j, rows in zip(qs, group, strict=True)]
        for k in model.POLICIES:
            e, lost = model.stream(iter(packets), k)
            cap = np.minimum(releases[qs]+1, [len(g) for g in group])
            count = np.bincount([x[2] for x in e], minlength=len(qs))
            if not model.legal_schedule(releases[qs], count, cap, 2, lost == 0):
                raise ValueError("Causal policy broke the release constraints")
            if any(not 0 <= t <= 2 for _, _, _, t in e) or len(e)+lost != releases[qs].sum():
                raise ValueError("Causal policy reused or advanced a credit")
            choices[k].extend(e)
            expired[k] += lost
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    raw = dict(cv=cv)
    matched = {}
    for k, e in choices.items():
        rows = np.asarray(e, dtype=int).reshape(-1, 4)
        raw[k], matched[k] = cv.copy(), cv.copy()
        raw[k][rows[:, 0]], matched[k][rows[:, 1]] = ego[rows[:, 0]], ego[rows[:, 1]]
        if len(np.unique(rows[:, 0])) != len(e) or len(np.unique(rows[:, 1])) != len(e):
            raise ValueError("An agent or a released credit was reused")
        logs[k] = dict(expired=expired[k], moved=int((rows[:, 3] > 0).sum()))
        if k == "immediate":
            np.testing.assert_array_equal(raw[k], matched[k])
    for k, pred in raw.items():
        metrics[k] = parent.score(d, cv, ols, pred, cut)
    for field in FIELDS:
        np.testing.assert_allclose(metrics["immediate"][field], old["metrics"]["risk"][field])
    for field in ("vs_cv", "vs_ols4"):
        np.testing.assert_allclose(metrics["immediate"][field], old["metrics"]["risk"][field], atol=1e-12, rtol=0.)
    for k in model.POLICIES[1:]:
        metrics[f"{k}_matched"] = parent.score(d, cv, ols, matched[k], cut)
        if metrics[k]["edits"] != metrics[f"{k}_matched"]["edits"]:
            raise ValueError("Realized credit budget is not matched")
        effects[f"{k}_vs_matched"] = prior.prior.paired_pp(matched[k], raw[k], y, valid, mass)
        effects[f"{k}_vs_immediate"] = prior.prior.paired_pp(raw["immediate"], raw[k], y, valid, mass)
    capacity = assess(d, cv, ols, ego, releases, legal, cut)
    if capacity["0"] != old_cap["counts"]:
        raise ValueError("Zero-delay exact capacity does not reproduce")
    lower = [capacity[str(t)]["qualified"] for t in DELAYS]
    upper = [capacity[str(t)]["qualified"]+capacity[str(t)]["unresolved"] for t in DELAYS]
    if any(a > b for a, b in zip(lower[:-1], upper[1:], strict=True)):
        raise ValueError("Expanding allowed delay lost feasible witnesses")
    return dict(metrics=metrics, effects=effects, schedule=logs, capacity=capacity)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    key, a, hard, old, source, risk, heads, cap = register()
    sites = {s: prior.forecast.load(s, source)[:3] for s in parent.source.SITES}
    result = {}
    for held, (d, _, starts) in sites.items():
        cv, ols, *_ = parent.prior.analytic(d)
        cut = np.quantile(np.concatenate([v[0]["cut_samples"] for k, v in sites.items() if k != held]), [.25, .75])
        result[held] = []
        for v in a["by_site"][held]:
            seed = v["seed"]
            name = f"{held}_{seed}"
            cell = next(r for r in old["by_site"][held] if r["seed"] == seed)
            with np.load(run.checked(cell["prediction"]), allow_pickle=False) as z:
                ego = z["comoving_ego"]
            n = prior.model.features(d["h"][:1], cv[:1], ego[:1])[0].shape[1]
            m = prior.prior.model.features(d["h"][:1]).shape[1]
            value = two.fit.load(heads[f"readout_{name}"], lambda: prior.model.network(n), prior.STEPS)
            hardness = two.fit.load(hard[name], lambda: prior.prior.model.network(m), prior.prior.STEPS)
            old_score = next(r for r in risk["by_site"][held] if r["seed"] == seed)
            old_cap = next(r for r in cap["by_site"][held] if r["seed"] == seed)
            result[held].append(dict(seed=seed, **evaluate(d, starts, cv, ols, ego, cut, value, hardness, old_score, old_cap)))
        print(held, "complete", flush=True)
    flat = [r for v in result.values() for r in v]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(fn):
        x = np.array([np.mean([fn(r) for r in v]) for v in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()))

    metrics = {k: dict(**{f: sum(r["metrics"][k][f] for r in flat) for f in FIELDS},
                      vs_cv=ci(lambda r: r["metrics"][k]["vs_cv"])) for k in flat[0]["metrics"]}
    effects = {k: ci(lambda r: r["effects"][k]) for k in flat[0]["effects"]}
    capacity = {str(k): sum((r["capacity"][str(k)] for r in flat), Counter()) for k in DELAYS}
    schedule = {k: {f: sum(r["schedule"][k][f] for r in flat) for f in ("moved", "expired")} for k in model.POLICIES}
    run.once(OUT/"analysis.json", run.stable(dict(registration=key, by_site=result,
        summary=dict(metrics=metrics, effects=effects, capacity=capacity, schedule=schedule),
        original_2pct=True, confirmation_access=False, submission_ready=False)))


if __name__ == "__main__":
    main()
