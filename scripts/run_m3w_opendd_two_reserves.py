"""Evaluate independent gain and support slots on the exposed OpenDD roster."""

from collections import Counter
from pathlib import Path
import platform
import sys

import numpy as np

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_slate_queue as study
from scripts import run_m3w_opendd_reserve_only as sparse
from src.world_model import m3w_opendd_two_reserves as model

v, two, run, hard, prior = sparse.v, sparse.two, sparse.run, sparse.hard, sparse.prior
OUT = sparse.OUT.parent/"opendd_two_reserves_v1"
ARMS = ("cv", "gain", "support_only", "two_roles", "two_gain_matched", "two_roles_full_support", "oracle_same_query")
PAIRS = (("gain", "cv"), ("two_roles", "cv"), ("two_roles", "gain"), ("two_gain_matched", "cv"),
         ("two_roles", "two_gain_matched"), ("support_only", "cv"), ("two_roles_full_support", "two_roles"))


def register():
    pkey, outer, old, source, _, _, _ = prior.register()
    root = prior.OUT
    a, h = (run.read(root/f"{k}.json") for k in ("analysis", "head_freeze"))
    files = [sparse.OUT/f"{k}.json" for k in ("registration", "analysis")]
    r, b = map(run.read, files)
    if (a["registration"] != pkey or h["registration"] != pkey or a["heads"] != run.sha(root/"head_freeze.json")
            or b["registration"] != run.sha(files[0])
            or any(z["confirmation_access"] or not z["original_2pct"] for z in (a, r, b))):
        raise ValueError("Require frozen sparse gain policy and role separation")
    p = OUT/"registration.json"
    run.once(p, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(f) for f in (*files, root/"head_freeze.json")],
        native_steps=[8, 12], seeds=two.study.prior.SEEDS, arms=ARMS, no_new_training=True,
        primary="two roles vs a second gain-ranked recipient at exactly the same query and amplitude",
        secondary="preserve the frozen gain slot; full two-role policy versus CV",
        max_slots_per_recording=2, gain_amplitude=model.GAIN_SCALE, support_amplitude=model.SUPPORT_SCALE,
        support_sensitivity_amplitude=model.GAIN_SCALE, support_trigger="frozen easy >= .5",
        oracle="outcome-aware recipient change only, same second query, amplitude and count",
        same_query_order="gain first then support; distinct rows; no terminal forcing",
        development_budget_only=True, original_2pct=True, confirmation_access=False, no_threshold_search=True,
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    return run.sha(p), outer, old, source, h["heads"], b


def best_second(cv, ols, path, y, valid, base, easy_cut):
    old = two.moments(cv, base, y, valid, easy_cut)
    new = two.moments(cv, path, y, valid, easy_cut)
    take = (base != cv).any((1, 2))
    legal = (path != cv).any((1, 2)) & ~take
    fields = ("all_excess", "easy_excess", "easy_signed")
    risk = np.column_stack([new[k]+old[k][take].sum() for k in fields])
    support = new["easy"] | old["easy"][take].any()
    low = []
    for ref in (cv, ols):
        base_gain = v.paired(ref, base, y, valid)[1].sum()
        added = (v.paired(ref, path, y, valid)[1]-v.paired(ref, cv, y, valid)[1]).sum(1)
        low.append(base_gain+added)
    value = np.min(low, axis=0)
    passed = legal & support & (risk <= [0., 0., .02*old["ref_easy"].sum()]).all(1) & (value > 0.)
    return int(np.where(passed, value, -np.inf).argmax()) if passed.any() else None


@torch.inference_mode()
def evaluate(d, bank, path, starts, release, net, cuts, ref):
    cv, ols = bank[:2]
    if model.GAIN_SCALE != prior.model.AMPLITUDE:
        raise ValueError("Gain action amplitude must reproduce the previous policy")
    large = cv+model.GAIN_SCALE*(path-cv)
    small = cv+model.SUPPORT_SCALE*(path-cv)
    x, scale = prior.feature(d, cv, large)
    g = net(torch.from_numpy(x)).numpy()[:, 0]*scale
    e = release(torch.from_numpy(hard.model.features(d["h"]))).softmax(1).numpy()[:, 0]
    q, rec, y, valid = (d[k] for k in ("query", "rec", "y", "valid"))
    legal = (large != cv).any((1, 2))
    legal_small = (small != cv).any((1, 2))
    if not np.array_equal(legal, legal_small):
        raise ValueError("The two amplitudes must share the same actionable rows")
    outputs = {k: cv.copy() for k in ARMS}
    matched, support, fixed = [], [], {}
    for r in np.unique(rec):
        at = rec == r
        ids = np.unique(q[at])
        if not (np.diff(starts[ids]) > 0).all():
            raise ValueError("Queries must be chronological")
        rows = [np.flatnonzero(at & (q == j) & legal) for j in ids]
        a, b, c = model.choose((z, g[z], e[z]) for z in rows)
        if a is not None:
            for k in ("gain", "two_roles", "two_gain_matched", "two_roles_full_support"):
                outputs[k][a] = large[a]
        if b is not None:
            if a in (b, c) or q[b] != q[c] or rec[b] != r or rec[c] != r:
                raise ValueError("The second slot is not matched or overwrites the gain action")
            outputs["support_only"][b] = outputs["two_roles"][b] = small[b]
            outputs["two_roles_full_support"][b] = large[b]
            outputs["two_gain_matched"][c] = small[c]
            support.append(b)
            matched.append(c)
            fixed[r] = (a, b)
    outputs["oracle_same_query"] = outputs["two_roles"].copy()
    for r, (a, b) in fixed.items():
        at = np.flatnonzero(rec == r)
        available = legal[at] & (q[at] == q[b])
        candidate = cv[at].copy()
        candidate[available] = small[at[available]]
        winner = best_second(cv[at], ols[at], candidate, y[at], valid[at], outputs["gain"][at], cuts[0])
        if winner is not None:
            outputs["oracle_same_query"][b] = cv[b]
            outputs["oracle_same_query"][at[winner]] = candidate[winner]
            part = dict(y=y[at], valid=valid[at], rec=rec[at])
            score = v.parent.score(part, cv[at], ols[at], outputs["oracle_same_query"][at], cuts)
            if score["dual_positive"] != 1 or score["edits"] != int(a is not None)+1:
                raise ValueError("Same-query witness fails original checks")
    stats = {k: v.parent.score(d, cv, ols, p, cuts) for k, p in outputs.items()}
    if any(stats["gain"][f] != ref["metrics"]["reserve_free"][f] for f in v.prior.FIELDS):
        raise ValueError("Did not recover the sparse gain reference")
    for k in ("two_roles", "two_gain_matched", "two_roles_full_support", "oracle_same_query"):
        if stats[k]["edits"] != stats["gain"]["edits"]+len(support):
            raise ValueError("Unexpected edit count")
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    effects = {f"{a}_vs_{b}": hard.paired_pp(outputs[b], outputs[a], y, valid, mass) for a, b in PAIRS}
    np.testing.assert_allclose(effects["gain_vs_cv"], ref["effects"]["reserve_free_vs_cv"], atol=1e-12, rtol=0.)
    context = two.moments(cv, cv, y, valid, cuts[0])
    cover = dict(support=len(support), same_recipient=sum(a == b for a, b in zip(support, matched, strict=True)),
        support_easy=int(context["easy"][np.asarray(support, dtype=int)].sum()),
        matched_easy=int(context["easy"][np.asarray(matched, dtype=int)].sum()))
    easy_net = {}
    for k, p in outputs.items():
        m = two.moments(cv, p, y, valid, cuts[0])
        total = m["ref_easy"].sum()
        easy_net[k] = float(100.*m["easy_signed"].sum()/total) if total > 0. else None
    changes = Counter()
    for r in np.unique(rec):
        at = rec == r
        d0 = dict(y=y[at], valid=valid[at], rec=rec[at])
        joint = [v.parent.score(d0, cv[at], ols[at], outputs[k][at], cuts)["dual_positive"]
                 for k in ("gain", "two_roles", "two_gain_matched")]
        for name, j in (("vs_gain", joint[0]), ("vs_matched", joint[2])):
            changes[f"{name}_gains"] += int(joint[1] > j)
            changes[f"{name}_losses"] += int(joint[1] < j)
    return dict(metrics=stats, effects=effects, cover=cover, easy_net_percent=easy_net, joint_changes=changes,
        relative=study.c.relative_effect(d, outputs["gain"], outputs["two_roles"]))


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    key, outer, old, source, heads, a = register()
    sites = {s: study.c.forecast.load(s, source)[:3] for s in v.parent.source.SITES}
    result = {}
    for s, (d, _, starts) in sites.items():
        bank = v.parent.prior.analytic(d)
        cut = np.quantile(np.concatenate([z[0]["cut_samples"] for t, z in sites.items() if t != s]), [.25, .75])
        width = hard.model.features(d["h"][:1]).shape[1]
        result[s] = []
        for row in old["by_site"][s]:
            seed = row["seed"]
            release = two.fit.load(outer[f"{s}_{seed}"], lambda: hard.model.network(width), hard.STEPS)
            with np.load(run.checked(row["prediction"]), allow_pickle=False) as z:
                path = z["comoving_ego"]
            n = prior.feature({k: d[k][:1] for k in ("h", "query")}, bank[0][:1], path[:1])[0].shape[1]
            net = two.fit.load(heads[f"{s}_{seed}"], lambda: prior.model.network(n), prior.STEPS)
            ref = next(r for r in a["by_site"][s] if r["seed"] == seed)
            result[s].append(dict(seed=seed, **evaluate(d, bank, path, starts, release, net, cut, ref)))
        print(s, "two-slot readout complete", flush=True)
    flat = [r for rows in result.values() for r in rows]

    def ci(fn):
        cells = [[fn(r) for r in rows] for rows in result.values()]
        x = np.array([np.mean(z) for z in cells if all(y is not None for y in z)])
        if not len(x):
            return None
        draw = np.random.default_rng(817).integers(len(x), size=(3000, len(x)))
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()), sites=len(x))

    summary = dict(metrics={k: {f: sum(r["metrics"][k][f] for r in flat) for f in v.prior.FIELDS} for k in ARMS},
        effects={k: ci(lambda r: r["effects"][k]) for k in flat[0]["effects"]},
        easy_net_percent={k: ci(lambda r: r["easy_net_percent"][k]) for k in ARMS},
        relative=ci(lambda r: r["relative"]),
        cover=sum((Counter(r["cover"]) for r in flat), Counter()),
        joint_changes=sum((r["joint_changes"] for r in flat), Counter()))
    run.once(OUT/"analysis.json", run.stable(dict(registration=key, by_site=result, summary=summary,
        original_2pct=True, confirmation_access=False, submission_ready=False)))


if __name__ == "__main__":
    main()
