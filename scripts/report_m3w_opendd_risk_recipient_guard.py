"""Freeze a recipient-preserving replay and attribute risk by slot and recipient."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_risk_moments as run

np, study, base, turn = run.np, run.study, run.base, run.turn
ARMS = ("soft_same_recipient", "prefix_same_recipient", "prefix_single_only", "prefix_double_only")


def guard(q, cv, parent, soft, prefix):
    base_edits, soft_edits, prefix_edits = ((p != cv).any((1, 2)) for p in (parent, soft, prefix))
    out = {k: parent.copy() for k in ARMS}
    counts = Counter()
    for at in run.slate.chunks(q):
        n = int(base_edits[at].sum())
        if not n:
            continue
        for k, path, edits in zip(ARMS[:2], (soft, prefix), (soft_edits, prefix_edits), strict=True):
            if np.array_equal(edits[at], base_edits[at]):
                out[k][at] = path[at]
            else:
                counts[f"{k}_recipient_rejects"] += 1
        out[ARMS[1+n]][at] = prefix[at]
        for k, p in out.items():
            counts[f"{k}_changes"] += int(not np.array_equal(p[at], parent[at]))
    for k, p in out.items():
        peak = turn.turn.peak(p-cv)
        ref = turn.turn.peak(parent-cv)
        np.testing.assert_allclose(peak[np.lexsort((peak, q))], ref[np.lexsort((ref, q))], atol=1e-9, rtol=1e-9)
        if k in ARMS[:2]:
            np.testing.assert_array_equal(peak > 0., ref > 0.)
    return out, counts


def rows(data, path, cut, outcomes):
    cv, ols, y, v, q, rec = (data[k] for k in ("cv", "ols", "y", "v", "q", "rec"))
    before = study.policy(path["pair_parent"], cv, cv, y, v, rec, easy_cut=cut[0], hard_cut=cut[1])["recordings"]
    out = {}
    for arm, p in outcomes.items():
        after = study.policy(p, cv, cv, y, v, rec, easy_cut=cut[0], hard_cut=cut[1])["recordings"]
        loss = Counter()
        for r, b in before.items():
            a = after[r]
            if not b["selected"]:
                continue
            loss["passes"] += a["supported"]
            loss["rescued"] += a["supported"] and not b["supported"]
            loss["lost"] += not a["supported"] and b["supported"]
            eb, ea = (z["completion_easy_risk_upper"] is None for z in (b, a))
            loss["easy_undefined"] += ea
            loss["easy_lost"] += ea and not eb
            loss["easy_recovered"] += eb and not ea
            if arm in ARMS[:2] and ea != eb:
                raise ValueError("Recipient-preserving selection changed easy-reference support")
        values = []
        for r in np.unique(rec):
            at = rec == r
            _, query = np.unique(q[at], return_inverse=True)
            a = turn.expanded.actual_contrast(cv[at], ols[at], p[at], path["pair_parent"][at],
                                              y[at], v[at], query, cut)
            if a is not None:
                values.append(dict(record=str(r), effect=a["pair_native"], qualified=a["qualified"]))
        out[arm] = (dict(loss), values)
    return out


def main():
    key, cohorts, _, _, _, _ = run.register()
    freeze, readout = (run.OUT/f"{n}.json" for n in ("freeze", "readout"))
    f, old = map(study.read, (freeze, readout))
    if f["registration"] != key or old["freeze"] != study.artifact(freeze) or old["reserved_outcomes_opened"]:
        raise ValueError("Require verified development parent")
    data = turn.exp.loader(cohorts)
    sites, counts = {}, Counter()
    for site in run.SITES:
        d = data[site]
        cut = np.quantile(np.concatenate([z["e"] for s, z in data.items() if s != site]), [.25, .75])
        per = {k: (Counter(), []) for k in ARMS}
        for seed in run.SEEDS:
            with np.load(study.checked(f["banks"][f"{site}_{seed}"]), allow_pickle=False) as p:
                for k, a in (("q", d["q"]), ("cv", d["cv"]), ("ols4", d["ols"])):
                    np.testing.assert_array_equal(p[k], a)
                path = {k: p[k] for k in ("pair_parent", "soft", "prefix")}
            variants, change = guard(d["q"], d["cv"], path["pair_parent"], path["soft"], path["prefix"])
            counts.update(change)
            for arm, (risk, result) in rows(d, path, cut, variants).items():
                per[arm][0].update(risk)
                per[arm][1].extend(result)
        sites[site] = {arm: dict(risk=dict(risk), pair=turn.effects.summary(v)) for arm, (risk, v) in per.items()}
    totals = {}
    for arm in ARMS:
        records = [v[arm] for v in sites.values()]
        risk = dict(sum((Counter(v["risk"]) for v in records), Counter()))
        totals[arm] = dict(risk=risk,
            qualified_positive_full_grid=sum(v["pair"]["positive_and_qualified"][1] for v in records),
            location_bootstrap_native=turn.effects.interval([v["pair"]["site_mean_native"] for v in records]))
    p = run.OUT/"recipient_guard.json"
    base.immutable(p, dict(parents=[study.artifact(x) for x in (freeze, readout, Path(__file__))],
        totals=totals, sites=sites, changes=dict(counts), result_source="fresh_run",
        exploratory_post_result=True, thresholds_tuned=False, original_2pct=True,
        independent_confirmation=False, reserved_outcomes_opened=False, deployment_changed=False))
    print(study.json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
