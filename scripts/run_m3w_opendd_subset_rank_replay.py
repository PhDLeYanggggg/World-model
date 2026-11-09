"""Compare learned rankings at exactly the old optional release and radii."""

import argparse
from collections import Counter
import fcntl
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_subset_student as run

np, torch, base, study = run.np, run.torch, run.base, run.study
optional, turn, slate, shared = run.optional, run.turn, run.slate, run.shared
ARMS = ("old", *run.TARGETS)
OUT = run.OUT
LOCAL = run.LOCAL/"matched_rank"


def registration():
    key, cohorts, parent, _, old, _, _ = run.register()
    f = run.OUT/"freeze.json"
    trained = study.read(f)
    if trained["registration"] != key or not trained["inference_past_only"] or len(trained["fits"]) != 21:
        raise ValueError("Require all student weights before matched ranking")
    p = OUT/"rank_registration.json"
    base.immutable(p, dict(parents=[study.artifact(f), study.artifact(optional.OUT/"freeze.json")],
        code=study.artifact(Path(__file__)), roles=ARMS, release="frozen optional prefix joint",
        ranking="sum action logits, no threshold or new fitting", original_2pct=True,
        independent_confirmation=False))
    return base.sha(p), cohorts, parent, old, trained


@torch.inference_mode()
def rank(net, h, z, p):
    q, cv, ols, bank = (z[k] for k in ("q", "cv", "ols4", "bank"))
    original = p["prefix_joint"]
    out = dict(old=original, **{k: original.copy() for k in ARMS[1:]})
    peak = turn.turn.peak(original-cv)
    for ix in slate.chunks(q):
        radii = peak[ix][peak[ix] > 0.]
        if not len(radii):
            continue
        c, o, b, hist = (a[ix] for a in (cv, ols, bank, h))
        paths, row, arm, opts, group = optional.options(c, b, radii)
        opts = opts[group == (2 if len(radii) == 2 else 0), :len(radii)]
        x, _, legal, _ = shared.costs.features(hist, c, o, paths)
        if not len(opts) or not legal[row, arm].all():
            raise ValueError("Frozen release has no candidate slate")
        a = np.column_stack((x[row, arm], np.full(len(row), np.log1p(len(c))))).astype(np.float32)
        u = net(torch.from_numpy(a)).numpy()[opts].sum(1)
        for k, j in zip(ARMS[1:], u.argmax(0), strict=True):
            out[k][ix] = optional.paint(c, paths, row, arm, opts[j])
    for p in out.values():
        r = turn.turn.peak(p-cv)
        np.testing.assert_allclose(r[np.lexsort((r, q))], peak[np.lexsort((peak, q))], atol=1e-9, rtol=1e-9)
        np.testing.assert_array_equal(np.bincount(q, weights=r > 0.), np.bincount(q, weights=peak > 0.))
    return dict(q=q, cv=cv, ols4=ols, **out)


def freeze(key, cohorts, parent, old, trained):
    saved = {}
    for site in run.SITES:
        h, q, cv, ols, _ = turn.menu.parent.source_inputs(cohorts[site])
        for seed in run.SEEDS:
            name = f"{site}_{seed}"
            net = run.Student().eval()
            ref = trained["fits"][name]
            if ref["identity"]["held"] != site or site in ref["identity"]["donors"]:
                raise ValueError("Matched ranking was fit on the scored location")
            net.load_state_dict(run.risk.state(ref, site, run.STEPS))
            with (np.load(study.checked(parent["banks"][name]), allow_pickle=False) as z,
                  np.load(study.checked(old["forecasts"][name]), allow_pickle=False) as p):
                for k, a in (("q", q), ("cv", cv), ("ols4", ols)):
                    np.testing.assert_array_equal(z[k], a)
                    np.testing.assert_array_equal(p[k], a)
                values = rank(net, h, z, p)
            saved[name] = turn.exp.previous.save_bank(LOCAL/f"{name}.npz", values)
            run.beat(phase="matched_rank_freeze", fold=name, completed=len(saved))
    base.immutable(OUT/"rank_freeze.json", dict(registration=key, forecasts=saved, outcomes_accessed=False))


def evaluate(d, p, cut):
    cv, ols, y, v, q = (d[k] for k in ("cv", "ols", "y", "v", "q"))
    old = p["old"]
    paths = {k: p[k] for k in ARMS}
    rows = []
    for rec in np.unique(d["rec"]):
        at = d["rec"] == rec
        c, o, z, mask = (a[at] for a in (cv, ols, y, v))
        qq = np.unique(q[at], return_inverse=True)[1]
        outcomes = {}
        for k, path in paths.items():
            r = study.policy(path[at], c, c, z, mask, ["r"]*len(c),
                             easy_cut=cut[0], hard_cut=cut[1])["recordings"]["r"]
            effect = turn.expanded.actual_contrast(c, o, path[at], old[at], z, mask, qq, cut)
            outcomes[k] = dict(risk=r, effect=effect)
        rows.append(dict(record=str(rec), outcomes=outcomes))
    return rows


def summarize(rows):
    out = {}
    for k in ARMS:
        r = [x["outcomes"][k]["risk"] for x in rows]
        v = [(x["record"], a) for x in rows if (a := x["outcomes"][k]["effect"]) is not None]
        out[k] = dict(edits=sum(x["selected"] for x in r), passes=sum(x["supported"] for x in r),
            undefined_easy=sum(x["selected"] > 0 and x["completion_easy_risk_upper"] is None for x in r),
            easy_over_2pct=sum((x["known_easy_net_degradation"] or 0.) > .02 for x in r),
            conservative_individual_pp=turn.effects.site_means([(rec, a["accuracy_pp"]) for rec, a in v]),
            **turn.effects.summary([dict(record=rec, effect=a["pair_native"], qualified=a["qualified"]) for rec, a in v]))
    return out


def score(key, cohorts):
    p = OUT/"rank_freeze.json"
    f = study.read(p)
    if f["registration"] != key or f["outcomes_accessed"] or len(f["forecasts"]) != 21:
        raise ValueError("Freeze rankings before scoring")
    d = turn.exp.loader(cohorts)
    sites, merged = {}, []
    for s in run.SITES:
        cut = np.quantile(np.concatenate([d[k]["e"] for k in run.SITES if k != s]), [.25, .75])
        rows = []
        for seed in run.SEEDS:
            name = f"{s}_{seed}"
            with np.load(study.checked(f["forecasts"][name]), allow_pickle=False) as z:
                np.testing.assert_array_equal(z["q"], d[s]["q"])
                np.testing.assert_array_equal(z["cv"], d[s]["cv"])
                rows.extend(evaluate(d[s], z, cut))
            run.beat(phase="matched_rank_score", fold=name)
        sites[s] = summarize(rows)
        merged.extend({**r, "record": f"{s}:{r['record']}"} for r in rows)
    total = summarize(merged)
    for k, v in total.items():
        if v["edits"] != 282:
            raise ValueError("Matched control changed the optional release budget")
        for metric in ("site_mean_native", "conservative_individual_pp"):
            v.pop(metric)
            v[f"{metric}_bootstrap"] = turn.effects.interval([a[k][metric] for a in sites.values()])
    if total["old"]["passes"] != 35:
        raise ValueError("Frozen optional policy did not replay")
    base.immutable(OUT/"rank_readout.json", dict(registration=key, freeze=study.artifact(p), totals=total,
        sites=sites, result_source="fresh_run", original_2pct=True, independent_confirmation=False))
    print(study.json.dumps(total, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "score"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, cohorts, parent, old, trained = registration()
        if phase == "freeze":
            freeze(key, cohorts, parent, old, trained)
        else:
            score(key, cohorts)


if __name__ == "__main__":
    main()
