"""Audit the first calibration choice without changing any release threshold."""

from collections import Counter
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_paired_qualification as run

np, torch, study, base, old, prev, eq, cost = (
    run.np, run.torch, run.study, run.base, run.old, run.prev, run.eq, run.cost)


def conditions(d, candidate, a, y, v, cut, a_metrics):
    c, o = d["cv"], d["ols4"]
    e = np.where(v, np.linalg.norm(c-y, axis=-1), 0.)
    mass = e.sum()
    ade = e.sum(1)/v.sum(1).clip(1)
    cap = .02*ade[(ade > 0.) & (ade <= cut[0])].sum()
    t = prev.pair.audit.terms(candidate[None], c, o, a, y, v, cut[0], 1., max(mass, 1e-12), 1)[0]
    return dict(additive_easy_defined=a_metrics["risk"]["completion_easy_risk_upper"] is not None,
        additive_risk=bool(a_metrics["risk"]["supported"]),
        additive_dual=bool(a_metrics["gain"] is not None and min(a_metrics["gain"]) > 0.),
        candidate_support=bool(t[-1] > 0. and mass > 0.),
        candidate_pair=bool(t[0] > 1e-10), candidate_dual=bool((t[1:3] > 0.).all()),
        candidate_noninferior=bool(t[3] >= -1e-8), candidate_positive_harm=bool((t[4:6] <= 0.).all()),
        candidate_signed_easy=bool(t[6] <= cap))


def frozen_ablation_release(frozen, data, source, control):
    """An exploratory readout of nonprimary thresholds, with no refit or reselection."""
    out = {}
    for m in run.MODES:
        if m == run.PRIMARY:
            continue
        trials, active_folds = [], 0
        for name, fold in frozen["folds"].items():
            threshold = fold["calibration"]["modes"][m]["threshold"]
            if threshold is None:
                continue
            active_folds += 1
            cut = fold["identity"]["cut"]
            with (np.load(study.checked(fold["forecast"]), allow_pickle=False) as z,
                  np.load(study.checked(control["folds"][name]["forecast"]), allow_pickle=False) as c):
                for k in ("names", "offsets"):
                    np.testing.assert_array_equal(z[k], c[k])
                take = np.flatnonzero(z[f"score_{m}"] >= threshold)
                truth = prev.load_targets(source, data, [z["names"][i] for i in take])
                for i in take:
                    n, lo, hi = z["names"][i], z["offsets"][i], z["offsets"][i+1]
                    p, a, d, (y, v) = z[m][lo:hi], c["additive"][lo:hi], data[n], truth[n]
                    _, j, _ = prev.label_terms(d, p[None], a, y, v, cut)
                    am, jm = (prev.metrics(path, d, y, v, cut) for path in (a, p))
                    a_ok = eq.ceiling.eligible_additive(am)
                    trials.append(dict(fold=name, additive_eligible=a_ok, candidate_qualified=bool(j[0]),
                        candidate_risk_pass=bool(jm["risk"]["supported"]),
                        both=bool(a_ok and j[0]), checks=conditions(d, p, a, y, v, cut, am)))
        out[m] = dict(threshold_folds=active_folds, held_views=len(trials),
                      held_both=sum(v["both"] for v in trials), trials=trials)
    return out


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, data, src, roles, _, outer, control, _, _, _ = run.register()
    path = run.OUT/"summary.json"
    result = study.read(path)
    frozen = study.read(study.checked(result["freeze"]))
    if frozen["registration"] != key or frozen["held_labels_used"] or result["reserved_outcomes_opened"]:
        raise ValueError("Audit only the frozen development experiment")
    totals = {m: Counter() for m in run.MODES}
    rows = {}
    for held, role in roles.items():
        cy = prev.load_targets(src, data, [n for n, d in data.items() if d["city"] == role["calibration"] and d["added"]])
        for seed in eq.SEEDS:
            name = f"{held}_{seed}"
            fold = frozen["folds"][name]
            cut = fold["identity"]["cut"]
            ck = torch.load(study.checked(fold["fit"]["checkpoint"]), weights_only=True, map_location="cpu")
            if ck["identity"] != fold["identity"] or ck["step"] != run.STEPS:
                raise ValueError("Mismatched frozen qualification head")
            width = ck["net"]["candidate.0.weight"].shape[1]
            net = run.model.Qualification(width).eval()
            net.load_state_dict(ck["net"])
            action = run.prior.load_action(control["folds"][name])
            raw, offsets = eq.predict(old.load_outer(outer["folds"][name]), data, list(cy))
            trials = {m: [] for m in run.MODES}
            for n, lo, hi in zip(cy, offsets[:-1], offsets[1:], strict=True):
                d = dict(data[n], bank=eq.expanded(data[n], raw[lo:hi]))
                if (s := cost.slate(d, seed, cut, action)) is None:
                    continue
                a, pp, scores, _, q = run.rank(net, s, d["cv"])
                y, v = cy[n]
                a_metrics = prev.metrics(a, d, y, v, cut)
                a_ok = eq.ceiling.eligible_additive(a_metrics)
                for m, p in pp.items():
                    _, j, _ = prev.label_terms(d, p[None], a, y, v, cut)
                    pred = conditions(d, p, a, y, v, cut, a_metrics)
                    if bool(j[0]) != all(v for k, v in pred.items() if k.startswith("candidate_")):
                        raise ValueError("Failure accounting does not reproduce the qualification label")
                    trials[m].append(dict(score=scores[m], q=q, a=a_ok, j=bool(j[0]), checks=pred))
            rows[name] = {}
            for m, r in trials.items():
                r.sort(key=lambda v: v["score"], reverse=True)
                top = [v for v in r if v["score"] == r[0]["score"]]
                good = next((i for i, v in enumerate(r, start=1) if v["a"] and v["j"]), None)
                top_pass = all(v["a"] and v["j"] for v in top)
                if top_pass != bool(fold["calibration"]["modes"][m]["feasible"]):
                    raise ValueError("Best nonempty prefix disagrees with exact calibration")
                for v in top:
                    totals[m].update({k: int(not ok) for k, ok in v["checks"].items()})
                    totals[m].update(top_views=1, both_pass=int(v["a"] and v["j"]),
                        additive_only_failure=int(not v["a"] and v["j"]),
                        candidate_only_failure=int(v["a"] and not v["j"]),
                        both_fail=int(not v["a"] and not v["j"]))
                rows[name][m] = dict(top=top, first_both_rank=good, queries=len(r))
    receipt = dict(source=study.artifact(path), code=study.artifact(Path(__file__)),
        totals={k: dict(v) for k, v in totals.items()}, by_fold=rows,
        nonprimary_frozen_release=frozen_ablation_release(frozen, data, src, control),
        ablation_readout="post-primary development diagnostic; frozen calibration only, no reselection",
        result_source="fresh_run", reserved_outcomes_opened=False)
    base.immutable(run.OUT/"calibration_failure.json", receipt)
    print(study.json.dumps(receipt["totals"], indent=2))


if __name__ == "__main__":
    main()
