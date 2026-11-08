"""Distinguish learned shape from shrinkage at each row's realized peak."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_completion_forecast as study

run, np, base = study.run, study.np, study.base


def verify_inputs():
    p = study.OUT / "analysis.json"
    f = study.OUT / "freeze.json"
    r = study.OUT / "registration.json"
    a, freeze, reg = (run.read(z) for z in (p, f, r))
    if (a["registration"] != run.sha(r) or a["freeze"] != run.sha(f)
            or freeze["registration"] != a["registration"] or a["confirmation_access"]
            or not a["original_2pct"] or not reg["no_held_selection"]):
        raise ValueError("Require frozen OpenDD-only forecasts")
    for k in ("code", "loss", "parent", "trainer"):
        run.checked(reg[k])
    return a, freeze


def compare(d, cv, ols, pred, masked, cuts):
    y, valid = d["y"], d["valid"]
    control = base.radius.match_radius(cv, pred, (masked - cv)[:, None])[:, 0]
    np.testing.assert_allclose(base.radius.peak(control - cv), base.radius.peak(pred - cv), atol=1e-10)
    np.testing.assert_array_equal((control != cv).any((1, 2)), (pred != cv).any((1, 2)))
    mass = np.linalg.norm(cv - y, axis=-1)[valid].sum()
    known, lower = base.paired(control, pred, y, valid)
    complete, partial = valid.all(1), valid.any(1) & ~valid.all(1)
    absent = ~valid.any(1)
    control_metrics = study.parent.score(d, cv, ols, control, cuts)
    raw_metrics = study.parent.score(d, cv, ols, pred, cuts)

    def slices(p):
        m = base.two.moments(cv, p, y, valid, cuts[0])
        hard = m["base"] > cuts[1]
        easy_mass, hard_mass = m["ref_easy"].sum(), m["base"][hard].sum()
        return dict(easy_known_degradation_percent=100. * float(m["easy_signed"].sum()) / float(easy_mass)
                    if easy_mass > 0. else None,
                    hard_known_gain_percent=100. * float(m["diff"][hard].sum()) / float(hard_mass)
                    if hard_mass > 0. else None)

    return dict(
        direct={k: float(100. * v.sum() / mass) for k, v in (
            ("known", known), ("lower", lower), ("complete", lower[complete]),
            ("partial", lower[partial]), ("absent", lower[absent]))},
        controls=dict(matched=control_metrics, trained=raw_metrics),
        slices=dict(matched=slices(control), trained=slices(pred), masked=slices(masked)))


def main():
    a, freeze = verify_inputs()
    key, _, _, _, _, source = study.register()
    if a["registration"] != key:
        raise ValueError("Base registration changed")
    sites = {s: base.load(s, source)[0] for s in study.parent.source.SITES}
    output = {}
    for s, d in sites.items():
        cv, ols = study.parent.prior.analytic(d)[:2]
        cuts = np.quantile(np.concatenate([v["cut_samples"] for k, v in sites.items() if k != s]), [.25, .75])
        output[s] = []
        for seed in study.SEEDS:
            p = run.checked(freeze["predictions"][f"{s}_{seed}"])
            with np.load(p, allow_pickle=False) as z:
                arms = {k: compare(d, cv, ols, z[k], z["masked"], cuts) for k in study.ARMS[1:]}
            output[s].append(dict(seed=seed, arms=arms))
    order = list(output)
    draw = np.random.default_rng(817).integers(len(order), size=(3000, len(order)))
    summary = {}
    for k in study.ARMS[1:]:
        out = summary[k] = {}
        for part in ("known", "lower", "complete", "partial", "absent"):
            x = np.array([np.mean([r["arms"][k]["direct"][part] for r in output[s]]) for s in order])
            out[part] = dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())
        out["controls"] = {m: {f: sum(r["arms"][k]["controls"][m][f] for group in output.values() for r in group)
                               for f in ("edits", "views", "undefined", "risk_pass", "dual_positive")}
                           for m in ("matched", "trained")}
        out["slices"] = {}
        for m in ("masked", "matched", "trained"):
            out["slices"][m] = {}
            for name in ("easy_known_degradation_percent", "hard_known_gain_percent"):
                values = [[r["arms"][k]["slices"][m][name] for r in output[s]] for s in order]
                site = [float(np.mean(x)) for x in values if all(v is not None for v in x)]
                out["slices"][m][name] = dict(
                    mean=float(np.mean(site)) if site else None, range=[min(site), max(site)] if site else None,
                    supported_sites=len(site))
    run.once(study.OUT / "radius_analysis.json", run.stable(dict(
        parent=run.artifact(study.OUT / "analysis.json"), code=run.artifact(Path(__file__)),
        result_source="fresh_run_frozen_predictions_rescored", rows=output, summary=summary,
        original_2pct=True, confirmation_access=False, deployment_changed=False)))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
