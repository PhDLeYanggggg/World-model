"""Redistribute amplitude at unchanged recipients and per-query movement."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_cv_tangent as study

run, torch, np = study.run, study.torch, study.np
ARMS = ("uniform_half", "uniform_prior", "uniform_tangent", "adaptive_prior", "adaptive_tangent")
PAIRS = (("adaptive_tangent", "uniform_tangent"), ("adaptive_tangent", "adaptive_prior"),
         ("uniform_tangent", "uniform_prior"))


def reallocate(radius, probability, offsets):
    """Keep each query's sum exactly half, and every nonzero recipient active."""
    result = np.zeros_like(radius)
    for lo, hi in zip(offsets[:-1], offsets[1:], strict=True):
        r, p = radius[lo:hi], probability[lo:hi]
        if r.sum() > 0.:
            center = (r@p)/r.sum()
            result[lo:hi] = r*(.5+.4*(p-center))
    return result


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, _, previous = study.register()
    p = study.OUT / "analysis.json"
    old = run.read(p)
    if old["registration"] != key or old["confirmation_access"]:
        raise ValueError("Require frozen development forecasts")
    receipt = study.OUT / "budget_registration.json"
    run.once(receipt, run.stable(dict(parent=run.artifact(p), code=run.artifact(Path(__file__)),
        fraction=.5, centered_probability_slope=.4, exact_query_radius_sum=True, keep_all_recipients=True,
        no_refit_or_calibration=True, original_2pct=True, confirmation_access=False)))
    outer = study.old.frozen.load_networks(torch, study.producer)
    sources = {s: study.old.site_data(s, torch, study.producer, study.shape) for s in study.old.SITES}
    marked = {Path(a["path"]).stem: a for a in old["models"]}
    results = {}
    for pos, site in enumerate(study.old.SITES):
        d = sources[site]
        cv = d["cv"]
        cuts = study.old.base.cuts_for([sources[s] for s in study.old.SITES if s != site])
        results[site] = []
        for seed in study.job.SEEDS:
            c = study.old.candidate(d, outer[seed][pos], torch, study.producer)
            fit = {}
            for arm in study.HEADS:
                ref = marked[f"{site}_{seed}_{arm}"]
                ck = torch.load(run.checked(ref), weights_only=True, map_location="cpu")
                if ck["identity"]["key"] != key or ck["identity"]["held"] != site:
                    raise ValueError("Checkpoint does not exclude this location")
                net = study.model.CVTangent(c["x"].shape[1])
                net.load_state_dict(ck["model"])
                fit[arm] = (net.eval(), cuts, ref)
            base, r = study.paths(d, c, fit, site, seed, previous)
            with torch.inference_mode():
                _, logit = fit["tangent"][0](c["x"], c["unit"], torch.ones(len(r)))
                probability = logit.sigmoid().numpy().astype(float)
            adaptive = reallocate(r, probability, d["offsets"])
            np.testing.assert_allclose(np.add.reduceat(adaptive, d["offsets"][:-1]),
                                       .5*np.add.reduceat(r, d["offsets"][:-1]), atol=1e-12)
            out = {}
            for name, donor in zip(ARMS, ("half", "previous_shape", "tangent", "previous_shape", "tangent"), strict=True):
                radius = adaptive if name.startswith("adaptive_") else .5*r
                out[name] = cv+study.shape.at_radius(base[donor]-cv, radius)
                np.testing.assert_array_equal((out[name] != cv).any((1, 2)), r > 0.)
            scored = {}
            for name, f in out.items():
                row = study.policy(f, cv, cv, d["y"], d["valid"], d["recordings"],
                                   easy_cut=cuts[0], hard_cut=cuts[1])
                rr = list(row["recordings"].values())
                scored[name] = dict(vs_cv=study.old.base.paired_summary(d, f, cv, r > 0)["lower_pp"],
                    passes=sum(v["supported"] for v in rr), total=len(rr),
                    undefined_easy=sum(v["completion_easy_risk_upper"] is None for v in rr),
                    no_easy_in_roster=sum(v["known_easy_net_degradation"] is None for v in rr),
                    all_risk_violations=sum(v["completion_all_risk_upper"] is not None and
                        v["completion_all_risk_upper"] > .02 for v in rr),
                    worst_defined_easy_percent=max(100*v["known_easy_net_degradation"] for v in rr
                                                  if v["known_easy_net_degradation"] is not None))
            paired = {f"{a}_vs_{b}": study.old.base.paired_summary(d, out[a], out[b], r > 0)["lower_pp"]
                      for a, b in PAIRS}
            results[site].append(dict(seed=seed, policies=scored, paired=paired))
    draws = np.random.default_rng(817).integers(len(results), size=(3000, len(results)))
    def ci(fn):
        x = np.array([np.mean([fn(r) for r in items]) for items in results.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draws].mean(1), [.025, .975]).tolist())
    flat = [r for v in results.values() for r in v]
    summary = {}
    for k in ARMS:
        summary[k] = {f: sum(r["policies"][k][f] for r in flat) for f in
                      ("passes", "total", "undefined_easy", "no_easy_in_roster", "all_risk_violations")}
        summary[k]["vs_cv"] = ci(lambda r: r["policies"][k]["vs_cv"])
    paired = {k: ci(lambda r: r["paired"][k]) for k in flat[0]["paired"]}
    run.once(study.OUT / "budget.json", run.stable(dict(registration=run.sha(receipt), summary=summary,
        paired=paired, by_location=results, no_refit=True, no_calibration=True, confirmation_access=False)))


if __name__ == "__main__":
    main()
