"""Compute the label-aware ceiling for the selected bridge experiment's analytic bank."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_bridge_policy_bottleneck as solver
from scripts import run_m3w_selected_bridge_hurdle as study


def main():
    run, np = study.run, study.np
    registration, _ = study.register()
    path = study.OUT / "analysis.json"
    result = run.read(path)
    if (result["registration"] != registration or result["confirmation_access"]
            or result["heads"] != run.sha(study.OUT / "head_freeze.json")):
        raise ValueError("Require bound selected bridge scores")
    sites = {s: study.prior.prior.old.load_site(s) for s in study.source.SITES}
    counts, by_site = Counter(), {}
    for held, d in sites.items():
        cut = np.quantile(np.concatenate([v["cut_samples"] for s, v in sites.items() if s != held]), [.25, .75])
        cv, ols, bank, *_ = study.prior.analytic(d)
        site = Counter()
        for rec in sorted(set(d["rec"])):
            on = d["rec"] == rec
            status = solver.exact(dict(cv=cv[on], ols=ols[on], bank=bank[on]),
                                  d["query"][on], d["y"][on], d["valid"][on], rec, cut)
            site[status] += 1
        by_site[held] = dict(site)
        counts.update(site)
    n = sum(counts.values())
    if n * 3 != result["summary"]["arms"]["support_hurdle"]["views"]:
        raise ValueError("Recording count does not match the three-seed experiment")
    run.once(study.OUT / "exact_analytic_capacity.json", run.stable(dict(
        analysis=run.artifact(path), solver=run.artifact(Path(solver.__file__)), code=run.artifact(Path(__file__)),
        original_2pct=True, confirmation_access=False, labels_hindsight_only=True,
        analytic_bank_amplitudes=[.25, .5, 1.], query_cap="min(2, query size - 1)",
        distinct_recordings=n, counts=dict(counts), by_site=by_site)))


if __name__ == "__main__":
    main()
