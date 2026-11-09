"""Attribute original recording risk and isolate the learned missing-label debit."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_slate_relative_gain as run
from src.evaluation import m3w_recording_risk_attribution as attribution

np, base, study, turn = run.np, run.base, run.study, run.turn
ARMS = ("additive", "risk_joint", "observed_joint", "full_grid_joint")


def report():
    key, cohorts, parent, _, _ = run.register()
    paths = [run.OUT / f"{x}.json" for x in ("registration", "freeze", "readout")]
    frozen, readout = map(study.read, paths[1:])
    if (frozen["registration"] != key or readout["registration"] != key
            or readout["freeze"] != study.artifact(paths[1]) or readout["reserved_outcomes_opened"]):
        raise ValueError("Mismatched readout or forecasts")
    data = turn.exp.loader(cohorts)
    all_risk, locations, totals = {k: [] for k in ARMS}, {}, []
    changed_queries = 0
    for site in run.SITES:
        d = data[site]
        cut = np.quantile(np.concatenate([z["e"] for s, z in data.items() if s != site]), [.25, .75])
        site_rows = []
        for seed in run.SEEDS:
            name = f"{site}_{seed}"
            with (np.load(study.checked(frozen["forecasts"][name]), allow_pickle=False) as p,
                  np.load(study.checked(parent["banks"][name]), allow_pickle=False) as z):
                for k, a in (("q", d["q"]), ("cv", d["cv"]), ("ols4", d["ols"]),
                             ("additive", z["turn_additive"]), ("risk_joint", z["turn_joint"])):
                    np.testing.assert_array_equal(p[k], a)
                forecast = {k: p[k] for k in ARMS}
            changed = (np.abs(forecast["observed_joint"]-forecast["full_grid_joint"]) > 1e-9).any((1, 2))
            changed_queries += len(np.unique(d["q"][changed]))
            cv, ols, y, v = (d[k] for k in ("cv", "ols", "y", "v"))
            for k, f in forecast.items():
                records = study.policy(f, cv, cv, y, v, d["rec"],
                                       easy_cut=cut[0], hard_cut=cut[1])["recordings"]
                all_risk[k].extend(z for z in records.values() if z["selected"])
            for r in np.unique(d["rec"]):
                at = d["rec"] == r
                q = np.unique(d["q"][at], return_inverse=True)[1]
                c = turn.expanded.actual_contrast(cv[at], ols[at], forecast["full_grid_joint"][at],
                    forecast["observed_joint"][at], y[at], v[at], q, cut)
                if c is not None:
                    site_rows.append(dict(record=str(r), effect=c["pair_native"], qualified=c["qualified"]))
        locations[site] = turn.effects.summary(site_rows)
        totals.extend({**x, "record": f"{site}:{x['record']}"} for x in site_rows)
    risk = {k: attribution.summarize(rows) for k, rows in all_risk.items()}
    for k, r in risk.items():
        old = readout["totals"]["arms"][k]
        if (r["switches"] != old["edits"] or r["disjoint_recording_patterns"].get("pass", 0) != old["risk_pass"]
                or r["views"] != 439):
            raise ValueError("Attribution failed to reproduce original scoring")
    effect = turn.effects.summary(totals)
    effect.pop("site_mean_native")
    effect["location_bootstrap_native"] = turn.effects.interval([a["site_mean_native"] for a in locations.values()])
    result = dict(bindings=[study.artifact(x) for x in (*paths, Path(__file__), Path(attribution.__file__))],
                  paired_changed_queries=changed_queries, debit_vs_no_debit=effect, risk=risk,
                  result_source="fresh_run", original_2pct=True, independent_confirmation=False)
    base.immutable(run.OUT / "risk_attribution.json", result)
    print(study.json.dumps(result, indent=2))


if __name__ == "__main__":
    report()
