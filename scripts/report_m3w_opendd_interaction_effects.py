"""Compare realized interaction effects using paired source-level resampling."""

from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_opendd_expanded_interaction_frontier as expanded
from scripts import train_m3w_opendd_expanded_costs as refit

original, np, study = expanded.original, expanded.np, expanded.study
ARMS = ("frozen_original", "frozen_expanded", "refitted_expanded")


def site_means(records):
    seeds = defaultdict(list)
    for record, effect in records:
        seeds[record].append(effect)
    return np.mean([np.mean(xs, axis=0) for xs in seeds.values()], axis=0).tolist() if seeds else None


def interval(values):
    x = np.asarray([v for v in values if v is not None])
    if not len(x):
        return None
    ix = np.random.default_rng(3183).integers(len(x), size=(3000, len(x)))
    draws = x[ix].mean(1)
    return dict(site_mean=x.mean(0).tolist(), ci95=np.quantile(draws, [.025, .975], axis=0).T.tolist(),
                sites=len(x), resampling_unit="location")


def summary(rows):
    keep = [(r["record"], r["effect"]) for r in rows if r["effect"] is not None]
    good = [r for r in rows if r["qualified"] and r["effect"] is not None]
    return dict(estimable=len(keep), recordings=len({r for r, _ in keep}), joint_qualified=len(good),
                positive_and_qualified=[sum(r["effect"][k] > expanded.frontier.INTERACTION_TOL for r in good)
                                        for k in (0, 1)],
                unique_recordings_positive_and_qualified=[
                    len({r["record"] for r in good if r["effect"][k] > expanded.frontier.INTERACTION_TOL})
                    for k in (0, 1)],
                site_mean_native=site_means(keep))


def run():
    _, cohorts, _, _, old = original.register()
    first_path = original.OUT / "readout.json"
    second_path = expanded.OUT / "readout.json"
    costs_path = refit.OUT / "freeze.json"
    cost_reg = refit.OUT / "registration.json"
    first, second, costs = map(study.read, (first_path, second_path, costs_path))
    if (first["reserved_outcomes_opened"] or second["reserved_outcomes_opened"]
            or costs["registration"] != original.base.sha(cost_reg) or not costs["held_scoring_not_run"]):
        raise ValueError("Need frozen development-only sources")
    data = original.exp.loader(cohorts)
    groups, overall = {}, {name: [] for name in ARMS}
    for site in original.SITES:
        src = data[site]
        cut = np.quantile(np.concatenate([d["e"] for s, d in data.items() if s != site]), [.25, .75])
        rows = {arm: [] for arm in ARMS}
        for seed in original.SEEDS:
            key = f"{site}_{seed}"
            cert = {r["record"]: r["arms"]["two_edits_only"]
                    for r in study.read(study.checked(old["checkpoints"][key]))["rows"]}
            b = {r["record"]: r for r in study.read(study.checked(first["checkpoints"][key]))["rows"]}
            e = {r["record"]: r for r in study.read(study.checked(second["checkpoints"][key]))["rows"]}
            receipt = costs["fits"][key]["identity"]
            if receipt["held"] != site or set(receipt["donors"]) != set(original.SITES) - {site}:
                raise ValueError("Held location entered the refitted model")
            with np.load(study.checked(costs["banks"][key]), allow_pickle=False) as z:
                for k, expected in (("q", src["q"]), ("cv", src["cv"]), ("ols4", src["ols"])):
                    np.testing.assert_array_equal(z[k], expected)
                ta, tj = z["trained_additive"], z["trained_joint"]
            if not (set(cert) == set(b) == set(e) == set(map(str, np.unique(src["rec"])))):
                raise ValueError("Mismatch in paired recording roster")
            for r in np.unique(src["rec"]):
                at = src["rec"] == r
                q = np.unique(src["q"][at], return_inverse=True)[1]
                identity = str(r)
                c, base, plus = cert[identity], b[identity], e[identity]["actual"]
                actual = expanded.actual_contrast(src["cv"][at], src["ols"][at], tj[at], ta[at],
                                                  src["y"][at], src["v"][at], q, cut)
                base_eligible = base.get("pair_queries", [0, 0])
                frozen_ok = c.get("joint_dual", False) and c["joint_lower_pp"] >= -expanded.frontier.UTILITY_TOL_PP
                vals = [(base.get("learned_pair_native") if all(base_eligible) else None, frozen_ok)]
                vals.extend((item["pair_native"], item["qualified"]) if item is not None else (None, False)
                            for item in (plus, actual))
                if len({effect is not None for effect, _ in vals}) != 1:
                    raise ValueError("Arms do not share estimable recording-seed views")
                for arm, (effect, ok) in zip(ARMS, vals, strict=True):
                    rows[arm].append(dict(record=identity, effect=effect, qualified=ok))
        groups[site] = {arm: summary(z) for arm, z in rows.items()}
        for arm in ARMS:
            overall[arm].extend({**r, "record": f"{site}:{r['record']}"} for r in rows[arm])
    totals = {arm: summary(z) for arm, z in overall.items()}
    for arm, r in totals.items():
        r["location_bootstrap_native"] = interval([site[arm]["site_mean_native"] for site in groups.values()])
    losses = {}
    for k, objective in enumerate(original.OBJECTIVES):
        stages = Counter()
        for receipt in first["checkpoints"].values():
            for row in study.read(study.checked(receipt))["rows"]:
                if row.get("relaxed_pair_native", [0., 0.])[k] <= expanded.frontier.INTERACTION_TOL:
                    continue
                state = row["accuracy_parent_state"]
                if row.get("solve", {}).get(objective, {}).get("verified"):
                    z = row["solve"][objective]["effect_native"][k]
                    state = "strict_positive" if z > expanded.frontier.INTERACTION_TOL else "no_pair_gain_after_gate"
                elif row["state"] == "accuracy_would_decrease":
                    state = row["state"]
                stages[state] += 1
        losses[objective] = dict(stages)
    path = expanded.OUT / "realized_effects.json"
    original.base.immutable(path, dict(source="fresh_run", matched_controls=True,
        bindings=[study.artifact(x) for x in (first_path, second_path, costs_path, Path(__file__))],
        objectives=original.OBJECTIVES, bootstrap_draws=3000,
        totals=totals, sites=groups, original_relaxed_positive_attribution=losses,
        independent_confirmation=False, reserved_outcomes_opened=False))
    print(study.json.dumps(dict(totals=totals, losses=losses), indent=2))


if __name__ == "__main__":
    run()
