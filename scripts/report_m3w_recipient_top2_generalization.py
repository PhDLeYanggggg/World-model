"""Post-hoc same-source comparison for train-seen and source-held rankers."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_recipient_top2 as fit

np, torch = fit.np, fit.torch


def headroom(d, qs, paths, nominated, eligible):
    """Factorial hindsight only; do not choose amplitudes or candidates."""
    _, candidates, _, _, legal, _ = fit.prior.model.design(d["geometry"])
    y, valid, floor = d["target"], d["valid"], paths["ols4"]
    cv = fit.prior.ade(paths["cv"], y, valid)[0]
    sites, src = np.unique(d["sites"], return_inverse=True)
    weights = np.empty(len(y))
    for q in qs:
        weights[q] = 1. / len(q)

    def mass(value):
        return np.bincount(src, weights=weights * value, minlength=len(sites))

    def gain(change):
        return fit.prior.prior.scores(floor + change, floor, y, valid)["full_grid_lower"]

    denom = mass(cv)
    draws = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))
    cells = {}
    for name, alpha in (("quarter", .25), ("full", 1.)):
        g = gain(nominated * (alpha / fit.model.AMPLITUDE))
        top = fit.model.choose(g, eligible, qs)
        cells[f"fixed_{name}"] = 100. * mass(g * top) / denom
        values = np.column_stack([gain(alpha * candidates[:, i]) for i in range(candidates.shape[1])])
        best = np.where(legal, values, -np.inf).max(1)
        top = fit.model.choose(best, legal.any(1), qs)
        cells[f"menu_{name}"] = 100. * mass(np.where(top, best, 0.)) / denom
    intervals = fit.prior.prior.parent.old.metric.intervals
    return {k: intervals(v[:, None], draws) for k, v in cells.items()}


def main():
    _, jobs = fit.register()
    frozen = fit.prev.receipt(fit.PUBLIC, "train")
    parent = fit.prev.receipt(fit.PUBLIC, "replay")
    torch.set_num_threads(fit.CFG["torch_threads"])
    torch.set_num_interop_threads(1)
    d, qs, packed = fit.data(jobs, lambda **_: None)
    x, context, _, paths, delta, eligible, _, _ = packed
    x = {a: torch.from_numpy(context if a == "top2_context" else x) for a in fit.model.ARMS}
    y, valid, floor = d["target"], d["valid"], paths["ols4"]
    g = fit.prior.prior.scores(floor + delta, floor, y, valid)["full_grid_lower"]
    cv = fit.prior.ade(paths["cv"], y, valid)[0]
    sites = sorted(set(d["sites"]))
    observed = {}
    with np.load(fit.run.checked(frozen["scores"]), allow_pickle=False) as z:
        np.testing.assert_array_equal(z["ids"], d["ids"])
        held_scores = z["scores"]
    for row in frozen["fits"]:
        ident = row["identity"]
        held, arm, seed = ident["held"], ident["arm"], ident["seed"]
        checkpoint = torch.load(fit.run.checked(row["checkpoint"]), map_location="cpu", weights_only=False)
        if checkpoint["identity"] != ident or checkpoint["step"] != fit.CFG["updates"]:
            raise ValueError("The frozen checkpoint identity changed")
        n = fit.model.network(x[arm].shape[-1]).eval()
        n.load_state_dict(checkpoint["model"])
        k, a = fit.prev.SEEDS.index(seed), fit.model.ARMS.index(arm)
        iv = d["sites"] == held
        with torch.inference_mode():
            raw = n(x[arm]).flatten().numpy()
            checked = n(x[arm][iv]).flatten().numpy()
        np.testing.assert_array_equal(checked, held_scores[k, a, iv])
        raw[iv] = checked
        picked = fit.model.choose(raw, eligible, qs)
        for source in sites:
            queries = [q for q in qs if d["sites"][q[0]] == source]
            denom = sum(cv[q].mean() for q in queries)
            score = 100. * sum((g[q] * picked[q]).mean() for q in queries) / denom
            observed.setdefault((arm, seed, str(source)), {"train_seen": [], "held": []})[
                "held" if source == held else "train_seen"].append(float(score))

    summary = {}
    for arm in fit.model.ARMS:
        pairs = []
        for seed in fit.prev.SEEDS:
            for source in sites:
                values = observed[(arm, seed, str(source))]
                if len(values["train_seen"]) != 8 or len(values["held"]) != 1:
                    raise ValueError("Expected exactly eight train-seen fits and one held fit")
                pairs.append(dict(seed=seed, source=str(source),
                    train_seen_pp=float(np.mean(values["train_seen"])), held_pp=values["held"][0]))
        seen = float(np.mean([p["train_seen_pp"] for p in pairs]))
        held = float(np.mean([p["held_pp"] for p in pairs]))
        np.testing.assert_allclose(held, parent["utility"][arm]["full_grid_lower"]["point"], atol=1e-6)
        summary[arm] = dict(train_seen_pp=seen, held_pp=held, difference_pp=seen-held, paired=pairs)
    opportunity = headroom(d, qs, paths, delta, eligible)
    np.testing.assert_allclose(opportunity["fixed_quarter"]["point"],
        parent["utility"]["oracle"]["full_grid_lower"]["point"], atol=1e-9)
    prior = fit.prev.receipt(fit.prior.PUBLIC, "replay")
    np.testing.assert_allclose(opportunity["menu_full"]["point"],
        prior["utility"]["oracle"]["full_grid_lower"]["point"], atol=1e-6)
    fit.run.once(fit.PUBLIC / "generalization_diagnostic.json", dict(summary=summary, hindsight=opportunity,
        checkpoint_count=len(frozen["fits"]), fixed_same_source_windows=True,
        train_seen_is_resubstitution_not_validation=True, inference_replay_matches=True,
        bootstrap_for_train_seen_difference=False, no_retuning=True, independent_confirmation=False,
        main_sha256=fit.run.sha(fit.PUBLIC / "analysis.json"), implementation_sha256=fit.run.sha(Path(__file__))))


if __name__ == "__main__":
    main()
