"""Attribute frozen fixed-budget regret to recipient versus expert choice."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_expert_risk_allocation as fit

np = fit.np


def decompose(select, oracle, scores, legal):
    chosen = np.flatnonzero(select >= 0)
    best = np.flatnonzero(oracle >= 0)
    per_agent = np.where(legal, scores, -np.inf).max(1)
    actual = np.zeros(len(select))
    actual[chosen] = scores[chosen, select[chosen]]
    recipient_oracle = np.zeros_like(actual)
    recipient_oracle[chosen] = per_agent[chosen]
    budget_oracle = np.zeros_like(actual)
    budget_oracle[best] = per_agent[best]
    return actual, recipient_oracle, budget_oracle


def main():
    cut, jobs = fit.register()
    frozen = fit.prev.receipt(fit.PUBLIC, "train")
    parent = fit.prev.receipt(fit.PUBLIC, "replay")
    d = fit.prior.parent.data(jobs, lambda **_: None)
    queries = fit.prior.parent.old.fitting_layout(jobs[0]["identity"], d)
    _, delta, scale, paths, legal, _ = fit.model.design(d["geometry"])
    with np.load(fit.run.checked(frozen["scores"]), allow_pickle=False) as z:
        np.testing.assert_array_equal(d["ids"], z["ids"])
        risk = z["scores"][:, 1].astype(np.float64)
    y, valid = d["target"], d["valid"]
    cv, _ = fit.ade(paths["cv"], y, valid)
    easy = (cv > 0.) & (cv <= cut["easy"])
    true_gain = fit.model.targets(paths["ols4"], delta, y, valid, scale, easy)[..., 0] * (12. * scale[:, None])
    truth = fit.model.choose(true_gain, legal, queries)["independent"]
    sites, site = np.unique(d["sites"], return_inverse=True)
    weight = np.empty(len(site))
    for q in queries:
        weight[q] = 1. / len(q)

    def mass(z):
        return np.bincount(site, weights=weight * z, minlength=len(sites))

    denominator = mass(cv)
    values = {k: [] for k in ("action_regret", "recipient_regret", "total_regret",
        "learned", "same_recipients_oracle", "fixed_budget_oracle")}
    overlap, positive, harm_vs_ols = [], [], []
    floor_err = fit.ade(paths["ols4"], y, valid)[0]
    for seed in risk:
        predicted = (seed[..., 0] - .5*(seed[..., 1] + seed[..., 2])) * (12. * scale[:, None])
        select = fit.model.choose(predicted, legal, queries)["independent"]
        a, b, c = decompose(select, truth, true_gain, legal)
        for name, value in zip(values, (b-a, c-b, c-a, a, b, c), strict=True):
            values[name].append(100. * mass(value) / denominator)
        overlap.append(int(((select >= 0) & (truth >= 0)).sum()))
        positive.append(int(((select >= 0) & (a > 0.)).sum()))
        p = fit.model.selected_paths(paths["ols4"], delta, select)
        err = fit.ade(p, y, valid)[0]
        harm = mass(easy * np.maximum(err - floor_err, 0.))
        support = mass(easy * floor_err)
        harm_vs_ols.append({str(s): float(100*harm[i]/support[i]) if support[i]>0 else None
                            for i, s in enumerate(sites)})
    draws = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))
    parts = {key: fit.prior.parent.old.metric.intervals(np.stack(v, 1), draws)
             for key, v in values.items()}
    np.testing.assert_allclose(parts["total_regret"]["point"],
        parent["fixed_budget_regret"]["independent"]["point"], atol=1e-5)
    fit.run.once(fit.PUBLIC / "regret_diagnostic.json", dict(contrasts=parts, rows=len(site),
        shared_recipients_with_oracle=overlap, positive_learned_actions=positive,
        positive_easy_harm_relative_to_ols4=harm_vs_ols, no_retuning=True,
        hindsight_only=True, score_sha256=fit.run.sha(fit.PUBLIC / "analysis.json"),
        implementation_sha256=fit.run.sha(Path(__file__))))


if __name__ == "__main__":
    main()
