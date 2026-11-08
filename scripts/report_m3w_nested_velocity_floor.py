"""Separate selected-action error from the effect of the fallback floor."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_nested_velocity_selection as exp

np = exp.np


def readout(jobs, freeze):
    d = exp.parent.parent.parent.data(jobs, lambda **_: None)
    queries = exp.parent.parent.parent.old.fitting_layout(jobs[0]["identity"], d)
    paths = exp.parent.parent.model.inputs(d["geometry"])[-1]
    with np.load(exp.run.checked(freeze["scores"]), allow_pickle=False) as z, np.load(
            exp.run.checked(freeze["outer_bank"]), allow_pickle=False) as b:
        for bank in (z, b):
            np.testing.assert_array_equal(bank["ids"], d["ids"])
        scores, delta = z["scores"], b["delta"][:, exp.parent.model.ARMS.index("masked_plain")]
    v = d["valid"]
    y = np.where(v[..., None], d["target"], 0.)
    w = np.empty(len(y))
    for q in queries:
        w[q] = 1/len(q)
    _, src = np.unique(d["sites"], return_inverse=True)
    mass = lambda x: np.bincount(src, weights=w*x, minlength=9)
    cv = np.where(v, np.linalg.norm(paths["cv"]-y, axis=2), 0.).sum(1)/v.sum(1).clip(1)
    denom = mass(cv)
    all_rows = {}
    for i, seed in enumerate(exp.prev.SEEDS):
        proposal = paths["ols4"]+delta[i]
        size = np.linalg.norm(delta[i], axis=-1)
        legal = size.max(1) > 1e-8
        labels = {}
        for floor in ("cv", "ols4"):
            error = np.linalg.norm(paths[floor]-y, axis=2)-np.linalg.norm(proposal-y, axis=2)
            debit = np.linalg.norm(paths[floor]-proposal, axis=2)
            observed = np.where(v, error, 0.).mean(1)
            penalty = np.where(~v, debit, 0.).mean(1)
            labels[floor] = (observed, penalty)
        r, p = labels["cv"]
        ranks = dict(zip(exp.model.ARMS, scores[i], strict=True))
        ranks.update(min_change=-size.mean(1), random=np.random.default_rng(4533+seed).random(len(y)),
                     cv_oracle=r-p, ols4_oracle=labels["ols4"][0]-labels["ols4"][1])
        for arm, rank in ranks.items():
            pick = exp.model.choose(rank, legal, queries)
            for floor, (observed, penalty) in labels.items():
                row = dict(observed_grid_contribution=100*mass(pick*observed)/denom,
                           missing_grid_debit=100*mass(pick*penalty)/denom,
                           selected_action_lower=100*mass(pick*(observed-penalty))/denom)
                all_rows.setdefault(f"{arm}_on_{floor}", []).append(row)
    draws = np.random.default_rng(73107).integers(9, size=(3000, 9))
    ci = exp.parent.parent.parent.old.metric.intervals
    return dict(results={a: {m: ci(np.stack([r[m] for r in rows], 1), draws)
                             for m in rows[0]} for a, rows in all_rows.items()},
        same_indices_when_changing_floor=True, oracle_not_policy=True, original_rule_unchanged=True,
        development_targets=False, external_access=False, deployment=False)


if __name__ == "__main__":
    _, jobs, _ = exp.register()
    exp.prev.receipt(exp.PUBLIC, "replay")
    exp.run.once(exp.PUBLIC/"floor_diagnostic.json",
                 exp.run.stable(readout(jobs, exp.prev.receipt(exp.PUBLIC, "train"))))
