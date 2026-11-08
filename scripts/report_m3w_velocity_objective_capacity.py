"""Score exact one-change-per-query hindsight capacity, never deployment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_velocity_objectives as exp

np = exp.np


def evaluate(jobs, freeze):
    d = exp.parent.parent.data(jobs, lambda **_: None)
    queries = exp.parent.parent.old.fitting_layout(jobs[0]["identity"], d)
    _, _, _, _, _, paths = exp.parent.model.inputs(d["geometry"])
    ref = paths["ols4"]
    y, valid = d["target"], d["valid"]
    with np.load(exp.run.checked(freeze["predictions"]), allow_pickle=False) as z, np.load(
            exp.run.checked(freeze["parent_predictions"]), allow_pickle=False) as b:
        for bank in (z, b):
            np.testing.assert_array_equal(bank["ids"], d["ids"])
        new, original = z["delta"], b["delta"][:, exp.parent.model.ARMS.index("near")]
    _, src = np.unique(d["sites"], return_inverse=True)
    w = np.empty(len(src))
    for q in queries:
        w[q] = 1./len(q)
    mass = lambda value: np.bincount(src, weights=w*value, minlength=9)
    clean = np.where(valid[..., None], y, 0.)
    cv = np.where(valid, np.linalg.norm(paths["cv"]-clean, axis=-1), 0.).sum(1)/valid.sum(1).clip(1)
    denom = mass(cv)
    res, counts = {}, []
    for k, seed in enumerate(exp.prev.SEEDS):
        c = dict(grid_harm=original[k], fixed_neighbor=paths["fixed_neighbor"]-ref)
        c.update({a: new[k, j] for j, a in enumerate(exp.model.ARMS)})
        eligible = {a: np.linalg.norm(v, axis=-1).max(1)>1e-8 for a, v in c.items()}
        # Same query roster and exactly one nonidentity switch in every arm.
        active = [q for q in queries if len(q)>1 and all(legal[q].any() for legal in eligible.values())]
        counts.append(len(active))
        for a, v in c.items():
            gain = exp.parent.model.scores(ref+v, ref, y, valid)["full_grid_lower"]
            coin = np.random.default_rng(4533+seed).random(len(src))
            candidates = [q[eligible[a][q]] for q in active]
            arms = {
                "oracle": np.asarray([q[gain[q].argmax()] for q in candidates], dtype=int),
                "random": np.asarray([q[coin[q].argmax()] for q in candidates], dtype=int),
            }
            for control, ids in arms.items():
                value = np.zeros(len(src))
                value[ids] = gain[ids]
                res.setdefault(a, {}).setdefault(control, []).append(100*mass(value)/denom)
    draws = np.random.default_rng(73107).integers(9, size=(3000, 9))
    ci = exp.parent.parent.old.metric.intervals
    return dict(one_switch_per_query=True, intervention_queries=counts,
        results={a: {k: ci(np.stack(v, 1), draws) for k, v in row.items()} for a, row in res.items()},
        rows=len(src), source_groups=9, oracle_uses_labels=True, external_access=False,
        independent_confirmation=False, original_rule_unchanged=True, deployment=False)


if __name__ == "__main__":
    _, jobs, _ = exp.register()
    exp.prev.receipt(exp.PUBLIC, "replay")
    freeze = exp.prev.receipt(exp.PUBLIC, "train")
    exp.run.once(exp.PUBLIC/"capacity_diagnostic.json", exp.run.stable(evaluate(jobs, freeze)))
