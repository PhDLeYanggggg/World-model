"""Compare all frozen nominations at the same fixed quarter amplitude."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_query_rank as exp

np = exp.np


def score(jobs, nested, freeze):
    d, queries, _, ref = exp.load_data(jobs, lambda **_: None)
    bank, prior = exp.banks(d, nested)
    with np.load(exp.run.checked(freeze["scores"]), allow_pickle=False) as z:
        np.testing.assert_array_equal(d["ids"], z["ids"])
        scores = z["scores"]
    target, valid = d["target"], d["valid"]
    y = np.where(valid[..., None], target, 0.)
    cv = ref["cv"]
    err = np.where(valid, np.linalg.norm(cv-y, axis=-1), 0.).sum(1)/valid.sum(1).clip(1)
    _, src = np.unique(d["sites"], return_inverse=True)
    w = np.empty(len(err))
    for q in queries:
        w[q] = 1/len(q)
    mass = lambda z: np.bincount(src, weights=w*z, minlength=9)
    denom = mass(err)
    rows, direct = {}, {}
    prefix = exp.model.prefix_control(d["geometry"][:, :16].reshape(-1, 8, 2))
    for k, seed in enumerate(exp.prev.SEEDS):
        shift = ref["ols4"]+bank[k]-cv
        p = cv+.25*shift
        gain = exp.forecast.scores(p, cv, target, valid)
        legal = np.linalg.norm(bank[k], axis=2).max(1)>1e-8
        ranks = dict(zip(exp.model.ARMS, scores[k], strict=True))
        ranks.update(min_cv_change=-np.linalg.norm(shift, axis=2).mean(1),
            prior_ols4=prior[k], prefix=prefix, random=np.random.default_rng(seed+4533).random(len(err)),
            oracle=gain["full_grid_lower"])
        pick = {a: exp.parent.model.choose(x, legal, queries) for a, x in ranks.items()}
        a = np.where(pick["cv_rank"][:, None, None], p, cv)
        for key, mask in pick.items():
            np.testing.assert_array_equal([mask[q].sum() for q in queries],
                                          [pick["cv_rank"][q].sum() for q in queries])
            b = np.where(mask[:, None, None], p, cv)
            rows.setdefault(key, []).append(100*mass(mask*gain["full_grid_lower"])/denom)
            paired = exp.forecast.scores(a, b, target, valid)["full_grid_lower"]
            direct.setdefault(key, []).append(100*mass(paired)/denom)
    draws = np.random.default_rng(73107).integers(9, size=(3000, 9))
    ci = lambda r: exp.data_source.old.metric.intervals(np.stack(r, 1), draws)
    return dict(quarter=0.25, full_grid_lower={k: ci(r) for k, r in rows.items()},
                direct_cv_rank_vs={k: ci(r) for k, r in direct.items()},
                equal_counts=True, independent_confirmation=False, deployment=False)


if __name__ == "__main__":
    _, jobs, _, nested = exp.register()
    exp.prev.receipt(exp.PUBLIC, "replay")
    exp.run.once(exp.PUBLIC/"quarter_matched_diagnostic.json",
                 exp.run.stable(score(jobs, nested, exp.prev.receipt(exp.PUBLIC, "train"))))
