"""One causal amplitude-cap diagnostic on fixed CV rank decisions."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_query_rank as exp

np = exp.np


def past_radius(history):
    """Median absolute CV backtest error on observed points only."""
    errors = []
    for n in (4, 5):
        p = history[:, :n]
        t = np.arange(1., 9-n)[None, :, None]
        predicted = p[:, -1:]+np.diff(p[:, -2:], axis=1)*t
        errors.append(np.linalg.norm(predicted-history[:, n:], axis=2))
    return np.median(np.column_stack(errors), axis=1)


def score(jobs, cut, nested, freeze):
    d, queries, _, ref = exp.load_data(jobs, lambda **_: None)
    bank, _ = exp.banks(d, nested)
    with np.load(exp.run.checked(freeze["scores"]), allow_pickle=False) as z:
        np.testing.assert_array_equal(d["ids"], z["ids"])
        rank = z["scores"][:, exp.model.ARMS.index("cv_rank")]
    v = d["valid"]
    y = np.where(v[..., None], d["target"], 0.)
    cv = ref["cv"]
    cv_error = np.where(v, np.linalg.norm(cv-y, axis=2), 0.).sum(1)/v.sum(1).clip(1)
    easy = (cv_error > 0.) & (cv_error <= cut["easy"])
    hard = (cv_error >= cut["hard"]) & v.any(1)
    radius = past_radius(d["geometry"][:, :16].reshape(-1, 8, 2).astype(float))
    w = np.empty(len(y))
    for q in queries:
        w[q] = 1./len(q)
    _, src = np.unique(d["sites"], return_inverse=True)
    mass = lambda z: np.bincount(src, weights=w*z, minlength=9)
    denom, easy_mass = mass(cv_error), mass(easy*cv_error)
    rows = {}
    for k, seed in enumerate(exp.prev.SEEDS):
        delta = ref["ols4"]+bank[k]-cv
        legal = np.linalg.norm(bank[k], axis=2).max(1)>1e-8
        chosen = exp.parent.model.choose(rank[k], legal, queries)
        r = np.linalg.norm(delta, axis=2).max(1)
        ratio = np.minimum(1., radius/r.clip(1e-8))
        for arm, alpha in dict(full=np.ones(len(y)), fixed_quarter=np.full(len(y), .25),
                               causal_cap=ratio).items():
            shift = delta*(chosen*alpha)[:, None, None]
            gain = exp.forecast.scores(cv+shift, cv, y, v)
            known_harm = 100*mass(easy*np.maximum(-gain["observed"], 0.))[easy_mass>0]/easy_mass[easy_mass>0]
            rows.setdefault(arm, []).append(dict(lower=100*mass(gain["full_grid_lower"])/denom,
                hard=100*mass(hard*gain["full_grid_lower"])/denom, observed=100*mass(gain["observed"])/denom,
                nominated=int(chosen.sum()), nonidentity=int((chosen & (alpha*r>1e-8)).sum()),
                median_amplitude=float(np.median(alpha[chosen])),
                max_known_easy_harm=float(known_harm.max()), undefined_easy=int((easy_mass==0).sum())))
    draws = np.random.default_rng(73107).integers(9, size=(3000, 9))
    ci = exp.data_source.old.metric.intervals
    return dict(arms={a: {m: ci(np.stack([r[m] for r in rr], 1), draws) if isinstance(rr[0][m], np.ndarray)
                          else [r[m] for r in rr] for m in rr[0]} for a, rr in rows.items()},
        same_nominations=True, amplitude_search=False, no_threshold_search=True, original_rule_unchanged=True,
        independent_confirmation=False, external_access=False, deployment=False)


if __name__ == "__main__":
    cut, jobs, _, nested = exp.register()
    exp.prev.receipt(exp.PUBLIC, "replay")
    freeze = exp.prev.receipt(exp.PUBLIC, "train")
    exp.run.once(exp.PUBLIC/"cap_diagnostic.json", exp.run.stable(score(jobs, cut, nested, freeze)))
