"""Source-development cap-event diagnostics with locality-level uncertainty."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from src.evaluation.m3w_harm_tail_diagnostics import top_mass_share


def support(target, sites, recordings, agents):
    target, sites, recordings, agents = map(np.asarray, (target, sites, recordings, agents))
    rows = []
    for site in sorted(set(sites)):
        use = (sites == site) & np.isfinite(target); positive = use & (target == 1)
        count_tracks = lambda mask: len(set(zip(recordings[mask].tolist(), agents[mask].tolist())))
        rows.append(dict(site=str(site), known=int(use.sum()), positive=int(positive.sum()),
                         negative=int((use & (target == 0)).sum()),
                         recordings=len(set(recordings[use].tolist())),
                         positive_recordings=len(set(recordings[positive].tolist())),
                         agents=count_tracks(use), positive_agents=count_tracks(positive)))
    return rows


def measures(target, score, overshoot, prior):
    target, score, overshoot = map(lambda v: np.asarray(v, float), (target, score, overshoot))
    if target.shape != score.shape or target.shape != overshoot.shape or target.ndim != 1:
        raise ValueError('Aligned one-dimensional diagnostic inputs required')
    use = np.isfinite(target)
    if not use.any():
        return dict(status='not_estimable', reason='no_positive_disagreement_label_support')
    y, s, h = target[use], score[use], overshoot[use]
    if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all() or not np.isfinite(h).all() or (h < 0).any():
        raise ValueError('Valid binary event, score and nonnegative overshoot required')
    rate = float(y.mean()); both = 0 < rate < 1
    w = np.full(len(y), 1/len(y))
    out = dict(status='measured', rows=len(y), positive=int(y.sum()), prevalence=rate,
               AUROC=float(roc_auc_score(y, s)) if both else None,
               AP=float(average_precision_score(y, s)) if both else None,
               top10_overshoot_mass=top_mass_share(s, h, w, .1) if both else None)
    if prior is not None:
        if not 0 < prior < 1 or ((s < 0) | (s > 1)).any():
            raise ValueError('Probabilistic scoring requires training prior and probability scores')
        p = s.clip(1e-7, 1-1e-7)
        out.update(Brier=float(((s-y)**2).mean()),
                   BCE=float(-(y*np.log(p)+(1-y)*np.log1p(-p)).mean()),
                   prior_Brier=float(((prior-y)**2).mean()),
                   prior_BCE=float(-(y*np.log(prior)+(1-y)*np.log1p(-prior)).mean()),
                   predicted_rate=float(s.mean()))
    return out


def paired_interval(values, *, seed, draws):
    """Seeds averaged inside each locality; overlapping assignments stay separate."""
    a = np.asarray(values, float)
    if a.shape != (3, 4) or not np.isfinite(a).all():
        return dict(status='not_estimable', reason='all_three_seeds_four_localities_required')
    local = a.mean(0)
    rng = np.random.default_rng(seed)
    samples = local[rng.integers(0, 4, size=(draws, 4))].mean(1)
    low, high = np.quantile(samples, [.025, .975])
    return dict(status='measured', point=float(local.mean()), low=float(low), high=float(high),
                sign='positive' if low > 0 else 'negative' if high < 0 else 'overlap',
                localities=4, seeds=3, bootstrap=draws)
