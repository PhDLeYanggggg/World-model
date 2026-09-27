"""Locality-first paired forecast summaries, not pooled-pixel deployment scores."""
import numpy as np


def endpoint_references(data, ids, baseline_index, endpoint):
    if endpoint not in ('ADE', 'FDE'):
        raise ValueError('ADE or FDE required')
    costs = data['baseline_' + endpoint.lower()][ids]
    return costs[:, baseline_index], costs[:, 1]


def masks(cv, easy_cut, hard_cut):
    cv = np.asarray(cv, float)
    known = np.isfinite(cv)
    if easy_cut <= 0 or hard_cut < easy_cut:
        raise ValueError('Training-defined positive easy/hard cuts required')
    return dict(all=known, positive_easy=known & (cv > 0) & (cv <= easy_cut),
                hard=known & (cv >= hard_cut), zero_CV=known & (cv == 0))


def slice_metrics(new, old, reference, cv, subset):
    new, old, reference, cv, subset = map(np.asarray, (new, old, reference, cv, subset))
    if not (new.ndim == 1 and new.shape == old.shape == reference.shape == cv.shape == subset.shape):
        raise ValueError('Aligned one-dimensional predictions required')
    if not np.array_equal(np.isnan(new), np.isnan(old)):
        raise ValueError('Paired label support differs')
    use = subset & np.isfinite(new) & np.isfinite(reference) & np.isfinite(cv)
    if not use.any(): return dict(rows=0, status='unsupported')
    a, b, r, c = (np.asarray(v[use], float) for v in (new, old, reference, cv))
    if any(not np.isfinite(v).all() or (v < 0).any() for v in (a, b, r, c)):
        raise ValueError('Finite nonnegative costs required')
    def gain(x, y): return None if y.mean() == 0 else float(100*(1-x.mean()/y.mean()))
    return dict(rows=int(use.sum()), status='defined', new_mean=float(a.mean()), old_mean=float(b.mean()),
        reference_mean=float(r.mean()), cv_mean=float(c.mean()),
        gain_vs_legacy_percent=gain(a, b), gain_vs_reference_percent=gain(a, r), gain_vs_CV_percent=gain(a, c),
        absolute_harm_vs_legacy=float((a-b).mean()), new_p95=float(np.quantile(a, .95)),
        old_p95=float(np.quantile(b, .95)), new_p99=float(np.quantile(a, .99)), old_p99=float(np.quantile(b, .99)),
        increased_error_fraction=float((a>b).mean()),
        mean_positive_harm_vs_legacy=float(np.maximum(a-b, 0).mean()),
        mean_positive_gain_vs_legacy=float(np.maximum(b-a, 0).mean()))


def paired_localities(rows, sites, key, draws=3000, seed=71431):
    """Average producer contexts/seeds within each site before resampling sites."""
    by_site = {}
    for site in sites:
        entries = [r for r in rows if r['site'] == site]
        values = [r['metric'].get(key) for r in entries]
        if not entries or any(v is None for v in values):
            by_site[site] = None
        else:
            by_site[site] = float(np.mean(values))
    if any(v is None for v in by_site.values()):
        return dict(by_site=by_site, point=None, ci95=None, status='incomplete_fixed_roster')
    x = np.array(list(by_site.values()))
    boot = np.random.default_rng(seed).choice(x, size=(draws, len(x))).mean(1)
    return dict(by_site=by_site, point=float(x.mean()), worst_locality=float(x.min()),
        ci95=np.quantile(boot, [.025, .975]).tolist(), status='defined',
        bootstrap_draws=draws, resampled_localities=len(sites),
        uncertainty='exploratory_source_localities_overlapping_producer_contexts_not_confirmation')
