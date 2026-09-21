"""Training-only cost geometry and support summaries; not a deployment rule."""
import numpy as np


def rollout_distance(candidate, baseline, scale):
    c, b, s = np.asarray(candidate, float), np.asarray(baseline, float), np.asarray(scale, float)
    if (c.shape != b.shape or c.ndim != 3 or c.shape[1:] != (12, 2)
            or s.shape != (len(c),) or not np.isfinite(c).all()
            or not np.isfinite(b).all() or not np.isfinite(s).all() or np.any(s <= 0)):
        raise ValueError('Finite twelve-step predictions and past-only scales required')
    return np.linalg.norm(c-b, axis=-1).mean(1)*s


def verify_cost_geometry(distance, cv, harm, benefit, full):
    d, cv, h, b, full = map(np.asarray, (distance, cv, harm, benefit, full))
    if (full.dtype != bool or any(x.shape != d.shape for x in (cv, h, b, full))
            or not np.isfinite(d).all() or np.any(d < 0)
            or not all(np.isfinite(x[full]).all() for x in (cv, h, b))
            or any(np.any(x[full] < 0) for x in (cv, h, b))):
        raise ValueError('Aligned complete supported costs required')
    tolerance = 1e-10*(1+d)
    if np.any(h[full] > (d+tolerance)[full]) or np.any(b[full] > (d+tolerance)[full]):
        raise ValueError('Full-horizon triangle bound violated')
    z = full & (cv == 0)
    np.testing.assert_allclose(h[z], d[z], rtol=1e-10, atol=1e-10)
    return dict(complete_rows=int(full.sum()), zero_reference_rows=int(z.sum()),
        zero_reference_identity_max_error=float(np.max(np.abs(h[z]-d[z]))) if z.any() else None,
        maximum_harm_bound_excess=float(np.max(h[full]-d[full])) if full.any() else None,
        all_checks_passed=True)


def summarize_group(mask, full, cv, benefit, harm, probability, expected, distance, tracks, sites):
    use = mask & full
    z = use & (cv == 0)
    event = z & (harm > 0)
    cost = np.where(cv == 0, harm, 0.)
    positive = use & (cost > 0)
    def mean(x):
        return float(np.mean(x[use])) if use.any() else None
    squared = np.sort(cost[use]**2)
    k = max(1, int(np.ceil(len(squared)*.01)))
    return dict(indexed_rows=int(mask.sum()), supported_rows=int(use.sum()),
        incomplete_rows=int((mask & ~full).sum()),
        positive_event_rows=int(event.sum()), unique_positive_tracks=len(np.unique(tracks[event])),
        positive_rows_by_scene={s:int((event & (sites == s)).sum()) for s in sorted(set(sites))},
        positive_tracks_by_scene={s:len(np.unique(tracks[event & (sites == s)])) for s in sorted(set(sites))},
        zero_reference_rows=int(z.sum()), beneficial_rows=int((use & (benefit > 0)).sum()),
        harmful_rows=int((use & (harm > 0)).sum()), mean_gain=mean(benefit-harm),
        predicted_event_mean=mean(probability), realized_event_rate=mean(event.astype(float)),
        predicted_protected_cost_mean=mean(expected), realized_protected_cost_mean=mean(cost),
        direct_expected_cost_mse=mean((expected-cost)**2),
        probability_times_distance_mse=mean((probability*distance-cost)**2),
        top_one_percent_squared_target_share=float(squared[-k:].sum()/squared.sum()) if squared.sum() > 0 else None,
        positive_cost_quantiles=np.quantile(cost[positive], [.5, .9, .99, 1]).tolist() if positive.any() else None,
        false_negative_score_le_0p01_rows=int((event & (probability <= .01)).sum()),
        false_negative_score_le_0p01_tracks=len(np.unique(tracks[event & (probability <= .01)])))
