"""Fixed-policy conditional cost diagnostics, never an inference repair."""
import numpy as np

ARMS = ('native', 'fraction', 'tempered')


def strict_bits(score, past, distance):
    score, past, distance = map(np.asarray, (score, past, distance))
    n = len(distance)
    if (score.shape != (n, 2) or past.shape != (n, 8, 2)
            or distance.shape != (n,) or not np.isfinite(score).all()
            or not np.isfinite(past).all() or not np.isfinite(distance).all()
            or (score < 0).any() or (distance < 0).any()):
        raise ValueError('Aligned finite past-only scores, history and disagreement required')
    return ((distance > 0) & np.any(past[:, -1] != past[:, -2], axis=1)
            & (score[:, 0] > score[:, 1]) & (score[:, 1] <= .1*score[:, 0]))


def fixed_groups(bits, eligible):
    eligible = np.asarray(eligible)
    if set(bits) != set(ARMS) or eligible.ndim != 1 or eligible.dtype.kind != 'b':
        raise ValueError('All three fixed causal decision vectors required')
    for x in bits.values():
        if (np.asarray(x).shape != eligible.shape or np.asarray(x).dtype.kind != 'b'
                or np.any(x & ~eligible)):
            raise ValueError('Boolean choices must be supported by causal eligibility')
    union = np.logical_or.reduce([bits[a] for a in ARMS])
    both = np.logical_and.reduce([bits[a] for a in ARMS])
    groups = dict(all=np.ones(len(eligible), bool), eligible=eligible,
                  union=union, intersection=both, eligible_rejected=eligible & ~union)
    groups.update({a+'_selected':bits[a] for a in ARMS})
    for a in ('native', 'fraction'):
        groups['tempered_not_'+a] = bits['tempered'] & ~bits[a]
        groups[a+'_not_tempered'] = bits[a] & ~bits['tempered']
    return groups


def conditional_stats(scores, target, full, has_any, mask, distance):
    target, full, has_any, mask, distance = map(np.asarray, (target, full, has_any, mask, distance))
    n = len(distance)
    if (set(scores) != set(ARMS) or target.shape != (n, 2)
            or any(v.shape != (n,) or v.dtype.kind != 'b' for v in (full, has_any, mask))
            or np.any(full & ~has_any) or not np.isfinite(target[full]).all()
            or (target[full] < 0).any() or not np.isfinite(distance).all() or (distance < 0).any()):
        raise ValueError('Explicit aligned cost support and frozen diagnostic mask required')
    for p in scores.values():
        if np.asarray(p).shape != target.shape or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError('Finite nonnegative score pairs required')
    use = mask & full
    record = dict(rows=int(mask.sum()), complete=int(use.sum()),
                  unknown_ADE=int((mask & ~has_any).sum()), incomplete=int((mask & ~full).sum()))
    if not use.any():
        return dict(record, costs=None)
    y, d = target[use], distance[use]
    den = np.where(d > 0, d, 1.)
    h = y[:, 1]
    harmful = h > 0
    costs = dict(realized_benefit=float(y[:, 0].mean()), realized_harm=float(h.mean()),
        realized_net_gain=float((y[:, 0]-h).mean()), harm_frequency=float(harmful.mean()),
        harm_given_event=None if not harmful.any() else float(h[harmful].mean()),
        harm_p95=float(np.quantile(h, .95)), harm_p99=float(np.quantile(h, .99)),
        realized_fractional_harm=float((h/den).mean()), mean_disagreement=float(d.mean()))
    for a in ARMS:
        p = scores[a][use]
        costs[a] = dict(predicted_benefit=float(p[:, 0].mean()), predicted_harm=float(p[:, 1].mean()),
            harm_bias=float((p[:, 1]-h).mean()), native_MSE=float(((p-y)**2).mean()),
            fraction_MSE=float((((p-y)/den[:, None])**2).mean()),
            predicted_fractional_harm=float((p[:, 1]/den).mean()))
    return dict(record, costs=costs)
