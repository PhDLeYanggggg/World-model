"""Frozen-action risk accounting; not a calibrator or inference policy."""
import numpy as np


def decompose(y, p, selected, tail_cut, *, easy=False):
    y, p = np.asarray(y, float), np.asarray(p, float)
    selected = np.asarray(selected)
    if (y.ndim != 2 or y.shape[1] != 5 or p.shape != y.shape or selected.shape != (len(y),)
            or selected.dtype != bool or not np.isfinite(p).all() or (p < 0).any()
            or not np.isfinite(tail_cut) or tail_cut < 0):
        raise ValueError('Aligned native moments, frozen actions and training-only tail cutoff required')
    known = np.isfinite(y).all(1)
    if not (known | np.isnan(y).all(1)).all() or (y[known] < 0).any():
        raise ValueError('Unknown rows must remain entirely unknown')
    h, r = (4, 3) if easy else (1, 2)
    at = known & selected
    truth_h, truth_r, estimate_h, estimate_r = (z[at, k] for z, k in ((y, h), (y, r), (p, h), (p, r)))
    reference = float(truth_r.sum()); harm = float(truth_h.sum())
    predicted_harm, predicted_reference = float(estimate_h.sum()), float(estimate_r.sum())
    predicted_excess = predicted_harm-.02*predicted_reference
    harm_error = harm-predicted_harm
    reference_error = .02*(predicted_reference-reference)
    true_excess = harm-.02*reference
    np.testing.assert_allclose(predicted_excess+harm_error+reference_error, true_excess, atol=1e-9, rtol=1e-10)
    tail = truth_h > tail_cut
    underestimated = np.maximum(truth_h-estimate_h, 0)
    out = dict(rows=len(y), selected_rows=int(selected.sum()), known_selected=int(at.sum()),
        unknown_selected=int((selected & ~known).sum()), true_harm=harm, true_reference=reference,
        predicted_harm=predicted_harm, predicted_reference=predicted_reference,
        true_excess=true_excess, predicted_excess=predicted_excess,
        harm_underestimate=harm_error, reference_overestimate=reference_error,
        positive_harm_underestimate=float(underestimated.sum()),
        tail_harm=float(truth_h[tail].sum()), tail_positive_underestimate=float(underestimated[tail].sum()),
        tail_rows=int(tail.sum()), positive_harm_rows=int((truth_h > 0).sum()),
        harmful_switch_fraction=float((truth_h > 0).mean()) if len(truth_h) else None,
        predicted_risk_ratio=predicted_harm/predicted_reference if predicted_reference > 0 else None,
        realized_risk_ratio=harm/reference if reference > 0 else None)
    for k in ('true_excess', 'predicted_excess', 'harm_underestimate', 'reference_overestimate'):
        out[k+'_over_selected_reference'] = out[k]/reference if reference > 0 else None
    out['tail_harm_share'] = out['tail_harm']/harm if harm > 0 else None
    out['tail_positive_underestimate_share'] = (out['tail_positive_underestimate']/out['positive_harm_underestimate']
                                               if out['positive_harm_underestimate'] > 0 else None)
    out['risk_certified'] = False
    return out


def boundary_bins(y, p, eligible, scale, edges):
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError('Positive training-only cost scale required')
    edges = np.asarray(edges, float)
    if not np.all(np.diff(edges) > 0):
        raise ValueError('Fixed ordered boundaries required')
    score = (p[:, 1]-.02*p[:, 2])/scale
    index = np.searchsorted(edges, score, side='right')
    result = []
    for i in range(len(edges)+1):
        at = eligible & (index == i)
        result.append(dict(bin=i, lower=None if i == 0 else float(edges[i-1]),
            upper=None if i == len(edges) else float(edges[i]),
            rows=int(at.sum()), known_rows=int((at & np.isfinite(y).all(1)).sum()),
            unknown_rows=int((at & ~np.isfinite(y).all(1)).sum())))
    assert sum(r['rows'] for r in result) == int(eligible.sum())
    return index, result


def diagnostic(a, meta, cfg):
    required = {'ids', 'targets', 'envelope', 'moving', 'support', 'recording', 'frame'}
    for arm in cfg['arms']:
        required.update((arm+'_pred', arm+'_selected', arm+'_matched'))
    if set(a) != required:
        raise ValueError('Frozen packet schema mismatch')
    n = len(a['ids']); y = a['targets']; known = np.isfinite(y).all(1)
    assert n > 0 and len(np.unique(a['ids'])) == n
    assert set(meta['source_roles']['producer_sites']).isdisjoint({meta['site'], meta['train_site']})
    assert set(meta['source_roles']['controller_sites']).isdisjoint({meta['site'], meta['train_site']})
    if meta['phase'] == 'internal_transfer':
        assert meta['site'] != meta['train_site']
        assert not {meta['site'], meta['train_site']} & set(meta['outer_held_sites'])
    else:
        assert meta['phase'] == 'fitting' and meta['site'] == meta['train_site']
    eligible = a['moving'] & a['support']
    out = dict(meta=meta, rows=n, known_rows=int(known.sum()), unknown_rows=int((~known).sum()), arms={})
    for arm in cfg['arms']:
        p = a[arm+'_pred']
        assert p.shape == (n, 5) and (p[:, :2].sum(1) <= a['envelope']+1e-5).all()
        assert (p[:, 3] <= p[:, 2]+1e-8).all() and (p[:, 4] <= p[:, 1]+1e-8).all()
        selected, matched = a[arm+'_selected'], a[arm+'_matched']
        assert not (matched & ~selected).any() and not (selected & ~eligible).any()
        z = p[:, 1]-.02*p[:, 2]
        assert not (selected & ((p[:, 0]-p[:, 1] <= 0) | (z > 0) | (p[:, 4]-.02*p[:, 3] > 0))).any()
        scopes = dict(eligible=eligible, selected=selected, matched=matched,
            near_boundary=eligible & (np.abs(z/meta['cost_scale']) <= cfg['boundary_half_width']))
        index, bins = boundary_bins(y, p, eligible, meta['cost_scale'], cfg['bin_edges'])
        entry = dict(scopes={}, bins=bins)
        for name, at in scopes.items():
            entry['scopes'][name] = {event: decompose(y, p, at, meta['tail_cuts'][event], easy=event == 'easy')
                                    for event in ('all', 'easy')}
        entry['bin_excess'] = [decompose(y, p, eligible & (index == i), meta['tail_cuts']['all'])
                               for i in range(len(bins))]
        out['arms'][arm] = entry
    keys = np.rec.fromarrays([a['recording'], a['frame']], names='recording,frame')
    _, inv = np.unique(keys, return_inverse=True)
    counts = [np.bincount(inv, weights=a[arm+'_matched'].astype(int)) for arm in cfg['arms']]
    np.testing.assert_array_equal(*counts)
    out['query_count'] = len(counts[0])
    out['parameter_updates'] = 0
    out['policy_changed'] = False
    return out
