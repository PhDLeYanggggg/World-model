"""Measure the best-case radius needed to tie OLS4 on exposed Moving sources."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_record_peak_allocation as audit

np, old = audit.np, audit.old


def minimum_peak(cap, weight, gap):
    """Minimum total peak mass for a single-reference, unrestricted utility tie."""
    if gap <= 0.:
        return 0.
    lengths = np.diff(np.sort(cap, axis=1), axis=1, prepend=0.).ravel()
    slopes = (weight[:, None] * np.arange(cap.shape[1], 0, -1)).ravel()
    keep = lengths > 0.
    lengths, slopes = lengths[keep], slopes[keep]
    order = np.argsort(-slopes, kind='stable')
    z, s = lengths[order], slopes[order]
    gains = np.cumsum(z * s)
    j = int(np.searchsorted(gains, gap))
    if j == len(gains):
        return None
    return float(z[:j].sum() + (gap - (gains[j-1] if j else 0.)) / s[j])


def diagnose(cv, ols, y, valid, query, peak):
    y = np.where(valid[..., None], y, 0.)
    c, o, d = (np.linalg.norm(a-b, axis=-1) for a, b in ((cv, y), (ols, y), (ols, cv)))
    w = audit.old.weight(query)
    known_gap = float(w @ np.where(valid, c-o, 0.).sum(1))
    missing_gap = float(w @ np.where(valid, 0., d).sum(1))
    ref = float(w @ np.where(valid, c, 0.).sum(1))
    needed = {'known_only': minimum_peak(np.where(valid, c, 0.), w, known_gap),
              'with_missing_debit': minimum_peak(np.where(valid, c, d), w, known_gap+missing_gap)}
    return dict(required_peak=needed, required_multiple={k: v/peak if peak > 0. and v is not None else None
                 for k, v in needed.items()}, gap_known_pp=100.*known_gap/ref, gap_missing_pp=100.*missing_gap/ref)


def main():
    reg, cells, _ = audit.register()
    parent = audit.OUT / 'readout.json'
    p = old.read(parent)
    if (p['registration'] != reg or not p['original_2pct'] or not p['hindsight_only']
            or any(p[k] for k in ('a_labels_opened', 'confirmation_access', 'independent_confirmation'))):
        raise ValueError('Require the checked Moving development allocation')
    d = audit.previous.base.fit_population(lambda **_: None)
    linear = audit.previous.prior.study.motion.bank(d['history'])[:, :2]
    rows = []
    lookup = {(z['identity']['fold'], z['identity']['seed'], z['identity']['recording']): z for z in p['rows']}
    for cell in cells:
        i = cell['identity']
        ix, h = audit.previous.base.subset(d, i['held'])
        held = linear[ix]
        for r in np.unique(h['recording']):
            at = h['recording'] == r
            old_row = lookup[i['fold'], i['seed'], str(r)]
            peak = old_row['budget']['peak']
            cv, ols = held[at].transpose(1, 0, 2, 3)
            result = diagnose(cv, ols, *(h[k][at] for k in ('target', 'valid', 'query')), peak)
            needed = result['required_peak']['with_missing_debit']
            upper = old_row['arbitrary']['record'][1]
            if ((needed is None or needed > peak + 1e-7*(1.+peak)) and upper > 1e-6
                    or needed is not None and needed < peak - 1e-7*(1.+peak) and upper < -1e-6):
                raise ValueError('Break-even budget contradicts pooled upper bound')
            rows.append(dict(identity=old_row['identity'], **result))
    if len(rows) != len(lookup):
        raise ValueError('Roster changed')
    summary = {}
    for k in ('known_only', 'with_missing_debit'):
        z = np.array([r['required_multiple'][k] for r in rows if r['required_multiple'][k] is not None])
        summary[k] = dict(finite_nonzero_budget_views=len(z), ties_within_original_budget=int((z <= 1.).sum()),
            median_multiple=float(np.median(z)), q25_q75_multiple=np.quantile(z, [.25, .75]).tolist())
    summary['median_gap_pp'] = {k: float(np.median([r[k] for r in rows])) for k in ('gap_known_pp', 'gap_missing_pp')}
    old.once(audit.OUT / 'break_even.json', old.stable(dict(parent=old.artifact(parent), code=old.artifact(Path(__file__)),
        summary=summary, rows=rows, ignores_risk_and_count=True, hindsight_only=True, original_2pct=True,
        a_labels_opened=False, confirmation_access=False, independent_confirmation=False)))
    print(summary, flush=True)


if __name__ == '__main__':
    main()
