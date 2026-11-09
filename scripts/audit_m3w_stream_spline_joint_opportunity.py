"""Measure how often fixed streaming slots even permit same-query coupling."""
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_stream_spline_calibration as m
from scripts import run_m3w_donor_shape_ceiling as ceiling

run, np = m.run, m.np


def main():
    reg, source, _, _, _, _, _, _ = m.register()
    fp, rp = (m.OUT / k for k in ('freeze.json', 'readout.json'))
    f, r = map(run.read, (fp, rp))
    if (f['registration'] != reg or r['freeze'] != run.sha(fp) or not f['before_held_scoring']
            or not r['original_2pct'] or r['confirmation_access'] or r['independent_confirmation']):
        raise ValueError('Require a frozen development-only readout')
    sites = m.m.load(source)
    results = []
    for c in f['cells']:
        d = sites[c['held']]['d']
        bank = m.control.parent.prior.analytic(d)
        cv = bank[0]
        with np.load(run.checked(c['paths']), allow_pickle=False) as z:
            old, new = z['anchor'], z['forced']
        rows = np.flatnonzero((old != cv).any((1, 2)))
        ids, counts = np.unique(d['query'][rows], return_counts=True)
        if (counts > 2).any():
            raise ValueError('Streaming query exceeded its two-slot budget')
        two = np.isin(d['query'][rows], ids[counts == 2])
        b = np.stack((old, new), 1)
        a = np.arange(2) < counts[:, None]
        menu = dict(row=np.full(a.shape, -1), arm=np.zeros(a.shape, int))
        menu['row'][a] = rows
        restricted = b.copy()
        blocked = rows[~two]
        restricted[blocked, 1] = restricted[blocked, 0]
        cap = {k: dict(ceiling.count(d, bank, dict(paths=p[:, None]), menu, counts, c['cut']))
               for k, p in (('double_slots_only', restricted), ('all_slots', b))}
        if any(x.get('unresolved', 0) for x in cap.values()):
            # A timeout is not evidence that no qualifying plan exists.
            print('Unresolved recording capacity', c['held'], c['seed'], cap, flush=True)
        rec = d['rec'][rows]
        paired = m.control.paired(old[rows], new[rows], d['y'][rows], d['valid'][rows])[1].sum(1)
        results.append(dict(site=c['held'], seed=c['seed'], active_queries=len(ids), double_queries=int(two.sum()//2),
            slots=len(rows), double_slots=int(two.sum()), recording_views_with_double=len(set(rec[two])),
            positive_lower_single_slots=int(((paired > 0.) & ~two).sum()),
            positive_lower_double_slots=int(((paired > 0.) & two).sum()), capacity=cap))
    summary = {k:sum(z[k] for z in results) for k in results[0] if k not in ('site', 'seed', 'capacity')}
    summary['capacity'] = {k:dict(sum((Counter(z['capacity'][k]) for z in results), Counter()))
                           for k in results[0]['capacity']}
    run.once(m.OUT/'joint_opportunity.json', run.stable(dict(parent=run.artifact(rp),
        code=run.artifact(Path(__file__)), summary=summary, rows=results,
        label_aware_only=True, confirmation_access=False, original_2pct=True)))
    print(summary, flush=True)


if __name__ == '__main__':
    main()
