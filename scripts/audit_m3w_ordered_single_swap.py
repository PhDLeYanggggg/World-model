"""Compare same-query and ordered one-for-one capacity at equal recording count."""
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_atomic_swap as exp
from src.evaluation import m3w_support_acquisition as oracle

run, np = exp.run, exp.np


def witnesses(pool):
    p = pool
    aq, dq = p['query'], p['selected_query']
    count = np.bincount(dq, minlength=len(p['cap']))
    spare = p['cap'] - count
    base = p['selected_cost'].sum(1)
    limits = np.array([0., 0., .02*p['full_easy']])
    result = {}
    for start in range(0, len(aq), 128):
        rows = slice(start, start+128)
        a, d = aq[rows, None], dq[None]
        same = a == d
        later = (a < d) & (spare[a] > 0)
        lower = p['gain'][rows, None] + p['deletion_gain'][None, :]
        ok = (same | later) & (lower >= 0.)
        for k in range(3):
            ok &= (base[k] + p['cost'][k, rows, None] - p['selected_cost'][k, None, :]) <= limits[k]
        for name, allowed in (('same_query', same), ('later_query', later)):
            if name not in result:
                hits = np.argwhere(ok & allowed)
                if len(hits):
                    i, j = hits[0]
                    result[name] = start+int(i), int(j)
        if len(result) == 2:
            break
    return result


def main():
    capacity = exp.PUBLIC/'capacity.json'
    if run.read(exp.PUBLIC/'capacity_execution.json')['artifact']['sha256'] != run.sha(capacity):
        raise ValueError('Require verified atomic capacity')
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    records = exp.h.previous.prior.blocks(d, lab['frames'])
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    cells, controls, ranks = exp.h.cost.banks()
    cuts, r = run.cuts_by_source(), run.parent
    counts = Counter()
    for cell in cells:
        m = exp.h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        for rec, (ix, g) in records.items():
            site = str(d['sites'][ix[0]])
            e, h = cuts[site]['easy'], cuts[site]['hard']
            c, b, dc = m.floor[ix], m.bank[ix], m.direct[ix]
            paths = {r.CV: c, 'direct': run.model.selected_paths(dc, b, c)}
            args = y[ix], valid[ix], d['queries'][ix], rec
            before = r.readout.score_recording(paths, {}, *args, easy_cut=e, hard_cut=h)
            if before['metrics']['direct']['risk']['completion_easy_risk_upper'] is not None:
                continue
            counts['missing_reference_views'] += 1
            counts['zero_direct_views'] += int(not (dc >= 0).any())
            pool = oracle.pool(c, b, y[ix], valid[ix], e, dc, g, np.ones(b.shape[:2], bool))
            found = witnesses(pool)
            counts['same_query'] += int('same_query' in found)
            counts['later_query'] += int('later_query' in found)
            counts['later_only'] += int('later_query' in found and 'same_query' not in found)
            counts['ordered_one_pair'] += int(bool(found))
            for name, (i, j) in found.items():
                z = dc.copy()
                z[pool['row'][i]] = pool['action'][i]
                z[pool['selected'][j]] = -1
                out = r.readout.score_recording({**paths, 'witness': run.model.selected_paths(z, b, c)},
                    {'pair': ('direct', 'witness', r.CV)}, *args, easy_cut=e, hard_cut=h)
                paired = out['comparisons']['pair']
                if ((z >= 0).sum() != (dc >= 0).sum()
                        or any((z[q] >= 0).sum() > min(2, len(q)-1) for q in g)
                        or not out['metrics']['witness']['risk']['supported']
                        or paired['known_delta']-paired['unknown_radius'] < -1e-8):
                    raise ValueError(f'{name} witness failed the original scoring contract')
        del m
    old = run.read(capacity)['scope']['missing_reference']
    if counts['missing_reference_views'] != old['views'] or counts['same_query'] != old['with_feasible_pair']:
        raise ValueError('Same-query enumeration failed independent crosscheck')
    run.once(exp.PUBLIC/'ordered_capacity.json', run.stable(dict(capacity_sha256=run.sha(capacity),
        views=dict(counts), independent_confirmation=False, thresholds_changed=False,
        oracle_choices_exported=False, result_source='fresh_label_aware_capacity')))


if __name__ == '__main__':
    main()
