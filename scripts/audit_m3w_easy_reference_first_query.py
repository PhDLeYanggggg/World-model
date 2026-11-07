"""Separate absent easy reference from first-query ranking failures."""
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_easy_reference_rank as study

run, np, cost = study.run, study.np, study.cost


def main():
    analysis = study.PUBLIC/'analysis.json'
    if run.read(study.PUBLIC/'replay_execution.json')['artifact']['sha256'] != run.sha(analysis):
        raise ValueError('Require exact component replay')
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    records = study.prior.blocks(d, lab['frames'])
    y, valid = lab['target_eval'] - d['origin'][:, None], lab['valid']
    parent, controls, ranks = cost.banks()
    frozen = {c['seed']: c['folds'] for c in run.read(study.PUBLIC/'freeze.json')['cells']}
    cuts = run.cuts_by_source()
    result = {k: Counter() for k in study.ARMS}
    views = possible = 0
    for cell in parent:
        m = study.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        scores = np.empty((len(m.floor), len(study.ARMS)), np.float32)
        mask = np.zeros(len(m.floor), bool)
        for f in frozen[m.seed]:
            at = m.ix[f['identity']['held']]
            with np.load(run.checked(f['cache']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
                scores[at], mask[at] = z['scores'], z['eligible']
        for ix, g in records.values():
            c = m.floor[ix]
            s = str(d['sites'][ix[0]])
            z = study.prune.model.moments(c, c, y[ix], valid[ix], cuts[s]['easy'])
            easy = z['ref_easy'] > 0.
            if easy[m.direct[ix] >= 0].any():
                continue
            q = next(q[mask[ix[q]]] for q in g if mask[ix[q]].any())
            views += 1
            available = easy[q].any()
            possible += int(available)
            first = q[scores[ix[q]].argmax(0)]
            for name, i in zip(study.ARMS, first):
                reason = ('hit' if easy[i] else 'unknown_outcome' if not z['known'][i]
                          else 'zero_floor_error' if z['base'][i] == 0. else 'above_easy_cutoff')
                result[name][reason] += 1
                result[name]['avoidable_rank_miss'] += int(available and not easy[i])
        del m
    a = run.read(analysis)
    if views != 35 or any(a['missing_first_query_hits'][k] != v['hit'] for k, v in result.items()):
        raise ValueError('Require original first-query cohort and hit counts')
    run.once(study.PUBLIC/'first_query_diagnostic.json', run.stable(dict(
        analysis_sha256=run.sha(analysis), missing_views=views, any_observed_easy=possible,
        absent_easy=views-possible, arms={k: dict(v) for k, v in result.items()},
        retrospective=True, choices_changed=False, independent_outcomes_opened=False)))


if __name__ == '__main__':
    main()
