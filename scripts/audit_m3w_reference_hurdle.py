"""Locate the lost one-insertion opportunities without changing frozen choices."""
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_reference_hurdle as exp
from scripts import run_m3w_support_acquisition as capacity

run, np = exp.run, exp.np


def feasible(problem):
    p = problem
    count = np.bincount(p['selected_query'], minlength=len(p['cap']))
    remaining = p['cap'] - count
    cost = p['selected_cost'].sum(1)[:, None] + p['cost']
    limit = np.array([0., 0., .02*p['full_easy']])[:, None]
    return ((remaining[p['query']] > 0) & (cost <= limit).all(0) & (p['gain'] >= 0.))


def main():
    analysis = exp.PUBLIC/'analysis.json'
    freeze = exp.PUBLIC/'freeze.json'
    ceiling = capacity.PUBLIC/'analysis.json'
    for result, receipt in ((analysis, exp.PUBLIC/'replay_execution.json'),
                            (freeze, exp.PUBLIC/'fit_execution.json'),
                            (ceiling, capacity.PUBLIC/'replay_execution.json')):
        if run.read(receipt)['artifact']['sha256'] != run.sha(result):
            raise ValueError('Require verified experiment parents')
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    records = exp.previous.prior.blocks(d, lab['frames'])
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    cells, controls, ranks = exp.cost.banks()
    frozen = {x['seed']: x for x in run.read(freeze)['cells']}
    cuts = run.cuts_by_source()
    count = Counter()
    by_arm = {a: Counter() for a in exp.model.ARMS}
    for cell in cells:
        m = exp.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        top = np.full((3, len(m.floor)), -1, np.int16)
        choice = np.full((4, len(m.floor)), -1, np.int16)
        thresholds = {}
        for f in frozen[m.seed]['folds']:
            site = f['identity']['held']
            at = m.ix[site]
            with np.load(run.checked(f['cache']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
                top[:, at], choice[:, at] = z['top'], z['choices']
            thresholds[site] = f['thresholds']
        np.testing.assert_array_equal(choice[0], m.direct)
        for rec, (ix, g) in records.items():
            s = str(d['sites'][ix[0]])
            e, h = cuts[s]['easy'], cuts[s]['hard']
            c, b, direct = m.floor[ix], m.bank[ix], m.direct[ix]
            paths = {k: run.model.selected_paths(z, b, c) for k, z in zip(exp.ARMS, choice[:, ix])}
            paths[run.parent.CV] = c
            out = run.parent.readout.score_recording(paths, {}, y[ix], valid[ix],
                d['queries'][ix], rec, easy_cut=e, hard_cut=h)
            risk = {k: out['metrics'][k]['risk'] for k in exp.ARMS}
            if risk['direct']['completion_easy_risk_upper'] is not None:
                continue
            pool = capacity.model.pool(c, b, y[ix], valid[ix], e, direct, g, np.ones(b.shape[:2], bool))
            ok = feasible(pool)
            q_by_row = np.empty(len(ix), int)
            for j, q in enumerate(g):
                q_by_row[q] = j
            rows = [np.flatnonzero(row >= 0) for row in top[:, ix]]
            first = [int(q_by_row[i[0]]) if len(i) else -1 for i in rows]
            if len(set(first)) != 1 or any(len(i) > 1 for i in rows):
                raise ValueError('Require a shared first eligible query')
            count['missing_views'] += 1
            count['any_feasible_insertion'] += int(ok.any())
            count['first_query_feasible_insertion'] += int((ok & (pool['query'] == first[0])).any())
            if ok.any():
                count['first_feasible_query_is_later'] += int(pool['query'][ok].min() > first[0])
            for j, arm in enumerate(exp.model.ARMS):
                row = rows[j]
                if len(row):
                    f = top[j, ix[row[0]]]
                    by_arm[arm]['probe_whole_record_feasible'] += int(
                        (ok & (pool['row'] == row[0]) & (pool['action'] == f)).any())
                inserted = (choice[j+1, ix] >= 0) & (direct < 0)
                if thresholds[s][j] is None:
                    state = 'no_calibrated_cut'
                elif not inserted.any():
                    state = 'cut_did_not_trigger'
                elif risk[arm]['completion_easy_risk_upper'] is None:
                    state = 'inserted_without_reference'
                elif not risk[arm]['supported']:
                    state = 'reference_without_recording_pass'
                else:
                    state = 'recording_pass'
                by_arm[arm][state] += 1
                if ok.any():
                    by_arm[arm]['feasible_view_'+state] += 1
        del m
    prior = run.read(ceiling)['groups']['all_missing']
    if (count['missing_views'] != prior['views']
            or count['any_feasible_insertion'] != prior['modes']['insert_only']['checked_feasible']):
        raise ValueError('Insertion capacity differs from independently checked solver')
    run.once(exp.PUBLIC/'operating_diagnostic.json', run.stable(dict(
        result_source='fresh_analysis_of_frozen_exposed_choices', analysis_sha256=run.sha(analysis),
        freeze_sha256=run.sha(freeze), ceiling_sha256=run.sha(ceiling), views=dict(count),
        by_arm={k: dict(v) for k, v in by_arm.items()}, thresholds_changed=False,
        oracle_choices_exported=False, independent_confirmation=False)))


if __name__ == '__main__':
    main()
