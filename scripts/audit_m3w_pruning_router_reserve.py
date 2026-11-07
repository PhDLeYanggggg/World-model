"""Hold routed thresholds fixed to isolate the support reserve."""
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_pruning_router as study

run, np, cost = study.run, study.np, study.cost


def main():
    analysis = study.PUBLIC/'analysis.json'
    if run.read(study.PUBLIC/'replay_execution.json')['artifact']['sha256'] != run.sha(analysis):
        raise ValueError('Require exact replay')
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    records = study.prior.blocks(d, lab['frames'])
    y, valid = lab['target_eval'] - d['origin'][:, None], lab['valid']
    parent, controls, ranks = cost.banks()
    frozen = {c['seed']: c['folds'] for c in run.read(study.PUBLIC/'freeze.json')['cells']}
    old = {c['seed']: {f['identity']['held']: f for f in c['folds']}
           for c in run.read(study.teacher.PUBLIC/'freeze.json')['cells']}
    cuts, p = run.cuts_by_source(), run.parent
    totals = {a: Counter() for a in ('route', 'fixed_reserve')}
    for cell in parent:
        m = study.base.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        seed = m.seed
        for f in frozen[seed]:
            s = f['identity']['held']
            at = m.ix[s]
            (r, a), _ = f['calibration'][0]
            with np.load(run.checked(f['cache']), allow_pickle=False) as z:
                ch, prob = z['choices'], z['probabilities']
            with np.load(run.checked(old[seed][s]['cache']), allow_pickle=False) as z:
                imit = z['coefficients']
            for ix, g in m.groups[s]:
                j = np.searchsorted(at, ix)
                dc, i, _, x, groups, support, _ = study.inputs(m, ix, g, m.co[s])
                q = np.asarray([prob[j[i[v[0]]]] for v in groups])
                score = study.teacher.model.scores(x, imit if len(imit) else None)[0]
                paths = {p.CV: m.floor[ix]}
                for label, reserve in (('route', False), ('fixed_reserve', True)):
                    z = dc.copy()
                    z[i[study.model.choose(groups, score, support, q, a, r, reserve)]] = -1
                    if not reserve:
                        np.testing.assert_array_equal(z, ch[3, j])
                    paths[label] = run.model.selected_paths(z, m.bank[ix], m.floor[ix])
                    totals[label]['deletions'] += int((z != dc).sum())
                res = p.readout.score_recording(paths, {}, y[ix], valid[ix], d['queries'][ix],
                    str(d['recordings'][ix[0]]), easy_cut=cuts[s]['easy'], hard_cut=cuts[s]['hard'])
                for label, t in totals.items():
                    risk = res['metrics'][label]['risk']
                    t['passes'] += int(risk['supported'])
                    t['undefined_easy'] += int(risk['completion_easy_risk_upper'] is None)
                    t['easy_violations'] += int(risk['known_easy_net_degradation'] is not None and
                                                risk['known_easy_net_degradation'] > .02)
        del m
    registered = run.read(analysis)['risk']
    if (totals['route']['passes'], totals['route']['deletions']) != (
            registered['route']['passes'], registered['direct']['switches']-registered['route']['switches']):
        raise ValueError('Require exact frozen route')
    run.once(study.PUBLIC/'reserve_diagnostic.json', run.stable(dict(
        analysis_sha256=run.sha(analysis), identical_route_and_action_cuts=True,
        diagnostic_only=True, choices_changed=False, outcomes_used_for_thresholds=False,
        arms={k: dict(v) for k, v in totals.items()})))


if __name__ == '__main__':
    main()
