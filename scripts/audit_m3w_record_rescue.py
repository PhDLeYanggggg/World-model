"""Explain changed risk states and bound ordered repair of the same insertion."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_record_rescue as exp
from src.evaluation import m3w_support_acquisition as oracle

run, np = exp.run, exp.np


def failures(risk):
    keys = ('completion_all_risk_upper', 'completion_easy_risk_upper', 'known_easy_net_degradation')
    return {k: risk[k] for k in keys if risk[k] is None or risk[k] > .02}


def main():
    analysis, freeze = (exp.PUBLIC/n for n in ('analysis.json', 'freeze.json'))
    for file, receipt in ((analysis, 'replay'), (freeze, 'fit')):
        if run.read(exp.PUBLIC/f'{receipt}_execution.json')['artifact']['sha256'] != run.sha(file):
            raise ValueError('Require verified frozen experiment')
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    records = exp.h.previous.prior.blocks(d, lab['frames'])
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    cells, controls, ranks = exp.h.cost.banks()
    frozen = {c['seed']: c for c in run.read(freeze)['cells']}
    cuts, readout, cv = run.cuts_by_source(), run.parent.readout, run.parent.CV
    summary = {a: dict(gained_reference=[], lost_pass=[]) for a in exp.model.ARMS}
    for cell in cells:
        m = exp.h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        choice = np.full((len(exp.ARMS), len(m.floor)), -1, np.int16)
        for f in frozen[m.seed]['folds']:
            at = m.ix[f['identity']['held']]
            with np.load(run.checked(f['cache']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
                choice[:, at] = z['choices']
        np.testing.assert_array_equal(choice[0], m.direct)
        for rec, (ix, g) in records.items():
            z, c, b = choice[:, ix], m.floor[ix], m.bank[ix]
            if not (z[1:] != z[0]).any():
                continue
            site = str(d['sites'][ix[0]])
            e, h = cuts[site]['easy'], cuts[site]['hard']
            args = y[ix], valid[ix], d['queries'][ix], rec
            paths = {a: run.model.selected_paths(v, b, c) for a, v in zip(exp.ARMS, z)}
            scored = readout.score_recording({cv: c, **paths}, {a: ('direct', a, cv)
                for a in exp.model.ARMS}, *args, easy_cut=e, hard_cut=h)
            before = scored['metrics']['direct']['risk']
            for arm, selected in zip(exp.model.ARMS, z[1:]):
                after = scored['metrics'][arm]['risk']
                pair = scored['comparisons'][arm]
                packet = dict(before_violations=failures(before), after_violations=failures(after),
                    paired_lower_mass=float(pair['known_delta']-pair['unknown_radius']))
                if before['supported'] and not after['supported']:
                    summary[arm]['lost_pass'].append(packet)
                if (before['completion_easy_risk_upper'] is None
                        and after['completion_easy_risk_upper'] is not None):
                    i = np.flatnonzero(selected != z[0])
                    if len(i) != 1 or selected[i[0]] < 0:
                        raise ValueError('Expected one insertion')
                    allowed = np.zeros(b.shape[:2], bool)
                    allowed[i, selected[i]] = True
                    p = oracle.pool(c, b, y[ix], valid[ix], e, z[0], g, allowed)
                    sol, status, _ = oracle.solve(p, 'shortlist_joint')
                    packet['same_insertion_ordered_deletion_status'] = status
                    if sol is not None:
                        drop, add = sol
                        final = selected.copy()
                        final[p['selected'][drop]] = -1
                        scored_final = readout.score_recording({cv: c, 'direct': paths['direct'],
                            'edit': run.model.selected_paths(final, b, c)}, {'pair': ('direct', 'edit', cv)},
                            *args, easy_cut=e, hard_cut=h)
                        pair_final = scored_final['comparisons']['pair']
                        if ((p['selected_query'][drop] < p['query'][add]).any()
                                or any((final[q] >= 0).sum() > min(2, len(q)-1) for q in g)
                                or not scored_final['metrics']['edit']['risk']['supported']
                                or pair_final['known_delta']-pair_final['unknown_radius'] < -1e-8):
                            raise ValueError('Oracle ordered repair failed exact postcheck')
                        packet['verified_removal_count'] = int(drop.sum())
                    summary[arm]['gained_reference'].append(packet)
        del m
    expected = run.read(analysis)['risk']
    for arm, rows in summary.items():
        if (len(rows['lost_pass']) != expected[arm]['lost_passes']
                or len(rows['gained_reference']) != expected['direct']['undefined_easy']-expected[arm]['undefined_easy']):
            raise ValueError('Audit disagrees with frozen aggregate transitions')
    run.once(exp.PUBLIC/'transition_diagnostic.json', run.stable(dict(
        analysis_sha256=run.sha(analysis), freeze_sha256=run.sha(freeze), transitions=summary,
        thresholds_changed=False, oracle_choices_exported=False, independent_confirmation=False)))


if __name__ == '__main__':
    main()
