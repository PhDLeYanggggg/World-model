"""Explain the supported early winners that did not translate to policy passes."""
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_family_supported_gain as study

run, np, cost = study.run, study.np, study.cost


def main():
    out = study.PUBLIC/'analysis.json'
    if run.read(study.PUBLIC/'replay_execution.json')['artifact']['sha256'] != run.sha(out):
        raise ValueError('Require exact replay of the registered experiment')
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    records = study.prior.blocks(d, lab['frames'])
    y, valid = lab['target_eval'] - d['origin'][:, None], lab['valid']
    parent, controls, old_rank = cost.banks()
    saved = {c['seed']: c for c in run.read(study.PUBLIC/'freeze.json')['cells']}
    old = {c['seed']: {f['identity']['held']: f for f in c['folds']}
           for c in run.read(cost.PUBLIC/'freeze.json')['cells']}
    cuts, p = run.cuts_by_source(), run.parent
    reasons, hypothetical, families = Counter(), Counter(), Counter()
    for cell in parent:
        m = study.portfolio.Seed(d, refs, records, cell, controls, old_rank)
        seed = m.seed
        for f in saved[seed]['folds']:
            site = f['identity']['held']
            at, co, rank = m.ix[site], m.co[site], m.rank(site)
            _, choice_coef = study.history.load_old(old[seed][site], d, at)
            with np.load(run.checked(f['cache']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
                ch, new = z['choices'], z['ranking']
            e, h = cuts[site]['easy'], cuts[site]['hard']
            for ix, groups in m.groups[site]:
                pre = []
                for q in groups:
                    if (m.direct[ix[q]] >= 0).any():
                        break
                    pre.append(q)
                _, _, examples = study.proposals(m, ix, pre, co, rank, new)
                examples = examples[:2]
                c, b = m.floor[ix], m.bank[ix]
                _, atoms, _ = study.anchor.model.labels(c, b, y[ix], valid[ix], e)
                found = None
                for t, (sample, target) in enumerate(zip(examples, cost.model.targets(atoms, m.scale[ix], examples))):
                    q, i, j, _ = sample
                    conf, row, family = cost.model.winners(sample, choice_coef)[1]
                    if (target[(i == row) & (j == family), 4] > 0.).any():
                        found = t, conf, row, family
                        break
                if found is None:
                    continue
                t, conf, row, family = found
                j = np.searchsorted(at, ix)
                direct, choice = ch[0, j], ch[2, j]
                threshold = f['threshold']
                if choice[row] == family:
                    reasons['selected_supported_winner'] += 1
                elif threshold is None:
                    reasons['no_donor_threshold'] += 1
                elif conf < threshold:
                    reasons['score_below_donor_threshold'] += 1
                elif (choice != direct).any():
                    reasons['different_opening_selected'] += 1
                else:
                    raise ValueError('Threshold replay cannot explain the frozen choice')
                families[run.NAMES[family]] += 1
                hypothetical['first_eligible' if t == 0 else 'second_eligible'] += 1
                a = direct.copy()
                a[row] = family
                paths = {k: run.model.selected_paths(v, b, c) for k, v in
                         (('direct', direct), ('frozen', choice), ('add_earliest_useful', a))}
                metrics = p.readout.score_recording({p.CV: c, **paths}, {}, y[ix], valid[ix],
                    d['queries'][ix], str(d['recordings'][ix[0]]), easy_cut=e, hard_cut=h)['metrics']
                success = {k: m['risk']['supported'] for k, m in metrics.items() if k in paths}
                for k, flag in success.items():
                    hypothetical[k+'_complete_pass'] += int(flag)
                hypothetical['new_pass_vs_direct'] += int(success['add_earliest_useful'] and not success['direct'])
        del m
    total = sum(reasons.values())
    if total != run.read(out)['prefix_first_two_all_recall']['family_gain_winner']:
        raise ValueError('Require exact previously measured useful-winner cohort')
    run.once(study.PUBLIC/'opening_diagnostic.json', run.stable(dict(
        analysis_sha256=run.sha(out), views=total, reasons=dict(reasons), families=dict(families),
        paired_full_record=dict(hypothetical), retrospective_label_aware_diagnostic=True,
        decisions_or_protocol_changed=False, independent_outcomes_opened=False)))


if __name__ == '__main__':
    main()
