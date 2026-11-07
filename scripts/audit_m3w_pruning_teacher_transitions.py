"""Measure paired pass transitions and overlap with deletion-only capacity."""
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_pruning_teacher as study

run, cost = study.run, study.cost


def main():
    analysis = study.PUBLIC/'analysis.json'
    if run.read(study.PUBLIC/'replay_execution.json')['artifact']['sha256'] != run.sha(analysis):
        raise ValueError('Require replay of the frozen study')
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    records = study.prior.blocks(d, lab['frames'])
    y, valid = lab['target_eval'] - d['origin'][:, None], lab['valid']
    parent, controls, ranks = cost.banks()
    frozen = {c['seed']: c for c in run.read(study.PUBLIC/'freeze.json')['cells']}
    summary = {k: Counter() for k in ('imitation', 'gain_guard')}
    seen = feasible = 0
    for cell in parent:
        m = study.base.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        seed = m.seed
        rows = []
        study.score(m, records, frozen[seed], y, valid, rows)
        for j, (r, rec) in enumerate(zip(rows, records, strict=True)):
            oracle = run.read(study.base.PRIVATE/f'oracle_{seed}_{j:03d}.json')
            if oracle['identity'] != dict(registration=run.sha(study.base.REG), seed=seed,
                                           recording=rec, source=r['source']):
                raise ValueError('Require aligned deletion capacity receipt')
            possible = oracle['results']['non_loss']['removed'] is not None
            old = r['risk']['direct']
            seen += 1
            feasible += int(possible and not old['supported'])
            for k, s in summary.items():
                new = r['risk'][k]
                active = new['selected'] < old['selected']
                s['changed_views'] += int(active)
                s['deletions'] += old['selected'] - new['selected']
                s['new_passes'] += int(new['supported'] and not old['supported'])
                s['lost_passes'] += int(old['supported'] and not new['supported'])
                s['lost_easy_reference'] += int(old['completion_easy_risk_upper'] is not None and
                                                 new['completion_easy_risk_upper'] is None)
                s['changed_feasible_not_already_passing'] += int(active and possible and not old['supported'])
                s['changed_without_nonloss_witness'] += int(active and not possible)
                s['changed_existing_passes'] += int(active and old['supported'])
        del m
    if (seen, feasible) != (102, 28):
        raise ValueError('Require exact cohort and capacity recovery count')
    result = run.read(analysis)
    for k, v in summary.items():
        if result['risk']['direct']['passes'] + v['new_passes'] - v['lost_passes'] != result['risk'][k]['passes']:
            raise ValueError('Pass transitions must match the registered readout')
    run.once(study.PUBLIC/'transition_diagnostic.json', run.stable(dict(
        analysis_sha256=run.sha(analysis), capacity_analysis_sha256=run.sha(study.base.PUBLIC/'analysis.json'),
        views=seen, new_nonloss_hindsight_capacity=feasible, arms={k: dict(v) for k, v in summary.items()},
        retrospective=True, decisions_changed=False, independent_outcomes_opened=False)))


if __name__ == '__main__':
    main()
