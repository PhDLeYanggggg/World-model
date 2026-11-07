"""Check whether early weighting loses existing first-query hits."""
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_early_reference_listwise as study

run, np, cost = study.run, study.np, study.cost


def main():
    a = study.PUBLIC/'analysis.json'
    if run.read(study.PUBLIC/'replay_execution.json')['artifact']['sha256'] != run.sha(a):
        raise ValueError('Require exact readout replay')
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    records = study.previous.prior.blocks(d, lab['frames'])
    y, valid = lab['target_eval'] - d['origin'][:, None], lab['valid']
    parent, controls, ranks = cost.banks()
    frozen = {c['seed']: c for c in run.read(study.PUBLIC/'freeze.json')['cells']}
    rows = []
    for cell in parent:
        m = study.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        study.score(m, records, frozen[m.seed], y, valid, rows)
        del m
    summary = {}
    for key, subset in (('all', rows), ('missing', [r for r in rows if r['missing']]),
                        ('missing_with_easy', [r for r in rows if r['missing'] and r['first_available']])):
        summary[key] = dict(views=len(subset),
            regained=sum(int(r['first'][3] and not r['first'][2]) for r in subset),
            lost=sum(int(r['first'][2] and not r['first'][3]) for r in subset),
            both_hit=sum(int(r['first'][2] and r['first'][3]) for r in subset))
    direction = Counter()
    for s in sorted({r['source'] for r in rows}):
        vals = np.array([[int(r['first'][3])-int(r['first'][2]) for r in rows
                         if r['source'] == s and r['seed'] == seed] for seed in run.kin.prior.SEEDS])
        z = vals.mean()
        direction['improved' if z > 0. else 'worse' if z < 0. else 'tied'] += 1
    if summary['all']['views'] != 102 or summary['missing_with_easy']['views'] != 28:
        raise ValueError('Require original cohorts')
    run.once(study.PUBLIC/'transition_diagnostic.json', run.stable(dict(analysis_sha256=run.sha(a),
        first_query_transition=summary, source_direction=dict(direction), retrospective=True,
        decisions_changed=False, independent_outcomes_opened=False)))


if __name__ == '__main__':
    main()
