"""Explain failed donor calibration without changing any held decision."""
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_record_pruning as study

run, np, model = study.run, study.np, study.model


def main():
    analysis = study.PUBLIC/'analysis.json'
    if run.read(study.PUBLIC/'replay_execution.json')['artifact']['sha256'] != run.sha(analysis):
        raise ValueError('Require exact score replay')
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    records = study.prior.blocks(d, lab['frames'])
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    cuts = run.cuts_by_source()
    parent, controls, ranks = study.cost.banks()
    frozen = {c['seed']: c['folds'] for c in run.read(study.PUBLIC/'freeze.json')['cells']}
    totals = Counter()
    thresholds = {str(t): Counter() for t in model.THRESHOLDS}
    for cell in parent:
        m = study.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        for f in frozen[m.seed]:
            ident = f['identity']
            co = m.co[ident['held']]
            with np.load(run.checked(f['cache']), allow_pickle=False) as z:
                coeff = z['coefficients']
            donors = []
            for s in ident['cal']:
                for ix, g in m.groups[s]:
                    _, i, a, x, group, _ = study.frozen_direct(m, ix, g, co)
                    v, full, w = study.targets(m, ix, i, a, y, valid, cuts[s]['easy'])
                    donors.append((group, v, model.deletion_rank(x, coeff), full, w))
            if model.calibrate(donors) != f['threshold']:
                raise ValueError('Require exact donor threshold replay')
            with_recovery, preserve_passes, preserve_easy, preserve_gain = False, False, False, False
            for t in model.THRESHOLDS:
                recovered = lost = easy = dropped = 0
                delta = 0.
                for g, v, score, full, w in donors:
                    q = model.choose(g, score, t)
                    old, bad = model.observe(v, np.zeros(len(score), bool), full)
                    new, worse = model.observe(v, q, full)
                    recovered += int(new and not old)
                    lost += int(old and not new)
                    easy += int(worse and not bad)
                    dropped += int(q.sum())
                    delta += float(w[q] @ v['remove_lower'][q])
                stats = thresholds[str(t)]
                stats['folds_with_any_deletion'] += int(dropped > 0)
                stats['folds_with_net_pass_gain'] += int(recovered > lost)
                stats['folds_with_nonnegative_paired_lower_mass'] += int(delta >= 0)
                stats['folds_with_extra_easy_violation'] += int(easy > 0)
                with_recovery |= recovered > 0
                preserve_passes |= recovered > 0 and not lost
                preserve_easy |= recovered > 0 and not lost and not easy
                preserve_gain |= recovered > 0 and not lost and not easy and delta >= 0
            totals['folds'] += 1
            totals['some_threshold_recovers_a_pass'] += int(with_recovery)
            totals['also_preserves_old_passes'] += int(preserve_passes)
            totals['also_no_added_easy_violation'] += int(preserve_easy)
            totals['also_nonnegative_paired_lower_mass'] += int(preserve_gain)
        del m
    if totals['folds'] != 42 or totals['also_nonnegative_paired_lower_mass'] != 0:
        raise ValueError('Unexpected calibration state')
    run.once(study.PUBLIC/'calibration_diagnostic.json', run.stable(dict(
        analysis_sha256=run.sha(analysis), donor_only=True,
        nested_requirements=dict(totals), thresholds={t: dict(v) for t, v in thresholds.items()},
        choices_changed=False, independent_outcomes_opened=False)))


if __name__ == '__main__':
    main()
