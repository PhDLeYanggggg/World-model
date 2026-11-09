"""Attribute a fixed-budget pair contrast without altering the risk or score."""

import numpy as np

from src.evaluation.m3w_opendd_location import paired


def decompose(treatment, control, valid, target, weight, mass, a_rows, b_rows):
    """Keep the full-roster CV denominator in all additive contributions."""
    if (a_rows.shape != b_rows.shape or a_rows.ndim != 2 or a_rows.shape[1] != 2
            or len(np.unique(a_rows)) != a_rows.size or len(np.unique(b_rows)) != b_rows.size
            or not 0. < mass < np.inf):
        raise ValueError('Expected two distinct edits per recording and positive CV mass')
    known, lower = paired(control, treatment, target, valid)
    debit = np.where(valid, 0., np.linalg.norm(treatment-control, axis=-1))
    scale = 100.*weight/mass
    available = known.sum(1)*scale
    penalty = debit.sum(1)*scale
    changed = (treatment != control).any(axis=-1)
    categories = {k: dict(recordings=0, known_pp=0., missing_debit_pp=0.) for k in
                  ('identical', 'all_disagreements_observed', 'some_disagreements_missing')}
    choice = {k: 0 for k in ('identical', 'same_recipients_different_paths', 'different_recipients')}
    covered = 0
    for a, b in zip(a_rows, b_rows, strict=True):
        rows = np.union1d(a, b)
        differs = changed[rows]
        if not differs.any():
            kind = 'identical'
            choice['identical'] += 1
        else:
            kind = ('some_disagreements_missing' if (differs & ~valid[rows]).any()
                    else 'all_disagreements_observed')
            choice['same_recipients_different_paths' if len(rows) == 2 else 'different_recipients'] += 1
        covered += np.count_nonzero(differs)
        block = categories[kind]
        block['recordings'] += 1
        block['known_pp'] += float(available[rows].sum())
        block['missing_debit_pp'] += float(penalty[rows].sum())
    if covered != np.count_nonzero(changed):
        raise ValueError('Disagreement outside the registered edits or overlapping recording rows')
    lo = float(lower.sum(1) @ scale)
    k, d = float(available.sum()), float(penalty.sum())
    np.testing.assert_allclose([k-d, sum(r['known_pp']-r['missing_debit_pp'] for r in categories.values())],
                               lo, rtol=0., atol=1e-10)
    return dict(known_pp=k, missing_debit_pp=d, lower_pp=lo, upper_pp=k+d,
                categories=categories, choice=choice)


def gate_transitions(treatment, control):
    def state(arm):
        return np.select([arm['missing'], ~arm['risk'], ~arm['dual']],
                         ['no_easy_support', 'risk_failed', 'reference_gain_failed'],
                         default='dual_pass')

    a, b = state(treatment), state(control)
    if a.shape != b.shape:
        raise ValueError('Expected matched recording risk outcomes')
    labels = ('no_easy_support', 'risk_failed', 'reference_gain_failed', 'dual_pass')
    return {ref: {new: int(((b == ref) & (a == new)).sum()) for new in labels} for ref in labels}
