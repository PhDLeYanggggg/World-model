"""Optimistic risk floor for policies that change at most one OLS4 row per query."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_motion_cost_stability as study
from src.evaluation.m3w_selection_forecast_readout import errors

np = study.np


def risk_floor(anchor_ade, cv_ade, valid, queries, subset):
    """Pretend each edited row has zero harm and allow maximum reference mass."""
    use = valid & subset
    if not np.isfinite(cv_ade[use]).all() or not np.isfinite(anchor_ade[use]).all():
        raise ValueError('Nonfinite known error')
    _, inv = np.unique(queries, return_inverse=True)
    harm = np.where(use, np.maximum(anchor_ade - cv_ade, 0.), 0.)
    save = np.zeros(inv.max() + 1) if len(inv) else np.zeros(0)
    np.maximum.at(save, inv, harm)
    denom = float(cv_ade[use].sum())
    remaining = max(0., float(harm.sum() - save.sum()))
    return dict(optimistic_harm=remaining, maximum_reference=denom,
                lower_ratio=remaining / denom if denom > 0 else None)


def main():
    read, sha, old = study.read, study.sha, study.prior.bank.old
    score, freeze = (study.OUT / f'{n}.json' for n in ('readout', 'freeze'))
    stat, local = read(score), read(study.PRIVATE / 'recording_scores.json')
    if stat['freeze'] != sha(freeze) or local['readout'] != sha(score) or stat['independent_confirmation']:
        raise ValueError('Require bound development-only readout')
    a = study.a_inputs()
    lab = old.prior.arrays(('target_eval', 'origin', 'valid'))
    truth, valid = lab['target_eval'] - lab['origin'][:, None], lab['valid']
    ols, cv, *_ = study.model.candidates(a['history'])
    oa, _ = errors(ols, truth, valid)
    ca, _ = errors(cv, truth, valid)
    cuts = read(old.PARENTS[4])['cuts']
    cut = {s: cuts[str(v['job']['producer_fold'])] for v in read(old.PARENTS[3])['forecasts']
           for s in v['job']['donor_sites']}
    recordings = a['recordings'].astype(str)
    risk = {k: [] for k in np.unique(recordings)}
    for row in local['rows']:
        risk[row['recording']].extend(row['metrics'][arm]['risk'] for arm in study.ARMS)
    by_record = []
    for rec in sorted(risk):
        ix = recordings == rec
        ca0 = ca[ix]
        q = a['queries'][ix]
        known = valid[ix].any(1)
        s = a['sites'][np.flatnonzero(ix)[0]]
        subsets = dict(all=np.ones(len(q), bool), easy=(ca0 > 0) & (ca0 <= cut[s]['easy']))
        result = {key: risk_floor(oa[ix], ca0, known, q, subset) for key, subset in subsets.items()}
        for key, value in result.items():
            lower = value['lower_ratio']
            if lower is None:
                continue
            upper = [r[f'completion_{key}_risk_upper'] for r in risk[rec]]
            if any(u is not None and u + 1e-10 < lower for u in upper):
                raise ValueError('An actual policy beat the optimistic bound')
        by_record.append(result)
    roster = {k: [r[k]['lower_ratio'] for r in by_record] for k in ('all', 'easy')}
    verdict = {k: dict(defined=sum(x is not None for x in v),
                        impossible=sum(x is not None and x > .02 for x in v),
                        highest=max(x for x in v if x is not None)) for k, v in roster.items()}
    joint = sum(any(r[k]['lower_ratio'] is not None and r[k]['lower_ratio'] > .02 for k in r) for r in by_record)
    study.once(study.OUT / 'sparse_anchor_risk.json', study.stable(dict(parent=study.artifact(score),
        code=study.artifact(Path(__file__)), recording_count=len(by_record), max_edits_per_query=1,
        risk_lower_ratios=roster, by_metric=verdict, impossible_recordings=joint,
        independent_confirmation=False, original_2pct=True, unknown_harm_assumed_zero=True)))
    print(verdict, joint)


if __name__ == '__main__':
    main()
