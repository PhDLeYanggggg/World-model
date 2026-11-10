"""Matched pNEUMA slate metrics. No source I/O or threshold fitting."""

import numpy as np

from src.evaluation import m3w_songdo_matched_readout as common
from src.evaluation.m3w_opendd_location import paired
from src.evaluation.m3w_selection_forecast_readout import fraction, policy
from src.evaluation.m3w_sparse_pair_bounds import compare as pair_bounds


ARMS, SEEDS = common.ARMS, common.SEEDS


def paths(cv, offsets, proposals):
    if not np.array_equal(offsets, proposals['offsets']):
        raise ValueError('Proposal offsets changed')
    nq = len(offsets)-1
    radius = proposals['radius']
    if radius.shape != (nq,) or not np.isfinite(radius).all() or (radius <= 0.).any():
        raise ValueError('Expected positive radius in every matched query')
    result, scores = {}, {}
    for arm in ARMS:
        row, path, score = (proposals[f'{arm}_{k}'] for k in ('rows', 'paths', 'score'))
        if (row.shape != (nq, 2) or not np.issubdtype(row.dtype, np.integer) or (row[:, 0] == row[:, 1]).any()
                or not ((row >= offsets[:-1, None]) & (row < offsets[1:, None])).all()
                or path.shape != (nq, 2, 12, 2) or score.shape != (nq,)
                or not np.isfinite(path).all() or not np.isfinite(score).all()):
            raise ValueError('Each arm must use two distinct legal edits per query')
        actual = np.linalg.norm(path-cv[row], axis=-1).max(-1)
        if not np.allclose(actual, radius[:, None], rtol=1e-9, atol=1e-9):
            raise ValueError('Mismatched edit radius')
        out = cv.copy()
        out[row.ravel()] = path.reshape(-1, 12, 2)
        result[arm] = out
        scores[arm] = float(score.mean())
    return result, scores


def contrast(p, ref, y, mask, weights, mass, offsets):
    known, lower = paired(ref, p, y, mask)
    _, reverse = paired(p, ref, y, mask)
    rel = [pair_bounds(p[i:j], ref[i:j], y[i:j], mask[i:j])
           for i, j in zip(offsets[:-1], offsets[1:], strict=True)]
    observed = [x['available'] for x in rel if x['available'] is not None]
    return dict(observed_pp=fraction(100.*weights @ known.sum(1), mass),
                lower_pp=fraction(100.*weights @ lower.sum(1), mass),
                upper_pp=fraction(-100.*weights @ reverse.sum(1), mass),
                relative_observed=float(np.mean(observed)) if observed else None,
                relative_lower=float(np.mean([x['lower'] for x in rel])),
                relative_upper=float(np.mean([x['upper'] for x in rel])),
                relative_observed_queries=len(observed))


def score(cv, ols, median, y, valid, offsets, proposals, *, easy_cut, hard_cut):
    if (cv.shape != ols.shape or cv.shape != median.shape or cv.shape != y.shape or cv.shape[1:] != (12, 2)
            or valid.dtype != bool or valid.shape != cv.shape[:2] or not np.isfinite(y[valid]).all()
            or any(not np.isfinite(p).all() for p in (cv, ols, median))
            or offsets.ndim != 1 or len(offsets) < 2 or not np.issubdtype(offsets.dtype, np.integer)
            or offsets[0] != 0 or offsets[-1] != len(cv) or np.any(np.diff(offsets) < 3)
            or not 0 < easy_cut <= hard_cut < np.inf):
        raise ValueError('Bad roster, targets, references or frozen risk cutoffs')
    full, coverage = common.coverage(valid, offsets)
    pred, scores = paths(cv, offsets, proposals)
    w = np.repeat(1./np.diff(offsets), np.diff(offsets))
    mass = float(w @ np.where(valid, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
    by_arm = {}
    for a, p in dict(cv=cv, ols4=ols, median4=median, **pred).items():
        absolute = common.absolute(p, y, valid, full)
        absolute.update(intervention_rate=float(np.any(p != cv, axis=(1, 2)).mean()))
        if a in ARMS:
            risk = policy(p, cv, cv, y, valid, np.repeat('member', len(y)), easy_cut=easy_cut, hard_cut=hard_cut)
            r = risk['recordings']['member']
            refs = dict(cv=contrast(p, cv, y, valid, w, mass, offsets),
                        ols4=contrast(p, ols, y, valid, w, mass, offsets))
            absolute.update(risk=r, vs_reference=refs, easy_known_gain_percent=risk['subsets']['easy']['gain_percent'],
                hard_known_gain_percent=risk['subsets']['hard']['gain_percent'],
                dual_pass=bool(r['supported'] and all(v['lower_pp'] is not None and v['lower_pp'] > 0. for v in refs.values())))
        by_arm[a] = absolute
    pair = {a: contrast(pred['joint'], pred[a], y, valid, w, mass, offsets) for a in ARMS[:-1]}
    return dict(coverage={k: int(v) for k, v in coverage.items()}, arms=by_arm, matched_contrasts=pair, scores=scores)


def date_interval(records, dates, getter, *, draws=3000, seed=1439):
    """Equal seeds within member, members within date, dates within contrast."""
    if set(records) != set(dates) or any(set(r) != {str(s) for s in SEEDS} for r in records.values()):
        raise ValueError('Need the complete matched roster and all seeds')
    date_values = {d: [] for d in sorted(set(dates.values()))}
    for name in sorted(records):
        values = [getter(records[name][str(s)]) for s in SEEDS]
        date_values[dates[name]].append(float(np.mean(values)) if all(v is not None for v in values) else None)
    by_date = {d: float(np.mean(v)) if all(x is not None for x in v) else None for d, v in date_values.items()}
    x = list(by_date.values())
    result = dict(by_date=by_date, member_count_by_date={d: len(v) for d, v in date_values.items()},
                  bootstrap_unit='acquisition_date_after_seed_then_member_average', one_city=True)
    if not x or any(v is None for v in x):
        return dict(result, point=None, ci95=None, status='required_member_not_estimable')
    if len(x) < 2:
        return dict(result, point=x[0], ci95=None, status='single_date')
    draws_value = np.random.default_rng(seed).choice(x, size=(draws, len(x))).mean(1)
    return dict(result, point=float(np.mean(x)), ci95=np.quantile(draws_value, [.025, .975]).tolist(),
                status='descriptive_date_bootstrap')


def summarize(records, dates):
    def ci(fn):
        return date_interval(records, dates, fn)

    metrics = ('observed_pp', 'lower_pp', 'upper_pp', 'relative_observed', 'relative_lower', 'relative_upper')
    absolute = ('available_ade_native', 'available_fde_native', 'available_ade_p95_native', 'intervention_rate')
    summary = dict(by_arm={a: {k: ci(lambda r: r['arms'][a][k]) for k in absolute}
                          for a in ('cv', 'ols4', 'median4', *ARMS)},
        joint_minus={a: {k: ci(lambda r: r['matched_contrasts'][a][k]) for k in metrics} for a in ARMS[:-1]})
    for a in ARMS:
        views = [r['arms'][a] for by_seed in records.values() for r in by_seed.values()]
        summary['by_arm'][a].update(
            risk=dict(views=len(views), supported=sum(v['risk']['supported'] for v in views),
                      undefined_easy=sum(v['risk']['completion_easy_risk_upper'] is None for v in views),
                      dual_pass=sum(v['dual_pass'] for v in views)),
            easy=ci(lambda r: r['arms'][a]['easy_known_gain_percent']),
            hard=ci(lambda r: r['arms'][a]['hard_known_gain_percent']),
            worst_member_seed_ade=max((v['available_ade_native'] for v in views if v['available_ade_native'] is not None),
                                     default=None),
            vs_reference={b: {k: ci(lambda r: r['arms'][a]['vs_reference'][b][k]) for k in metrics} for b in ('cv', 'ols4')})
    return summary
