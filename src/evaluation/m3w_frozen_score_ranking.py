"""Tie-aware fixed-budget ranking diagnostics on already evaluated proposals."""

from collections import Counter
from math import fsum, isfinite

RISK_NAMES = ('completion_all_risk_upper', 'completion_easy_risk_upper', 'known_easy_net_degradation')


def undefined(c):
    return any(v is None for v in c['risk'])


def metrics(replay):
    def risk(c):
        return replay.ratio_state(c['risk'])

    def violation(c):
        return any(v is not None and v > replay.RISK_LIMIT for v in c['risk'])

    def dual(c):
        return risk(c) and all(v is not None and v > 0. for v in c['gains'])

    return dict(dual=dual, risk_pass=risk, undefined_risk=undefined, measured_violation=violation)


def failure_reason(c, checks):
    if checks['risk_pass'](c):
        return 'supported'
    if undefined(c):
        return 'undefined_and_violation' if checks['measured_violation'](c) else 'undefined_only'
    return 'violation_only'


def equal_tie_weights(scores, k):
    if not 1 <= k <= len(scores) or not all(isfinite(s) for s in scores):
        raise ValueError('Invalid ranking budget or scores')
    cut = sorted(scores, reverse=True)[k-1]
    above = sum(s > cut for s in scores)
    tied = sum(s == cut for s in scores)
    return [1. if s > cut else (k-above)/tied if s == cut else 0. for s in scores]


def calibration_prefix(cells, is_supported, classify, risk_names):
    """Describe the highest-scoring failing block without skipping any ties."""
    if not cells or any(not isfinite(c['score']) for c in cells):
        raise ValueError('Missing eligible calibration scores')
    stop = max((c['score'] for c in cells if not is_supported(c)), default=None)
    block = [c for c in cells if c['score'] == stop]
    released = sum(stop is None or c['score'] > stop for c in cells)
    first_supported = max((c['score'] for c in cells if is_supported(c)), default=None)
    return dict(eligible=len(cells), released=released, risk_supported=sum(is_supported(c) for c in cells),
                first_block=dict(sorted(Counter(classify(c) for c in block).items())),
                first_block_undefined={k: sum(c['risk'][i] is None for c in block)
                                       for i, k in enumerate(risk_names)},
                scores_strictly_above_best_supported=(sum(c['score'] > first_supported for c in cells)
                                                      if first_supported is not None else None))


def mean(values):
    return fsum(values)/len(values)


def stratified_rank(cells, metrics):
    """Compare ranked selection to the exact uniform expectation on the same cells."""
    n = len(cells)
    k = max(1, n//4)
    weights = equal_tie_weights([c['score'] for c in cells], k)
    ranked, uniform = {}, {}
    for name, pred in metrics.items():
        flags = [pred(c) for c in cells]
        ranked[name] = fsum(w*v for w, v in zip(weights, flags))/k
        uniform[name] = mean(flags)
    return dict(ranked=ranked, uniform=uniform, eligible=n, quota=k)


def paired_interval(points, draws):
    estimates = sorted(mean([points[i] for i in sample]) for sample in draws)

    def q(p):
        u = (len(estimates)-1)*p
        i = int(u)
        return estimates[i]+(u-i)*(estimates[min(i+1, len(estimates)-1)]-estimates[i])

    return dict(point=mean(points), ci95=[q(.025), q(.975)])


def cells(groups, arm, seed):
    return [v['arms'][arm] for g in groups for f in g for v in f if v['seed'] == seed and v['arms'] is not None]


def diagnostic(evidence, frozen, replay):
    cal, held, draws = (evidence[k] for k in ('calibration', 'confirmation', 'source_resamples'))
    checks = metrics(replay)
    result = {}
    for arm in replay.ARMS:
        calibration = {}
        for seed in replay.SEEDS:
            c = cells(cal, arm, seed)
            row = calibration_prefix(c, checks['risk_pass'], lambda v: failure_reason(v, checks), RISK_NAMES)
            before = frozen['calibration_counts'][str(seed)][arm]
            if any(row[k] != before[k] for k in ('eligible', 'released')):
                raise ValueError('Calibration prefix changed')
            calibration[str(seed)] = dict(**row, dual_pass=sum(checks['dual'](v) for v in c))
        strata = [[stratified_rank(cells([g], arm, s), checks) for s in replay.SEEDS] for g in held]
        confirm = {}
        for metric in checks:
            top, uniform = ([mean([v[name][metric] for v in group]) for group in strata]
                            for name in ('ranked', 'uniform'))
            delta = [a-b for a, b in zip(top, uniform)]
            confirm[metric] = dict(top_quartile_rate=mean(top), uniform_rate=mean(uniform),
                                   ranked_minus_uniform=paired_interval(delta, draws),
                                   strictly_positive_components=sum(x > 1e-12 for x in delta),
                                   strictly_negative_components=sum(x < -1e-12 for x in delta))
        flattened = [v for g in strata for v in g]
        result[arm] = dict(calibration=calibration, confirmation=confirm,
            eligible_views=sum(v['eligible'] for v in flattened), equal_budget_views=sum(v['quota'] for v in flattened),
            full_quota_strata=sum(v['quota'] == v['eligible'] for v in flattened))
    return result
