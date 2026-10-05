"""Scalar verification using separate aggregation and decision implementations."""
import math
import numpy as np
from scripts.verify_m3w_selected_set_readout import scalar_bounds


def close(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        return sum(close(actual[k], v) for k, v in expected.items())
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        return sum(close(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, (float, np.floating)):
        assert actual is not None and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8), (actual, expected)
    else:
        assert actual == expected, (actual, expected)
    return 1


def verify(pr, pred, y, env, moving, support, sites, rec, frames, ids, result, actions):
    known = [all(math.isfinite(float(v)) for v in row) for row in y]
    groups = {}
    for i, key in enumerate(zip(rec.tolist(), frames.tolist())):
        groups.setdefault(key, []).append(i)
    manual = {}
    for name, p in pred.items():
        manual[name] = [bool(moving[i] and support[i] and b > h and h <= .02*r and eh <= .02*er)
                        for i, (b, h, r, er, eh) in enumerate(p)]
    for other in pred:
        if other == 'temporal': continue
        left, right = [False]*len(y), [False]*len(y)
        for group in groups.values():
            count = min(sum(manual[other][i] for i in group), sum(manual['temporal'][i] for i in group))
            for name, out in ((other, left), ('temporal', right)):
                pool = sorted((i for i in group if manual[name][i]), key=lambda i: (pred[name][i, 1]-pred[name][i, 0], ids[i]))
                for i in pool[:count]: out[i] = True
        manual[other+'_matched_temporal'], manual['temporal_matched_'+other] = left, right
    count = 0
    for arm, take in manual.items():
        count += close(actions[arm].tolist(), take)
        expected = scalar_bounds(y, np.array(take), env)
        r, h = expected['selected_known_easy_reference_mass'], expected['selected_known_easy_harm_mass']
        expected['known_easy_positive_risk'] = h/r if r > 0 else None
        expected['switch_rate'] = sum(take)/len(y) if len(y) else None
        count += close(result['policies'][arm], expected)
    # Query-equal error weights are built without the production weights helper.
    by_site = {}
    for i, yes in enumerate(known):
        if yes: by_site.setdefault(str(sites[i]), {}).setdefault((rec[i], frames[i]), []).append(i)
    weights = [0.]*len(y)
    for queries in by_site.values():
        for rows in queries.values():
            for i in rows: weights[i] = 1/(len(by_site)*len(queries)*len(rows))
    signed = lambda row: (row[0]-row[1], row[1]-.02*row[2], row[4]-.02*row[3])
    for arm, p in pred.items():
        for cohort in ('all', 'original_selected', 'own_selected'):
            chosen = [i for i in range(len(y)) if known[i] and (cohort == 'all' or manual['original' if cohort == 'original_selected' else arm][i])]
            mass = math.fsum(weights[i] for i in chosen)
            loss = math.fsum(weights[i]*math.fsum(((a-b)/pr['scale']/pr['rms'][5+j])**2
                for j, (a, b) in enumerate(zip(signed(p[i]), signed(y[i]))))/3 for i in chosen)
            count += close(result['scores'][arm][cohort], dict(known_rows=len(chosen), query_weight_mass=mass,
                global_MSE_contribution=loss, conditional_MSE=loss/mass if mass > 0 else None))
    denominator = math.fsum(float(y[i, 2]) for i in range(len(y)) if known[i])
    count += close(result['full_known_reference_mass'], denominator)
    for other in pred:
        if other == 'temporal': continue
        for mode in ('full', 'matched'):
            a = other if mode == 'full' else other+'_matched_temporal'
            b = 'temporal' if mode == 'full' else 'temporal_matched_'+other
            observed = math.fsum((int(manual[b][i])-int(manual[a][i]))*float(y[i, 0]-y[i, 1]) for i in range(len(y)) if known[i])
            unknown = math.fsum(float(env[i]) for i in range(len(y)) if not known[i] and manual[a][i] != manual[b][i])
            shared = sum(not known[i] and manual[a][i] and manual[b][i] for i in range(len(y)))
            count += close(result['pairs'][other+'_'+mode], dict(known_difference_mass=observed,
                unknown_exchanged_envelope_mass=unknown, lower_mass=observed-unknown,
                upper_mass=observed+unknown, shared_unknown_selected=shared))
            count += close(result['contrasts'][other+'_'+mode+'_paired_lower_percent'],
                100*(observed-unknown)/denominator if denominator > 0 else None)
            proxy = scalar_bounds(y, np.array(manual[b]), env)['selected_net_gain_lower_mass']-scalar_bounds(y, np.array(manual[a]), env)['selected_net_gain_lower_mass']
            count += close(result['contrasts'][other+'_'+mode+'_lower_proxy_delta_percent'],
                100*proxy/denominator if denominator > 0 else None)
        for cohort in ('all', 'original_selected'):
            a, b = [result['scores'][k][cohort]['conditional_MSE'] for k in (other, 'temporal')]
            count += close(result['contrasts'][other+'_'+cohort+'_MSE'], b-a if a is not None and b is not None else None)
    return count


def verify_summary(rows, cfg, summary):
    checks = 0
    for key, actual in summary['contrasts'].items():
        values = [r['result']['contrasts'][key] for r in rows]
        if any(v is None or not math.isfinite(v) for v in values):
            checks += close(actual['CI95'], None)
            continue
        localities = {}
        for r, v in zip(rows, values): localities.setdefault(r['source'], []).append(v)
        means = {s: math.fsum(v)/len(v) for s, v in sorted(localities.items())}
        population = list(means.values()); rng = np.random.default_rng(cfg['bootstrap_seed'])
        draws = [math.fsum(population[i] for i in rng.integers(0, len(population), len(population)))/len(population)
                 for _ in range(cfg['bootstrap_resamples'])]
        checks += close(actual['localities'], means)
        checks += close(actual['mean'], math.fsum(population)/len(population))
        checks += close(actual['CI95'], np.quantile(draws, [.025, .975]).tolist())
    for name, actual in summary['safety'].items():
        policies = [r['result']['policies'][name] for r in rows]
        risk = [p['easy_selected_risk_upper'] for p in policies]
        expected = dict(selected_occurrences=sum(p['selected_count'] for p in policies),
            unknown_selected_occurrences=sum(p['selected_unknown'] for p in policies),
            undefined_easy_risk=sum(r is None for r in risk),
            easy_upper_violations=sum(r is not None and r > cfg['risk_budget']+1e-12 for r in risk),
            worst_easy_upper=max((r for r in risk if r is not None), default=None),
            finite_completion_supported=sum(p['finite_completion_supported'] for p in policies))
        checks += close(actual, expected)
    failures = []
    for key, result in summary['contrasts'].items():
        ci = result['CI95']
        if ci is None or (ci[1] >= 0 if key.endswith('_all_MSE') else
            ci[1] > 0 if key.endswith('_original_selected_MSE') else ci[0] <= 0):
            failures.append('unsupported_'+key)
    for name in summary['safety']:
        if name == 'temporal' or name.startswith('temporal_matched_'):
            if not all(r['result']['policies'][name]['finite_completion_supported'] for r in rows):
                failures.append('absolute_original_risk_or_utility_not_supported_'+name)
    checks += close(summary['failure_reasons'], failures)
    checks += close(summary['advance_to_transfer_design'], not failures)
    return checks
