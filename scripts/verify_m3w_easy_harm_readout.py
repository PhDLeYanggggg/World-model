"""Independent scalar checks for fixed six-arm cost-loss evaluation."""
import math

import numpy as np

from scripts.verify_m3w_selected_set_readout import scalar_bounds
from scripts.verify_m3w_temporal_auxiliary_readout import close

TARGET = 'easy_deviance'


def verify(pr, predictions, y, env, moving, support, sites, rec, frames, ids, result, actions):
    known = [all(math.isfinite(float(v)) for v in row) for row in y]
    groups = {}
    for i, key in enumerate(zip(rec.tolist(), frames.tolist())):
        groups.setdefault(key, []).append(i)
    manual = {name:[bool(moving[i] and support[i] and b > h and h <= .02*r and eh <= .02*er)
                    for i,(b,h,r,er,eh) in enumerate(pred)] for name,pred in predictions.items()}
    others = [name for name in predictions if name != TARGET]
    for other in others:
        left, right = [False]*len(y), [False]*len(y)
        for indices in groups.values():
            count = min(sum(manual[other][i] for i in indices),sum(manual[TARGET][i] for i in indices))
            for name,dest in ((other,left),(TARGET,right)):
                pool = sorted((i for i in indices if manual[name][i]),
                              key=lambda i:(predictions[name][i,1]-predictions[name][i,0],ids[i]))
                for i in pool[:count]: dest[i] = True
        manual[other+'_matched_'+TARGET],manual[TARGET+'_matched_'+other] = left,right
    checks = 0

    def bounds(labels,take,envelope):
        value = scalar_bounds(labels,np.array(take,dtype=bool),envelope)
        for group in ('','easy_'):
            h = value['selected_known_'+group+'harm_mass']
            r = value['selected_known_'+group+'reference_mass']
            value['known_'+group+'positive_risk'] = h/r if r > 0 else None
        value['switch_rate'] = sum(take)/len(labels) if len(labels) else None
        return value

    for name,take in manual.items():
        checks += close(actions[name].tolist(),take)
        checks += close(result['policies'][name],bounds(y,take,env))
    for recording in set(rec.tolist()):
        ix = [i for i in range(len(y)) if rec[i] == recording]
        for name in predictions:
            checks += close(result['recording_results'][str(recording)][name],
                bounds(y[ix],[manual[name][i] for i in ix],env[ix]))
    site_groups = {}
    for i,yes in enumerate(known):
        if yes: site_groups.setdefault(str(sites[i]),{}).setdefault((rec[i],frames[i]),[]).append(i)
    weights = [0.]*len(y)
    for queries in site_groups.values():
        for ix in queries.values():
            for i in ix: weights[i] = 1/(len(site_groups)*len(queries)*len(ix))
    signed = lambda v:(v[0]-v[1],v[1]-.02*v[2],v[4]-.02*v[3])
    for name,pred in predictions.items():
        for cohort in ('all','original_selected','own_selected'):
            ix = [i for i in range(len(y)) if known[i] and (cohort == 'all' or manual['original' if cohort == 'original_selected' else name][i])]
            mass = math.fsum(weights[i] for i in ix)
            loss = math.fsum(weights[i]*math.fsum(((a-b)/pr['scale']/pr['rms'][5+j])**2
                for j,(a,b) in enumerate(zip(signed(pred[i]),signed(y[i]))))/3 for i in ix)
            checks += close(result['scores'][name][cohort],dict(known_rows=len(ix),query_weight_mass=mass,
                global_MSE_contribution=loss,conditional_MSE=loss/mass if mass > 0 else None))
        ix = [i for i in range(len(y)) if known[i] and manual[name][i]]
        predicted = [math.fsum(float(pred[i,j]) for i in ix) for j in range(5)]
        observed = [math.fsum(float(y[i,j]) for i in ix) for j in range(5)]
        harm = [float(y[i,1]/pr['scale']) for i in ix]
        quantiles = np.quantile(harm,[.5,.95,.99]).tolist() if harm else [None]*3
        checks += close(result['diagnostics'][name],dict(known_selected=len(ix),predicted_moments=predicted,
            observed_moments=observed,predicted_observed_easy_harm_ratio=predicted[4]/observed[4] if observed[4] > 0 else None,
            positive_harm_over_frozen_TRAIN_scale=dict(p50=quantiles[0],p95=quantiles[1],p99=quantiles[2],
                maximum=max(harm) if harm else None),unknown_outcomes_imputed=False))
    den = math.fsum(float(y[i,2]) for i in range(len(y)) if known[i])
    checks += close(result['full_known_reference_mass'],den)
    for other in others:
        for mode in ('full','matched'):
            a = other if mode == 'full' else other+'_matched_'+TARGET
            b = TARGET if mode == 'full' else TARGET+'_matched_'+other
            observed = math.fsum((int(manual[b][i])-int(manual[a][i]))*float(y[i,0]-y[i,1])
                                 for i in range(len(y)) if known[i])
            unknown = math.fsum(float(env[i]) for i in range(len(y)) if not known[i] and manual[a][i] != manual[b][i])
            shared = sum(not known[i] and manual[a][i] and manual[b][i] for i in range(len(y)))
            checks += close(result['pairs'][other+'_'+mode],dict(known_difference_mass=observed,
                unknown_exchanged_envelope_mass=unknown,lower_mass=observed-unknown,upper_mass=observed+unknown,
                shared_unknown_selected=shared))
            checks += close(result['contrasts'][other+'_'+mode+'_paired_lower_percent'],100*(observed-unknown)/den if den > 0 else None)
            proxy = bounds(y,manual[b],env)['selected_net_gain_lower_mass']-bounds(y,manual[a],env)['selected_net_gain_lower_mass']
            checks += close(result['contrasts'][other+'_'+mode+'_lower_proxy_delta_percent'],100*proxy/den if den > 0 else None)
        for cohort in ('all','original_selected'):
            a,b = [result['scores'][k][cohort]['conditional_MSE'] for k in (other,TARGET)]
            checks += close(result['contrasts'][other+'_'+cohort+'_MSE'],b-a if a is not None and b is not None else None)
    return checks


def verify_summary(rows, cfg, summary):
    checks = 0
    for field,actual in summary['contrasts'].items():
        values = [r['result']['contrasts'][field] for r in rows]
        if any(v is None or not math.isfinite(v) for v in values):
            checks += close(actual['CI95'],None)
            checks += close(actual['mean'],None)
            continue
        groups = {}
        for row,value in zip(rows,values): groups.setdefault(row['source'],[]).append(value)
        means = {k:math.fsum(v)/len(v) for k,v in sorted(groups.items())}
        population = list(means.values())
        rng = np.random.default_rng(cfg['bootstrap_seed'])
        boot = [math.fsum(population[i] for i in rng.integers(0,len(population),len(population)))/len(population)
                for _ in range(cfg['bootstrap_resamples'])]
        checks += close(actual['localities'],means)
        checks += close(actual['mean'],math.fsum(population)/len(population))
        checks += close(actual['CI95'],np.quantile(boot,[.025,.975]).tolist())
    for name,actual in summary['safety'].items():
        ps = [r['result']['policies'][name] for r in rows]
        upper = [p['easy_selected_risk_upper'] for p in ps]
        known = [p['known_easy_positive_risk'] for p in ps]
        expected = dict(selected_occurrences=sum(p['selected_count'] for p in ps),
            unknown_selected_occurrences=sum(p['selected_unknown'] for p in ps),
            undefined_easy_risk=sum(v is None for v in upper),
            known_easy_violations=sum(v is not None and v > .02+1e-12 for v in known),
            easy_upper_violations=sum(v is not None and v > .02+1e-12 for v in upper),
            worst_easy_upper=max((v for v in upper if v is not None),default=None),
            finite_completion_supported=sum(p['finite_completion_supported'] for p in ps))
        checks += close(actual,expected)
    primary = [other+'_'+mode+'_paired_lower_percent'
               for other in ('quadratic','original') for mode in ('full','matched')]
    failures = []
    for field in primary:
        interval = summary['contrasts'][field]['CI95']
        if interval is None or interval[0] <= 0: failures.append('unsupported_'+field)
    for name in (TARGET,TARGET+'_matched_quadratic',TARGET+'_matched_original'):
        ps = [r['result']['policies'][name] for r in rows]
        if not all(p['finite_completion_supported'] for p in ps):
            failures.append('incomplete_or_unsafe_completion_'+name)
        if any(p[k] is not None and p[k] > .02+1e-12 for p in ps
               for k in ('known_easy_positive_risk','easy_selected_risk_upper')):
            failures.append('selected_easy_risk_violated_'+name)
    checks += close(summary['primary_contrasts'],primary)
    checks += close(summary['failure_reasons'],failures)
    checks += close(summary['advance_to_transfer_design'],not failures)
    return checks
