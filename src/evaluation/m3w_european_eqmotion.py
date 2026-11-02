"""Paired SOURCE forecasting controls; unknown labels are never zero errors."""
import numpy as np
from src.evaluation.m3w_native_metrics import native_errors


def paired_group(prediction, control, target, valid, cv):
    p, q, y, mask, reference = map(np.asarray, (prediction, control, target, valid, cv))
    a, f = native_errors(p, y, mask, np.ones(len(p)))
    b, g = native_errors(q, y, mask, np.ones(len(p)))
    if reference.shape != a.shape or not np.array_equal(np.isfinite(reference), mask.any(1)):
        raise ValueError('The original CV reference must have identical outcome support')
    known = mask.any(1)
    mass = float(reference[known].sum())
    if not mass > 0:
        raise ValueError('A positive fixed known CV denominator is required')
    known_delta = float((b[known]-a[known]).sum())
    unknown_radius = float(np.linalg.norm(p[~known]-q[~known], axis=-1).max(1).sum()) if (~known).any() else 0.
    def score(ade, fde):
        return dict(available_ADE=float(ade[known].mean()) if known.any() else None,
            endpoint_FDE=float(fde[np.isfinite(fde)].mean()) if np.isfinite(fde).any() else None,
            gain_vs_CV_percent=100*float((reference[known]-ade[known]).sum())/mass,
            zero_CV_harmed=int(((reference==0)&(ade>0)).sum()))
    return dict(rows=len(p), known_rows=int(known.sum()), unknown_rows=int((~known).sum()),
        complete_rows=int(mask.all(1).sum()), partial_rows=int((known&~mask.all(1)).sum()),
        endpoint_rows=int(mask[:,-1].sum()), CV_error_mass=mass,
        eqmotion=score(a,f), PCT=score(b,g),
        known_pp=100*known_delta/mass,
        completion_lower_pp=100*(known_delta-unknown_radius)/mass,
        completion_upper_pp=100*(known_delta+unknown_radius)/mass,
        unknown_paired_radius=unknown_radius)


def interval(groups, field, sites, resamples, seed):
    sites = sorted(sites)
    if not sites or {r['site'] for r in groups} != set(sites):
        raise ValueError('The complete fixed locality roster is required')
    values = []
    for site in sites:
        rows = [r for r in groups if r['site']==site]
        keys = {(r['trial'],r['seed']) for r in rows}
        if len(keys)!=len(rows) or len(rows)!=6 or {r['seed'] for r in rows}!={17,29,43}:
            raise ValueError('Each SOURCE locality needs two excluded folds and three actual seeds')
        v = [r['scores'][field] for r in rows]
        if not np.isfinite(v).all():
            raise ValueError('Undefined required group is not a favorable zero')
        values.append(float(np.mean(v)))
    samples = np.random.default_rng(seed).choice(values,size=(resamples,len(values))).mean(1)
    return dict(point=float(np.mean(values)),CI95=np.quantile(samples,[.025,.975]).tolist(),
        by_locality=dict(zip(sites,values)),resamples=resamples,seed=seed,
        unit='SOURCE_locality_after_fold_and_actual_training_seed_averaging',
        independent_confirmation=False)
