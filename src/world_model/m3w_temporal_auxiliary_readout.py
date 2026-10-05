"""Frozen seven-arm development readout; no threshold or checkpoint selection."""
import math
import numpy as np

from src.world_model import m3w_component_calibration as policy
from src.world_model import m3w_source_forest as forest
from src.world_model.m3w_unknown_outcome_bounds import completion_bounds

CONTROLS = ('original', 'additive', 'poisson', 'cost')
TRAINED = ('none', 'rowmean', 'temporal')
ARMS = CONTROLS + TRAINED
COMPARATORS = CONTROLS + TRAINED[:2]


def decisions(predictions, moving, support, recordings, frames, ids):
    ids, rec, frame = map(np.asarray, (ids, recordings, frames))
    if (set(predictions) != set(ARMS) or ids.ndim != 1 or len(np.unique(ids)) != len(ids)
            or rec.shape != ids.shape or frame.shape != ids.shape):
        raise ValueError('Seven paired arms and unique aligned row IDs required')
    actions = {k: policy.eligible(v, moving, support) for k, v in predictions.items()}
    if any(v.shape != ids.shape for v in actions.values()):
        raise ValueError('Prediction and metadata rows must match')
    for other in COMPARATORS:
        left, right = policy.matched(actions[other], actions['temporal'],
            forest.core.signed(predictions[other])[:, 0],
            forest.core.signed(predictions['temporal'])[:, 0], rec, frame, ids)
        actions[other+'_matched_temporal'], actions['temporal_matched_'+other] = left, right
    return actions


def paired_completion(targets, left, right, envelope):
    """Same unknown outcome on both arms; shared selections cancel exactly."""
    y, env = np.asarray(targets, float), np.asarray(envelope, float)
    left, right = np.asarray(left), np.asarray(right)
    completion_bounds(y, left, env)
    completion_bounds(y, right, env)
    known = np.isfinite(y).all(1)
    difference = right.astype(int)-left.astype(int)
    observed = float(np.dot(difference[known], y[known, 0]-y[known, 1]))
    # Either sign is possible for a wholly unknown gain, within disagreement.
    mass = float(env[(left ^ right) & ~known].sum())
    return dict(known_difference_mass=observed, unknown_exchanged_envelope_mass=mass,
        lower_mass=observed-mass, upper_mass=observed+mass,
        shared_unknown_selected=int((left & right & ~known).sum()))


def score(pred, y, pr, sites, recordings, frames, cohort):
    known = np.isfinite(y).all(1)
    if not known.any():
        return dict(known_rows=0, query_weight_mass=0., global_MSE_contribution=0., conditional_MSE=None)
    w, _ = forest.core.weights(sites, recordings, frames, known)
    selected = np.asarray(cohort) & known
    err = (forest.core.signed(pred[selected])-forest.core.signed(y[selected]))/pr['scale']/pr['rms'][5:]
    mass = float(w[selected].sum())
    loss = float((w[selected, None]*err**2).sum()/3)
    return dict(known_rows=int(selected.sum()), query_weight_mass=mass,
        global_MSE_contribution=loss, conditional_MSE=loss/mass if mass > 0 else None)


def evaluate(pr, predictions, targets, envelope, moving, support, sites, recordings, frames, ids):
    actions = decisions(predictions, moving, support, recordings, frames, ids)
    y = np.asarray(targets, float)
    policies = {k: completion_bounds(y, a, envelope) for k, a in actions.items()}
    known = np.isfinite(y).all(1)
    den = float(y[known, 2].sum())
    if any(np.asarray(v).shape != (len(y),) for v in (sites, recordings, frames, ids)):
        raise ValueError('Aligned label and metadata rows required')
    scores = {arm: {name: score(pred, y, pr, sites, recordings, frames, mask)
        for name, mask in dict(all=np.ones(len(y), bool), original_selected=actions['original'],
                               own_selected=actions[arm]).items()} for arm, pred in predictions.items()}
    pairs, contrasts = {}, {}
    for other in COMPARATORS:
        for cohort in ('all', 'original_selected'):
            a, b = [scores[k][cohort]['conditional_MSE'] for k in (other, 'temporal')]
            contrasts[other+'_'+cohort+'_MSE'] = b-a if a is not None and b is not None else None
        for mode in ('full', 'matched'):
            a = other if mode == 'full' else other+'_matched_temporal'
            b = 'temporal' if mode == 'full' else 'temporal_matched_'+other
            pair = paired_completion(y, actions[a], actions[b], envelope)
            pairs[other+'_'+mode] = pair
            delta = policies[b]['selected_net_gain_lower_mass']-policies[a]['selected_net_gain_lower_mass']
            contrasts[other+'_'+mode+'_lower_proxy_delta_percent'] = 100*delta/den if den > 0 else None
            contrasts[other+'_'+mode+'_paired_lower_percent'] = 100*pair['lower_mass']/den if den > 0 else None
    for row in policies.values():
        r, h = row['selected_known_easy_reference_mass'], row['selected_known_easy_harm_mass']
        row['known_easy_positive_risk'] = h/r if r > 0 else None
        row['switch_rate'] = row['selected_count']/len(y) if len(y) else None
    return dict(rows=len(y), known_rows=int(known.sum()), unknown_rows=int((~known).sum()),
        full_known_reference_mass=den, scores=scores, policies=policies, pairs=pairs,
        contrasts=contrasts, inference_label_filter=False), actions


def interval(rows, field, draws, seed):
    """Missing head support invalidates the contrast rather than dropping it."""
    grouped = {}
    for row in rows:
        value = row['result']['contrasts'][field]
        if value is None or not math.isfinite(value):
            return dict(mean=None, CI95=None, localities={}, status='undefined_support_no_dropping')
        grouped.setdefault(row['source'], []).append(value)
    localities = {k: float(np.mean(v)) for k, v in sorted(grouped.items())}
    if len(localities) < 2:
        return dict(mean=None, CI95=None, localities=localities, status='insufficient_localities')
    values = np.array(list(localities.values()))
    samples = np.random.default_rng(seed).integers(0, len(values), (draws, len(values)))
    return dict(mean=float(values.mean()), CI95=np.quantile(values[samples].mean(1), [.025, .975]).tolist(),
        localities=localities, bootstrap_draws=draws,
        status='nominal_exposed_development_not_independent_confirmation')


def summarize(rows, expected, cfg):
    keys = lambda rr: [(r['group'], r['source'], r['head_seed']) for r in rr]
    if len(set(keys(rows))) != len(rows) or sorted(keys(rows)) != sorted(keys(expected)):
        raise ValueError('Every registered source/head is required exactly once')
    if len(rows) != cfg['source_heads'] or len({r['source'] for r in rows}) != 12:
        raise ValueError('The 72-head, 12-locality development grid is fixed')
    by_group = {}
    for row in rows:
        by_group.setdefault(row['group'], set()).add(row['head_seed'])
    if any(seeds != set(cfg['head_seeds']) for seeds in by_group.values()):
        raise ValueError('All three head seeds required in every context')
    contrasts = {key: interval(rows, key, cfg['bootstrap_resamples'], cfg['bootstrap_seed'])
                 for key in rows[0]['result']['contrasts']}
    failures = []
    for key, value in contrasts.items():
        ci = value['CI95']
        good = ci is not None and (ci[1] < 0 if key.endswith('_all_MSE') else
            ci[1] <= 0 if key.endswith('_original_selected_MSE') else ci[0] > 0)
        if not good:
            failures.append('unsupported_'+key)
    safety = {}
    for arm in rows[0]['result']['policies']:
        policies = [r['result']['policies'][arm] for r in rows]
        risk = [p['easy_selected_risk_upper'] for p in policies]
        safety[arm] = dict(selected_occurrences=sum(p['selected_count'] for p in policies),
            unknown_selected_occurrences=sum(p['selected_unknown'] for p in policies),
            undefined_easy_risk=sum(v is None for v in risk),
            easy_upper_violations=sum(v is not None and v > cfg['risk_budget']+1e-12 for v in risk),
            worst_easy_upper=max((v for v in risk if v is not None), default=None),
            finite_completion_supported=sum(p['finite_completion_supported'] for p in policies))
        if arm == 'temporal' or arm.startswith('temporal_matched_'):
            if safety[arm]['finite_completion_supported'] != len(rows):
                failures.append('absolute_original_risk_or_utility_not_supported_'+arm)
    return dict(groups=len(rows), localities=12, trained_cost_heads=cfg['neural_fits'],
        contrasts=contrasts, safety=safety, advance_to_transfer_design=not failures,
        failure_reasons=failures, bootstrap_resamples=cfg['bootstrap_resamples'],
        head_seeds_are_not_forecaster_seeds=True, independent_confirmation=False,
        deployment_changed=False, transfer_evaluated=False, stage5c_executed=False, smc_enabled=False)
