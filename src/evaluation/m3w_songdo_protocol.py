"""Source-role assignment and policy release on frozen Songdo predictions."""

import math

import numpy as np

from src.evaluation import m3w_songdo_matched_readout as readout

ARMS, SEEDS = readout.ARMS, readout.SEEDS
CALIBRATION_CUTOFF_RULE = 'calibration_complete_positive_cv_ade_q25_q75'


def cutoff_rule(spec):
    return spec.get('cutoff_rule', 'fixed_native')


def matched_only(spec):
    return spec['release']['method'] == 'matched_only'


def fixed_cuts(spec):
    return (type(spec.get('easy_cut')) in (int, float) and type(spec.get('hard_cut')) in (int, float)
            and 0 < spec['easy_cut'] <= spec['hard_cut'] < math.inf)


def assignments(spec, components, flight_sites):
    audited = [s for component in components for s in component]
    if len(audited) != len(set(audited)) or set(audited) != set(flight_sites.values()):
        raise ValueError('Source groups must partition the exact flight roster')
    mode, roles = spec['mode'], spec['site_roles']
    expected = {'calibration', 'confirmation'} if mode == 'calibrated' else {'confirmation'}
    if (mode not in ('calibrated', 'zero_shot') or set(roles) != set(audited)
            or set(roles.values()) != expected or not spec.get('approval_reference', '').strip()):
        raise ValueError('Require explicit prospective roles and approval')
    if any(len({roles[s] for s in group}) != 1 for group in components):
        raise ValueError('Linked sites cannot be assigned to different roles')
    rule = spec['release']
    if mode == 'calibrated':
        if rule != {'method': 'risk_passing_score_prefix'}:
            raise ValueError('Expected the prespecified calibration rule')
        if cutoff_rule(spec) not in ('fixed_native', CALIBRATION_CUTOFF_RULE):
            raise ValueError('Expected a declared calibration cutoff rule')
    elif rule == {'method': 'matched_only'}:
        if cutoff_rule(spec) != 'not_used':
            raise ValueError('Matched-only evaluation does not assign risk cutoffs')
    elif rule['method'] == 'fixed_scores' and rule.get('threshold_source', '').strip():
        check_thresholds(rule['thresholds'])
        if cutoff_rule(spec) != 'fixed_native':
            raise ValueError('Fixed-score zero-shot cannot estimate cutoffs')
    else:
        raise ValueError('Zero-shot needs an explicit release policy fixed before outcome access')
    if cutoff_rule(spec) == 'fixed_native':
        if not fixed_cuts(spec):
            raise ValueError('Require finite, positive fixed native-unit cutoffs')
    elif 'easy_cut' in spec or 'hard_cut' in spec:
        raise ValueError('Do not mix fixed cutoffs with a different declared rule')
    component_of = {s: '+'.join(sorted(group)) for group in components for s in group}
    return {name: (roles[site], component_of[site]) for name, site in flight_sites.items()}


def check_thresholds(thresholds):
    if set(thresholds) != {str(s) for s in SEEDS}:
        raise ValueError('Expected thresholds for all three frozen seeds')
    for by_arm in thresholds.values():
        if set(by_arm) != set(ARMS) or any(
            x is not None and x != 'all' and (type(x) not in (int, float) or not math.isfinite(x))
            for x in by_arm.values()
        ):
            raise ValueError('Expected a finite threshold, all, or no release for every arm')


def calibration(records):
    """Select the largest tie-complete prefix with supported 2% risk in every released flight."""
    thresholds, coverage = {}, {}
    for s in map(str, SEEDS):
        thresholds[s], coverage[s] = {}, {}
        for arm in ARMS:
            ranked = sorted((r[s]['scores'][arm], r[s]['matched']['arms'][arm]['risk']['supported'])
                            for r in records.values() if r[s]['matched']['matched_eligible'])
            bound = max((score for score, passed in ranked if not passed), default=-math.inf)
            prefix = [score for score, _ in ranked if score > bound]
            thresholds[s][arm] = min(prefix) if prefix else None
            coverage[s][arm] = dict(eligible=len(ranked), released=len(prefix))
    return dict(thresholds=thresholds, calibration_counts=coverage)


def positive_complete_cv_ade(cv, target, valid):
    if (cv.shape != target.shape or cv.ndim != 3 or cv.shape[1:] != (12, 2)
            or valid.shape != cv.shape[:2] or valid.dtype != bool or not np.isfinite(cv).all()
            or not np.isfinite(target[valid]).all()):
        raise ValueError('Expected aligned CV forecasts and targets')
    at = valid.all(axis=1)
    error = np.linalg.norm(cv[at]-target[at], axis=-1).mean(1)
    return error[error > 0.]


def fit_native_cutoffs(errors_by_flight):
    rows = [np.asarray(errors_by_flight[k], dtype=float) for k in sorted(errors_by_flight)]
    if not rows or any(x.ndim != 1 or not np.isfinite(x).all() or np.any(x <= 0) for x in rows):
        raise ValueError('Cutoff inputs must be positive finite complete CV errors')
    values = np.concatenate(rows)
    if not len(values):
        raise ValueError('No complete positive CV errors in calibration')
    easy, hard = np.quantile(values, [.25, .75], method='linear').tolist()
    return dict(easy_cut=easy, hard_cut=hard, support_rows=len(values),
                support_by_flight={k: len(errors_by_flight[k]) for k in sorted(errors_by_flight)})


def score_session(cv, ols, offsets, target, valid, options, spec, thresholds=None):
    if set(options) != set(SEEDS):
        raise ValueError('Expected all seed views of the flight')
    by_seed = {}
    only = matched_only(spec)
    if (only and thresholds is not None) or (not only and not fixed_cuts(spec)):
        raise ValueError('Matched-only and risk/release scoring require different inputs')
    if thresholds is not None:
        weight = np.repeat(1./np.diff(offsets), np.diff(offsets))
        cv_mass = float(weight @ np.where(valid, np.linalg.norm(cv-target, axis=-1), 0.).sum(1))
        fallback_vs_ols4 = readout.compare(cv, ols, target, valid, weight, cv_mass, None)
    for s, (proposal, scores) in options.items():
        if ((proposal is None and scores) or (proposal is not None and
                (set(scores) != set(ARMS) or not all(math.isfinite(v) for v in scores.values())))):
            raise ValueError('Expected finite frozen scores for each eligible arm')
        matched = readout.score_flight(cv, ols, target, valid, offsets, proposal,
                                      easy_cut=spec.get('easy_cut'), hard_cut=spec.get('hard_cut'))
        matched['coverage']['complete_pair_windows'] = int(matched['coverage']['complete_pair_windows'])
        row = by_seed[str(s)] = dict(scores=scores, matched=matched)
        if thresholds is not None:
            row['released'] = {}
            fallback = dict(**matched['arms']['cv'], risk=None, intervention_rate=0., vs_ols4=fallback_vs_ols4)
            for arm in ARMS:
                cut = thresholds[str(s)][arm]
                release = (arm in scores and cut is not None and (cut == 'all' or scores[arm] >= cut))
                row['released'][arm] = dict(edited=release, metrics=matched['arms'][arm] if release else fallback)
    return by_seed


def summarize(records, flight_components, *, only=False):
    views = {name: {int(s): r for s, r in by_seed.items()} for name, by_seed in records.items()}
    matched = {name: {s: r['matched'] for s, r in rows.items()} for name, rows in views.items()}
    all_views = [r for by_seed in views.values() for r in by_seed.values()]
    result = dict(flights=len(views), source_components=len(set(flight_components.values())),
                  flight_seed_views=len(all_views), matched_seed_views=sum(
                      r['matched']['matched_eligible'] for r in all_views),
                  matched_joint_minus={arm: {
                      k: readout.component_interval(matched, flight_components, arm, k)
                      for k in ('known_pp', 'lower_pp', 'relative_available_native', 'relative_lower_native')
                  } for arm in ARMS[:-1]})
    if only:
        eligible = {name: {s: dict(matched_contrasts=r['arms'] if r['matched_eligible'] else {})
                          for s, r in rows.items()} for name, rows in matched.items()}
        # All compared arms use exactly the same matched subset and label support.
        metrics = ('available_ade_native', 'available_fde_native', 'complete_roster_ade_native',
                   'complete_roster_fde_native', 'available_ade_p95_native', 'query_intervention_rate')
        result['matched_native_metrics'] = {
            arm: {k: readout.component_interval(eligible, flight_components, arm, k)
                  for k in metrics} for arm in ('cv', 'ols4', *ARMS)}
        for ref in ('vs_cv', 'vs_ols4'):
            paired = {name: {s: dict(matched_contrasts={
                a: row.get(ref, {}) for a, row in v['matched_contrasts'].items() if a in ARMS})
                for s, v in rows.items()} for name, rows in eligible.items()}
            result['matched_'+ref] = {a: {k: readout.component_interval(paired, flight_components, a, k)
                                         for k in ('known_pp', 'lower_pp')} for a in ARMS}
        result['risk_and_release'] = 'not_run_no_calibration'
        return result
    result['released'] = {}
    for arm in ARMS:
        rows = [r['released'][arm] for r in all_views]
        released = {name: {s: dict(matched_contrasts=r['released'][arm]['metrics']) for s, r in v.items()}
                    for name, v in views.items()}
        result['released'][arm] = dict(
            edited=sum(r['edited'] for r in rows), required_views=len(rows),
            risk_supported=sum(bool((r['metrics']['risk'] or {}).get('supported')) for r in rows),
            no_edit_not_certified=sum(not r['edited'] for r in rows),
            worst_available_ade_native=max((r['metrics']['available_ade_native'] for r in rows
                if r['metrics']['available_ade_native'] is not None), default=None),
            completion_lower_pp={ref: readout.component_interval(released, flight_components, ref, 'lower_pp')
                                 for ref in ('vs_cv', 'vs_ols4')})
    return result
