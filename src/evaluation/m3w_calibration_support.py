"""Read-only calibration feasibility checks for the fixed source producer design.

These checks do not assign data roles or certify statistical independence. They
complement the experiment contract by examining cost-target producer exposure.
"""
from itertools import combinations

import numpy as np

from src.world_model.m3w_joint_intervention import screen_cluster_risks
from src.world_model.m3w_native_nested import require_source_producer


def require_calibration_exclusion(*, calibration_sites, head_fit_sites,
                                  preprocessing_fit_sites, scoring_producer,
                                  target_producers):
    """Reject direct and indirect fitting exposure before any calibration read."""
    protected = set(calibration_sites)
    if not protected or any(not isinstance(s, str) or not s for s in protected):
        raise ValueError('Explicit nonempty calibration sites required')
    for name, sites in (('head fitting', head_fit_sites),
                        ('head preprocessing', preprocessing_fit_sites)):
        if protected & set(sites):
            raise ValueError(f'{name} exposes calibration sites: {sorted(protected & set(sites))}')
    if not target_producers:
        raise ValueError('Explicit cost-target producer lineage required')
    for kind, producer in [('scoring', scoring_producer)] + [
            ('cost target', p) for p in target_producers]:
        if producer['parents'] or producer['initialization'] != 'random_seed':
            raise ValueError('This source-only check requires random-init producers without parents')
        for field in ('training_sites', 'preprocessing_fit_sites',
                      'checkpoint_selection_sites', 'calibration_sites'):
            overlap = protected & set(producer[field])
            if overlap:
                raise ValueError(f'{kind} producer {producer["id"]}/{field} exposes {sorted(overlap)}')
    return {'fitting_exclusion_pass': True, 'independence_verified': False,
            'data_roles_assigned': False, 'risk_calibrated': False}


def audit_view(view, head_training_sites, roster):
    """Test existing-head reuse and row-only refitting for every proposed inner site."""
    roster, outer, seed = set(roster), view['outer_site'], view['seed']
    groups = view['groups']
    if (set(head_training_sites) != roster - {outer}
            or len(head_training_sites) != len(roster) - 1
            or len(groups) != len(roster) - 1
            or {g['inner_site'] for g in groups} != roster - {outer}):
        raise ValueError('Complete fixed outer-complement view required')
    scoring = view['outer_producer']['producer']
    require_source_producer(scoring, outer_site=outer, row_site=outer, roster=roster, seed=seed)
    for group in groups:
        require_source_producer(group['producer'], outer_site=outer,
            row_site=group['inner_site'], roster=roster, seed=seed)
    outer_check = require_calibration_exclusion(calibration_sites=[outer],
        head_fit_sites=head_training_sites, preprocessing_fit_sites=head_training_sites,
        scoring_producer=scoring, target_producers=[g['producer'] for g in groups])
    records = []
    for proposed in sorted(roster - {outer}):
        remaining = [g for g in groups if g['inner_site'] != proposed]
        proposed_forecaster = next(g['producer'] for g in groups if g['inner_site'] == proposed)
        inner_fit = sorted(roster - {outer, proposed})
        attempts = {}
        # Even after removing the site and replacing the scoring predictor, the
        # retained cost-target predictors still trained on that site's outcomes.
        for name, fitted, producer, targets in (
            ('reuse_current_head', head_training_sites, scoring, groups),
            ('drop_site_rows_and_refit_head_only', inner_fit, scoring, remaining),
            ('drop_rows_and_replace_scoring_predictor', inner_fit, proposed_forecaster, remaining)):
            try:
                result = require_calibration_exclusion(calibration_sites=[proposed],
                    head_fit_sites=fitted, preprocessing_fit_sites=fitted,
                    scoring_producer=producer, target_producers=[g['producer'] for g in targets])
                attempts[name] = dict(accepted=True, **result)
            except ValueError as exc:
                attempts[name] = dict(accepted=False, reason=str(exc))
        records.append(dict(outer_site=outer, proposed_calibration_site=proposed,
            seed=seed, remaining_head_fit_sites=inner_fit,
            remaining_training_rows=sum(g['rows'] for g in remaining),
            directly_removed_rows=next(g['rows'] for g in groups if g['inner_site'] == proposed),
            exposed_cost_target_producers=[g['producer']['id'] for g in remaining
                if proposed in g['producer']['training_sites']],
            required_target_exclusions=[dict(row_site=g['inner_site'],
                excluded_sites=sorted({outer, proposed, g['inner_site']})) for g in remaining],
            replacement_scoring_producer=proposed_forecaster['id'], attempts=attempts,
            current_outer_fitting_exclusion=outer_check['fitting_exclusion_pass'],
            current_outer_is_independent_confirmation=False,
            current_outer_already_read=True, risk_calibrated=False))
    return records


def required_producers(roster, seeds, available):
    """Enumerate missing triple exclusions; this does not approve their fitting."""
    if len(roster) != 4 or len(set(roster)) != 4 or len(set(seeds)) != len(seeds):
        raise ValueError('Exactly four source sites and unique seeds required')
    keys = {(tuple(sorted(p['excluded_sites'])), p['seed']) for p in available}
    return [dict(excluded_sites=list(excluded), training_sites=sorted(set(roster)-set(excluded)),
        seed=seed, available=(tuple(excluded), seed) in keys)
        for excluded in combinations(sorted(roster), 3) for seed in seeds]


def best_case_cluster_support(counts=(1, 2, 4), delta=.05):
    """Synthetic zero-loss lower limit on this existing screening bound's width.

    The unit-range loss is illustrative, NOT the project's 2% easy-ADE criterion.
    Every cluster here is assumed independent solely for the arithmetic example.
    """
    if any(type(n) is not int or n < 1 for n in counts):
        raise ValueError('Positive integer cluster counts required')
    result = []
    for n in counts:
        receipt = screen_cluster_risks(losses=np.zeros((n, 1, 1)),
            cluster_ids=[f'assumed_independent_{i}' for i in range(n)], fitted_cluster_ids=[],
            policy_ids=['fixed_illustration'], lower=[0.], upper=[1.], tolerance=[0.], delta=delta)
        result.append(dict(assumed_independent_clusters=n,
            best_case_unit_range_upper_bound=float(receipt['upper_risk_bound'][0, 0]),
            delta_illustration=delta, risk_tolerance_changed=False, real_calibration=False,
            same_as_relative_easy_degradation=False))
    return result
