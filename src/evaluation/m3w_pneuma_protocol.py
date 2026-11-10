"""Explicit acquisition-date roles and prior-only operating rules."""

from math import isfinite

import numpy as np

from src.evaluation import m3w_pneuma_matched_readout as readout
from src.evaluation.m3w_songdo_protocol import fixed_cuts, fit_native_cutoffs


CAL_RULE = 'complete_positive_cv_ade_q25_q75'
RELEASE_RULE = 'calibration_member_mean_score_risk_prefix'
ZERO_RULE = 'frozen_donor_cuts_matched_only'


def assignments(spec, source_dates):
    required = {'mode', 'approval_reference', 'date_roles', 'observed_steps', 'prediction_steps',
                'stride_ticks', 'risk_budget', 'comparison', 'rule'}
    if not required <= spec.keys() or not isinstance(spec['approval_reference'], str) or not spec['approval_reference'].strip():
        raise ValueError('An explicit prospective author decision is required')
    if (spec['observed_steps'], spec['prediction_steps'], spec['stride_ticks'], spec['risk_budget'], spec['comparison']) != (
            8, 12, 10, .02, 'all_past_defined_matched_queries'):
        raise ValueError('The available fixed-count pilot has a different clock, roster, or risk budget')
    dates = sorted(set(source_dates.values()))
    roles = spec['date_roles']
    if len(dates) != 4 or set(roles) != set(dates):
        raise ValueError('Assign entire acquisition dates; no member-level split')
    if spec['mode'] == 'zero_shot':
        if (set(roles.values()) != {'confirmation'} or spec['rule'] != ZERO_RULE or not fixed_cuts(spec)
                or not isinstance(spec.get('donor_cut_source'), str) or not spec['donor_cut_source'].strip()
                or set(spec)-required != {'donor_cut_source', 'easy_cut', 'hard_cut'}):
            raise ValueError('Zero-shot needs prespecified donor cuts and no Athens-fitted release')
    elif spec['mode'] == 'calibrated':
        if ([roles[d] for d in dates] != ['calibration']*2+['confirmation']*2 or spec['rule'] != RELEASE_RULE
                or spec.get('calibration_cut_rule') != CAL_RULE or set(spec)-required != {'calibration_cut_rule'}):
            raise ValueError('The chronological two-date calibration must be declared before access')
    else:
        raise ValueError('Unknown evaluation mode')
    return {n: roles[d] for n, d in source_dates.items()}


def thresholds(records):
    """Largest tie-complete member prefix passing the unchanged 2% rule."""
    cuts = {}
    for s in map(str, readout.SEEDS):
        cuts[s] = {}
        for a in readout.ARMS:
            score = [r[s]['scores'][a] for r in records.values()]
            if not score or not all(isfinite(v) for v in score):
                raise ValueError('Require finite frozen calibration scores')
            reject = [v for v, r in zip(score, records.values(), strict=True) if not r[s]['arms'][a]['risk']['supported']]
            admissible = [v for v in score if not reject or v > max(reject)]
            cuts[s][a] = min(admissible) if admissible else None
    return cuts


def check_thresholds(cuts):
    if not isinstance(cuts, dict) or set(cuts) != {str(s) for s in readout.SEEDS} or any(
            not isinstance(by_arm, dict) or set(by_arm) != set(readout.ARMS) or any(
                v is not None and (type(v) not in (float, int) or not isfinite(v)) for v in by_arm.values())
            for by_arm in cuts.values()):
        raise ValueError('Expected all frozen seed and arm cutoffs')


def calibrated_report(records, dates, cuts):
    result = {}
    for a in readout.ARMS:
        selected = {}
        for name, seeds in records.items():
            selected[name] = {}
            for s, r in seeds.items():
                t = cuts[s][a]
                edited = t is not None and r['scores'][a] >= t
                selected[name][s] = dict(edited=edited, **(r['arms'][a] if edited else r['arms']['cv']))
        values = [v for r in selected.values() for v in r.values() if v['edited']]
        result[a] = dict(edited_views=len(values), supported_edited_views=sum(v['risk']['supported'] for v in values),
            dual_edited_views=sum(v['dual_pass'] for v in values), fallback_views=3*len(records)-len(values),
            available_ade=readout.date_interval(selected, dates, lambda x: x['available_ade_native']),
            available_fde=readout.date_interval(selected, dates, lambda x: x['available_fde_native']),
            switch_rate=readout.date_interval(selected, dates, lambda x: x['intervention_rate']))
    return result
