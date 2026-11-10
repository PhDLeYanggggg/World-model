"""Replay two distinct sources of uncertainty from frozen scalar receipts."""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import report_m3w_songdo_secondary as storage


PUBLIC = Path(__file__).resolve().parents[1]/'outputs/publication_readiness_2026_09'
OUT = PUBLIC/'external_completion_panel_v1'
SONGDO = PUBLIC/'songdo_calibrated_20261110'
WEP = PUBLIC/'wepdtof_missing_label_interval_v1'/'receipt.json'
DRAW_SEED, DRAWS = 1439, 3000


def summarize_songdo(contrast, groups, views):
    """All quantities share the same frozen component order and CV mass."""
    seen, lower = (contrast[k] for k in ('known_pp', 'lower_pp'))
    for r in (seen, lower):
        if (r['status'] != 'estimable' or r['estimable_components'] != groups
                or r['defined_seed_views'] != views or r['required_seed_views'] != views):
            raise ValueError('Observed and conservative contributions must use the entire frozen roster')
    values = np.column_stack((lower['component_points'], seen['component_points']))
    if values.shape != (groups, 2) or not np.isfinite(values).all() or (
            values[:, 0] > values[:, 1]+1e-12).any():
        raise ValueError('Cannot recover a nonnegative missing-label debit')
    upper = 2.*values[:, 1]-values[:, 0]
    values = np.column_stack((values, upper))
    points = values.mean(0)
    draws = np.random.default_rng(DRAW_SEED).integers(groups, size=(DRAWS, groups))
    ci = np.quantile(values[draws].mean(1), [.025, .975], axis=0).T
    for i, r in enumerate((lower, seen)):
        np.testing.assert_allclose(points[i], r['point'], rtol=0., atol=1e-12)
        np.testing.assert_allclose(ci[i], r['ci95'], rtol=0., atol=1e-12)
    loo = (values.sum(0)-values)/(groups-1)
    return dict(lower=float(points[0]), observed=float(points[1]), upper=float(points[2]),
                lower_ci95=ci[0].tolist(), observed_ci95=ci[1].tolist(), upper_ci95=ci[2].tolist(),
                leave_one_component_out_upper_range=[float(loo[:, 2].min()), float(loo[:, 2].max())],
                components_with_positive_lower=int((values[:, 0] > 0.).sum()),
                components_with_negative_upper=int((values[:, 2] < 0.).sum()))


def audit():
    p, hp = storage.load(SONGDO/'confirmation_receipt.json')
    c, hc = storage.load(SONGDO/'calibration_receipt.json')
    w, hw = storage.load(WEP)
    if ((p['flights'], p['source_components'], p['matched_seed_views'], p['calibration_flights']) != (160, 14, 480, 40)
            or p['matching'] != 'exactly_two_edits_same_query_and_peak_radius'
            or p['binding']['published_calibration_receipt_sha256'] != hc
            or not p['no_fitting_on_confirmation'] or not p['original_2pct_unchanged'] or p['export_read_labels']
            or c['confirmation_outcomes_opened'] or not c['original_2pct_unchanged']
            or w['outcome_access'] or not w['original_2pct_unchanged']
            or (w['sources'], w['matched_estimable_sources'], w['views']) != (13, 10, 45)):
        raise ValueError('Require the frozen source-disjoint/read-only evaluation receipts')
    songdo = {a: summarize_songdo(r, p['source_components'], p['matched_seed_views'])
              for a, r in p['matched_joint_minus'].items()}
    row = w['comparisons']['fixed_independent']
    e = row['estimates']
    if not np.allclose(row['identification_envelope_pp'], [e[k]['estimate'] for k in ('lower', 'upper')],
                       atol=1e-12, rtol=0.) or not np.isclose(e['observed']['estimate'],
                       np.mean(row['identification_envelope_pp']), atol=1e-12, rtol=0.):
        raise ValueError('WEPDTOF point envelope changed')
    w_curve = {**{k: e[k]['estimate'] for k in ('lower', 'observed', 'upper')},
               **{f'{k}_ci95': e[k]['ci95'] for k in ('lower', 'observed', 'upper')}}
    if w['original_fixed_joint_dual_passes'] != 0 or p['released']['joint']['risk']['edited'] != 0:
        raise ValueError('Source policy result changed')
    return dict(result_source='fresh_scalar_replay_of_frozen_results', source_hashes=dict(songdo=hp, calibration=hc, wepdtof=hw),
        scale='percentage points of each source observed CV error, not pooled', primary_contrast='joint_minus_independent',
        wepdtof=dict(curve=w_curve, estimable_sources=w['matched_estimable_sources'], total_sources=w['sources'],
                     matched_seed_views=w['views'], original_joint_dual_passes=w['original_fixed_joint_dual_passes']),
        songdo=dict(controls=songdo, components=p['source_components'], flights=p['flights'],
                    matched_seed_views=p['matched_seed_views'], joint_released=p['released']['joint']['risk']['edited'],
                    independent_released=p['released']['independent']['risk']['edited'],
                    independent_released_dual=p['released']['independent']['risk']['dual_pass']),
        sampling_interval_targets_conservative_endpoint=True, missing_label_interval_conditional_on_sample=True,
        actual_new_outcome_access=False, original_2pct_unchanged=True, release_changed=False, submission_ready=False)


if __name__ == '__main__':
    result = audit()
    storage.write(OUT/'readout.json', result)
    print(json.dumps({k: result[k] for k in ('wepdtof', 'songdo')}, indent=2))
