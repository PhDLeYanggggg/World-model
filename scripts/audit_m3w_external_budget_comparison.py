"""Compare frozen external cohorts without pooling their errors or release counts."""

import argparse
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path
from random import Random
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'
OUT = PUBLIC/'external_budget_comparison_v1/readout.json'
SOURCES = {
    'wep': 'wepdtof_external_transfer_v1/external_readout.json',
    'wep_steps': 'wepdtof_equal_step_v1/readout.json',
    'songdo': 'songdo_calibrated_20261110/confirmation_receipt.json',
    'songdo_steps': 'songdo_step_budget_v1/readout.json',
    'athens': 'pneuma_confirmation_v1/confirmation.json',
    'athens_steps': 'pneuma_confirmation_v1/stepwise_budget_sensitivity.json',
}


def contrast(metrics, source_values, fields):
    points = [source_values(k) for k in fields]
    if len({len(v) for v in points}) != 1:
        raise ValueError('Incompatible source counts')
    roster = [i for i, v in enumerate(points[0]) if v is not None]
    if not roster or any([i for i, v in enumerate(p) if v is not None] != roster for p in points):
        raise ValueError('Incomplete paired source support')
    rows = [[p[i] for p in points] for i in roster]
    if any(not all(isfinite(v) for v in row) or not row[0] <= row[1] <= row[2] for row in rows):
        raise ValueError('Invalid source envelope')
    center = [mean(v) for v in zip(*rows)]
    if not all(isclose(v, metrics[k]['point'], abs_tol=1e-12) for k, v in zip(fields, center)):
        raise ValueError('Means do not match the frozen score')
    return dict(eligible_indices=roster, group_scores=rows, envelope_pp=[center[0], center[2]],
                observed_pp=center[1], lower_endpoint_ci95_pp=metrics[fields[0]]['ci95'])


def compare(original, matched):
    if original['eligible_indices'] != matched['eligible_indices']:
        raise ValueError('Movement-matched sensitivity must keep the original source roster')
    delta = [b[1]-a[1] for a, b in zip(original['group_scores'], matched['group_scores'], strict=True)]
    rng = Random(20261110)
    draws = sorted(mean(rng.choices(delta, k=len(delta))) for _ in range(3000))

    def quantile(p):
        i = p*(len(draws)-1)
        j = int(i)
        return draws[j]+(i-j)*(draws[j+1]-draws[j])

    return dict(original=original, equal_steps=matched,
                equal_steps_minus_original_observed_pp=mean(delta),
                paired_source_ci95_pp=[quantile(.025), quantile(.975)],
                bootstrap_draws=len(draws), units_resampled=len(delta))


def replay(raw):
    d = {k: json.loads(raw[k]) for k in SOURCES}
    hashes = {k: sha256(v).hexdigest() for k, v in raw.items()}
    w, ws = d['wep'], d['wep_steps']
    s, ss = d['songdo'], d['songdo_steps']
    a, ats = d['athens'], d['athens_steps']
    if (not w['external_scoring_completed'] or w['external_tuning'] or w['deployment_changed']
            or not w['original_2pct'] or not ws['original_2pct']
            or ws['new_external_targets_opened'] or ws['original_release_changed']
            or not ws['previously_scored_outcomes_reused'] or ws['independent_confirmation']
            or not s['no_fitting_on_confirmation'] or not s['original_2pct_unchanged']
            or not ss['diagonal_reproduced'] or not ss['original_2pct']
            or ss['new_cohort_opened'] or ss['deployment_changed'] or ss['unmatched_views']
            or a['release'] != 'not_run_zero_shot_fixed_count_only' or a['official_risk_budget'] != .02
            or ats['primary_confirmation'] != hashes['athens'] or not ats['primary_unchanged']
            or ats['thresholds_changed'] or ats['release_recomputed']):
        raise ValueError('External cohorts must retain their original roles and release rules')

    def wep(k):
        arm = 'fixed_independent'
        return contrast(ws['by_comparator'][arm][k], ws['source_values'][arm][k].get,
                        ('lower', 'observed', 'upper'))

    def songdo(k):
        m = ss['variants'][k]['direct']
        return contrast(m, lambda f: m[f]['source_means'], ('lower_pp', 'known_pp', 'upper_pp'))

    def athens(k):
        m = ats['by_comparator']['independent'][k]
        return contrast(m, lambda f: [x for _, x in sorted(m[f]['by_date'].items())],
                        ('lower_pp', 'observed_pp', 'upper_pp'))

    original = dict(wep=wep('original'), songdo=songdo('original'), athens=athens('original'))
    prior = (w['summary']['matched_comparisons']['fixed_independent']['estimate'],
             s['matched_joint_minus']['independent']['lower_pp']['point'],
             a['matched']['joint_minus']['independent']['lower_pp']['point'])
    if any(not isclose(original[k]['envelope_pp'][0], x, abs_tol=1e-12)
           for k, x in zip(original, prior)):
        raise ValueError('Original lower endpoint does not reproduce the primary result')
    for i, key in enumerate(('lower_pp', 'known_pp')):
        recorded = s['matched_joint_minus']['independent'][key]['component_points']
        if any(not isclose(r[i], x, abs_tol=1e-12)
               for r, x in zip(original['songdo']['group_scores'], recorded, strict=True)):
            raise ValueError('Songdo source pairing differs from primary')
    primary_athens = a['matched']['joint_minus']['independent']
    for i, key in enumerate(('lower_pp', 'observed_pp', 'upper_pp')):
        recorded = [v for _, v in sorted(primary_athens[key]['by_date'].items())]
        if [row[i] for row in original['athens']['group_scores']] != recorded:
            raise ValueError('Athens date pairing differs from primary')

    rows = {k: compare(original[k], fn('matched_steps')) for k, fn in
            (('wep', wep), ('songdo', songdo), ('athens', athens))}
    if [rows[k]['units_resampled'] for k in rows] != [10, 14, 4]:
        raise ValueError('Unexpected number of independent aggregation units')
    wa, sa, aa = w['summary']['arms'], s['released'], a['matched']['by_arm']['joint']['risk']
    rows['wep'].update(total_groups=w['summary']['sources'], group_unit='source',
                      joint_fixed_dual=[wa['fixed_joint']['dual_passes'], w['summary']['recording_seed_views']],
                      joint_released_edits=wa['joint']['edits'], release='evaluated')
    rows['songdo'].update(total_groups=s['source_components'], group_unit='source component',
                         joint_fixed_dual=[s['matched']['joint']['risk']['dual_pass'], s['matched_seed_views']],
                         joint_released_views=sa['joint']['risk']['edited'],
                         independent_released_views=sa['independent']['risk']['edited'],
                         independent_released_dual=sa['independent']['risk']['dual_pass'], release='evaluated')
    rows['athens'].update(total_groups=len(set(a['dates'].values())), group_unit='acquisition date',
                         joint_fixed_dual=[aa['dual_pass'], aa['views']], release='not_run')
    return dict(status='fresh_aggregate_comparison', source_sha256=hashes, cohorts=rows,
                pooling=False, new_outcomes_read=False, new_fit=False,
                original_risk_fraction=.02, release_recomputed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    raw = {k: (PUBLIC/p).read_bytes() for k, p in SOURCES.items()}
    content = (json.dumps(replay(raw), indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        with args.output.open('xb') as f:
            f.write(content)
    elif args.output.read_bytes() != content:
        raise FileExistsError(args.output)
    print(args.output)


if __name__ == '__main__':
    main()
