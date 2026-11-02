"""Actual public result reconstruction, not model retraining or confirmation."""
import hashlib
import json
from pathlib import Path

import numpy as np

from src.world_model import m3w_independent_risk_calibration as calibration
from src.evaluation import m3w_calibration_paired_effects as paired

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_independent_risk_calibration_v1'


def read(name):
    return json.loads((PUBLIC/name).read_text())


def test_complete_metric_replay_and_unchanged_original_selection():
    summary = read('summary.json'); replay = read('verify.json')
    assert summary['selection'] == calibration.select_multiplier(summary['curves'], 918)
    assert summary['metric_groups'] == 47736 and summary['unique_recordings'] == 102
    assert replay['all47736_metric_groups_exact']
    assert replay['policy_feasibility_and_per_query_count_replayed']
    assert not replay['milp_reoptimized']
    assert replay['summary_sha256'] == hashlib.sha256((PUBLIC/'summary.json').read_bytes()).hexdigest()
    assert not any(summary[k] for k in ('confirmation_access', 'population_risk_certificate',
        'complete12_ADE_claim', 't50_or_metric_claim', 'deployment_changed',
        'submission_ready', 'stage5c_executed', 'smc_enabled'))


def test_all_public_locality_points_and_bootstrap_intervals_reconstruct_exactly():
    registration = read('registration.json'); ref = registration['roles']; path = ROOT/ref['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ref['sha256']
    roles = json.loads(path.read_text())
    records = sorted((r for r in roles['recordings'] if r['role'] == 'risk_calibration_reserved'),
                     key=lambda r:r['source_member'])
    # The original evaluator uses first-encounter order, not sorted JSON keys.
    order = list(dict.fromkeys(r['locality_group'] for r in records))
    config = json.loads((ROOT/'configs/m3w_independent_risk_calibration_v1.json').read_text())
    assert len(order) == 12 and config['bootstrap_resamples'] == 3000
    for curve in read('summary.json')['curves']:
        for arm in curve['policies'].values():
            rng = np.random.default_rng(config['bootstrap_seed'])
            for kind in ('known', 'lower'):
                sites = arm['by_locality']
                values = [100*sites[s][kind+'_numerator']/sites[s]['denominator']
                          if sites[s]['denominator'] > 0 else None for s in order]
                saved = arm[kind+'_gain_percent']
                if None in values:
                    assert saved['point'] is None and saved['CI95'] is None
                else:
                    draws = rng.choice(values, size=(3000, len(order))).mean(1)
                    assert saved['point'] == float(np.mean(values))
                    np.testing.assert_array_equal(saved['CI95'], np.quantile(draws, [.025, .975]))


def test_paired_statistics_use_actual_pair_distance_and_common_reference():
    curves = {r['multiplier']:r['policies'] for r in read('summary.json')['curves']}
    output = read('paired_calibration_effects.json')
    assert len(output['comparisons']) == 39
    for row in output['comparisons']:
        policies = curves[row['multiplier']]
        a = policies['independent']['by_locality']
        b = policies[row['contrast'].removesuffix('_minus_independent')]['by_locality']
        values = {}
        for site in sorted(a):
            assert a[site]['denominator'] == b[site]['denominator']
            values[site] = paired.paired_values(a[site]['denominator'], a[site]['known_numerator'],
                b[site]['known_numerator'], row['unknown_pair_distance_by_locality'][site])
        for key, saved in row['effects'].items():
            assert saved == paired.interval({s:v[key] for s, v in values.items()})
