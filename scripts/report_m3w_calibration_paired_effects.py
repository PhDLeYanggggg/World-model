"""Descriptive same-bank policy contrasts after the full calibration readout."""
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from src.evaluation import m3w_calibration_paired_effects as api
from src.world_model.m3w_european_source_forecast import baseline_numpy
from src.world_model.m3w_independent_risk_calibration import load_record

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_independent_risk_calibration_v1'


def checked(ref):
    path = ROOT/ref['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
        raise ValueError('Frozen artifact changed')
    return path


def read(path):
    return json.loads(Path(path).read_text())


def main():
    complete = read(PUBLIC/'eval.json'); summary = read(checked(complete['summary']))
    registration = read(PUBLIC/'registration.json'); roles = read(checked(registration['roles']))
    labels = read(PUBLIC/'labels.json')
    if len(labels['label_receipts']) != 102 or len(complete['private_metric_receipts']) != 918:
        raise ValueError('Complete frozen calibration grid and scoring required')
    regsha = hashlib.sha256((PUBLIC/'registration.json').read_bytes()).hexdigest()
    floors = {r['trial']:r['baseline_index'] for r in registration['frozen_models']}
    roster = {r['source_member']:r for r in roles['recordings'] if r['role'] == 'risk_calibration_reserved'}
    comparisons = ('joint_matched', 'joint_free', 'frame_uniform')
    distances = {(c['multiplier'], arm): {} for c in summary['curves'] for arm in comparisons}
    seen = set()
    for reference in complete['private_metric_receipts']:
        metrics_path = checked(reference); metric = read(metrics_path); member = metric['source_member']
        directory = metrics_path.parent; record = directory.parent
        if member not in roster or (member, metric['trial']) in seen:
            raise ValueError('Unique calibration recording/producer required')
        seen.add((member, metric['trial'])); site = roster[member]['locality_group']
        # Validate the input-only interface separately from supervised masks.
        _, data, _ = load_record(record, roles, member, regsha)
        label_ref = next(r for r in labels['label_receipts'] if Path(r['path']).parent == record.relative_to(ROOT))
        label = read(checked(label_ref)); valid = np.load(checked(label['valid']), allow_pickle=False)
        unknown = ~valid.any(1)
        policy = read(checked(metric['policy'])); choices = np.load(checked(policy['choices']), mmap_mode='r', allow_pickle=False)
        inf = read(checked(policy['inference'])); neural = np.load(checked(inf['neural']), mmap_mode='r', allow_pickle=False)
        h = data['history'][unknown]
        physical = np.stack([baseline_numpy(h, k) for k in range(6)], 1)
        paths = np.concatenate((physical, neural[unknown, None]), 1)
        if policy['multipliers'] != [c['multiplier'] for c in summary['curves']]:
            raise ValueError('Same full registered grid required')
        floor = floors[metric['trial']]
        for j, multiplier in enumerate(policy['multipliers']):
            control = choices[j, policy['policy_names'].index('independent'), unknown]
            aa = paths[np.arange(len(paths)), np.where(control < 0, floor, control)]
            for arm in comparisons:
                chosen = choices[j, policy['policy_names'].index(arm), unknown]
                bb = paths[np.arange(len(paths)), np.where(chosen < 0, floor, chosen)]
                bucket = distances[multiplier, arm]
                bucket[site] = bucket.get(site, 0.)+api.unknown_distance(aa, bb)
    results = []
    for curve in summary['curves']:
        control = curve['policies']['independent']['by_locality']
        for arm in comparisons:
            candidate = curve['policies'][arm]['by_locality']; effects = {}
            for site in sorted(control):
                if candidate[site]['denominator'] != control[site]['denominator']:
                    raise ValueError('Exactly matched known-reference denominator required')
                effects[site] = api.paired_values(control[site]['denominator'], control[site]['known_numerator'],
                    candidate[site]['known_numerator'], distances[curve['multiplier'], arm][site])
            results.append(dict(multiplier=curve['multiplier'], contrast=arm+'_minus_independent',
                effects={key:api.interval({s:v[key] for s,v in effects.items()}) for key in
                         ('known_pp', 'completion_lower_pp', 'completion_upper_pp')},
                unknown_pair_distance_by_locality=distances[curve['multiplier'], arm]))
    output = dict(result_source='fresh_run_post_scoring_descriptive_paired_calibration_analysis',
        frozen_predictions_and_known_sufficient_statistics='cached_verified',
        paired_unknown_distance_and_bootstrap='fresh_run',
        complete_pipeline_record_views=len(seen), comparisons=results,
        chosen_multiplier_unchanged=summary['selection']['selected_multiplier'],
        per_locality_reference_weighted_then_equal_locality_average=True,
        paired_unknown_distance_computed_directly=True, partial_missing_steps_not_recovered=True,
        empirical_calibration_not_independent_confirmation=True, new_model_or_threshold_fit=False,
        confirmation_access=False, deployment_changed=False)
    (PUBLIC/'paired_calibration_effects.json').write_text(json.dumps(output, indent=2, sort_keys=True)+'\n')
    lines = ['# Paired Calibration Policy Effects', '',
        'All fixed-grid contrasts are descriptive, not another selector or parameter search.',
        'Intervals resample 12 locality groups after pooling native reference mass within',
        'each locality. Calibration selection is not undone by these intervals.',
        'The unknown-row term is the actual policy-pair path disagreement, not the',
        'difference of two separate floor-relative lower bounds. Missing steps inside',
        'partially observed trajectories remain unaddressed.', '',
        '|Lambda|Contrast|Known pp [CI95]|Paired lower pp [CI95]|', '|---:|---|---:|---:|']
    def fmt(stat):
        if stat['point'] is None:
            return 'N/A'
        lo, hi = stat['CI95']; return f'{stat["point"]:.6f} [{lo:.6f}, {hi:.6f}]'
    for row in results:
        lines.append(f'|{row["multiplier"]}|{row["contrast"]}|{fmt(row["effects"]["known_pp"])}|'
                     f'{fmt(row["effects"]["completion_lower_pp"])}|')
    (PUBLIC/'paired_calibration_effects.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(pipeline_record_views=len(seen), fixed_grid_contrasts=len(results))))


if __name__ == '__main__':
    main()
