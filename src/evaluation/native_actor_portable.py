"""Standalone native-stride aggregate reductions; standard library only."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

SCOPE = 'native_moving_drone_observation8_prediction12_stride1_exposed_A'
FALSE_CLAIMS = ('new_training', 'new_prediction_readout', 'independent_confirmation',
    'full_training_reproduction', 'risk_certified', 'deployment', 'submission_ready',
    'stage5c_executed', 'smc_enabled')


def close(actual, expected):
    if not (math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9)):
        raise ValueError('Scalar reconstruction differs from archived evidence')


def quantile(values, p):
    ordered = sorted(values); index = (len(ordered)-1)*p
    low = int(math.floor(index)); high = int(math.ceil(index)); frac = index-low
    return ordered[low]+frac*(ordered[high]-ordered[low])


def reconstruct(evidence):
    if (evidence['schema_version'] != 1 or evidence['scope'] != SCOPE
            or evidence['risk_budget'] != .02 or any(evidence[k] is not False for k in FALSE_CLAIMS)):
        raise ValueError('Changed data role,coordinate/time scope or unsupported claim')
    matrices = evidence['bootstrap_indices_by_seed']
    if set(matrices) != {'39017', '39107'}: raise ValueError('Keep both original registered bootstrap seeds')
    for draws in matrices.values():
        if (len(draws) != 3000 or any(len(row) != 14 for row in draws)
                or any(type(i) is not int or i < 0 or i >= 14 for row in draws for i in row)):
            raise ValueError('Complete original source-bootstrap index matrix required')
    if len(evidence['experiments']) != 3: raise ValueError('Retain all three frozen experiment primaries')
    results = []
    for experiment in evidence['experiments']:
        draws = matrices[str(experiment['bootstrap_seed'])]
        rows = experiment['sources']
        if (len(rows) != 14 or len({r['source'] for r in rows}) != 14
                or {r['producer_fold'] for r in rows} != {0, 1}
                or sum(r['prefixes'][-1]['rows'] for r in rows) != 125540):
            raise ValueError('Changed complete source population')
        for row in rows:
            if len(row['prefixes']) != 4: raise ValueError('All1/3/6/12prefixes required')
            for summary in row['prefixes']:
                values = [summary[k] for k in ('known_pp', 'lower_pp', 'upper_pp')]
                if any(v is None or not math.isfinite(v) for v in values):
                    raise ValueError('Undefined values cannot be omitted or replaced with zero')
                if not values[1]-1e-9 <= values[0] <= values[2]+1e-9:
                    raise ValueError('Known effect outside completion envelope')
                if (summary['known_rows']+summary['wholly_unknown_rows'] != summary['rows']
                        or summary['complete_rows']+summary['partial_rows'] != summary['known_rows']):
                    raise ValueError('Changed known,partial or unknown population accounting')
        known = [r['prefixes'][-1]['known_pp'] for r in rows]
        lower = [r['prefixes'][-1]['lower_pp'] for r in rows]
        average = lambda values: math.fsum(values)/len(values)
        replicates = [average([lower[i] for i in draw]) for draw in draws]
        interval = [quantile(replicates, p) for p in (.025, .975)]
        folds = {str(f): average([r['prefixes'][-1]['known_pp'] for r in rows if r['producer_fold'] == f]) for f in (0, 1)}
        positives = sum(v > 0 for v in known)
        screen = average(lower) > 0 and interval[0] > 0 and positives >= 8 and min(folds.values()) > 0
        expected = experiment['archived_primary']
        close(average(known), expected['known_pp']); close(average(lower), expected['completion_lower_pp'])
        for a, b in zip(interval, expected['interval95']): close(a, b)
        for key in folds: close(folds[key], expected['fold_known_means'][key])
        if positives != expected['positive_known_sources'] or screen != expected['component_progression_screen_pass']:
            raise ValueError('Archived negative screen or source count changed')
        results.append(dict(experiment=experiment['label'], known_pp=average(known), lower_pp=average(lower),
            interval95=interval, positive_known_sources=positives, fold_known_means=folds,
            component_screen_pass=screen, prefix_known_pp=[average([r['prefixes'][i]['known_pp'] for r in rows]) for i in range(4)]))
    return dict(scope=SCOPE, rows_per_experiment=125540, source_groups=14, forecaster_seeds=1,
        experiments=results, numerical_tolerance=dict(relative=1e-12, absolute=1e-9),
        uncertainty='replayed3000fixed_registered_source_draws_not_new_independent_bootstrap',
        **{k: False for k in FALSE_CLAIMS})


def outputs(result):
    stream = io.StringIO(newline='')
    fields = ('experiment', 'known_pp', 'lower_pp', 'CI_low', 'CI_high', 'positive_sources', 'screen_pass')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n'); writer.writeheader()
    for row in result['experiments']:
        writer.writerow(dict(experiment=row['experiment'], known_pp=f'{row["known_pp"]:.9f}',
            lower_pp=f'{row["lower_pp"]:.9f}', CI_low=f'{row["interval95"][0]:.9f}', CI_high=f'{row["interval95"][1]:.9f}',
            positive_sources=row['positive_known_sources'], screen_pass=row['component_screen_pass']))
    return {'primary_table.csv': stream.getvalue(), 'replayed.json': json.dumps(result, indent=2, allow_nan=False)+'\n'}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--verify', action='store_true')
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    root = Path(__file__).resolve().parent; manifest = json.loads((root/'manifest.json').read_text())
    if manifest['package_scope'] != 'deidentified_native_actor_aggregate_v1': raise ValueError('Changed package scope')
    for name, info in manifest['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe artifact path')
        path = root/relative
        if any(p.is_symlink() for p in (path, *path.parents) if p != root.parent): raise ValueError('Symlink in artifact')
        value = path.read_bytes()
        if len(value) != info['bytes'] or hashlib.sha256(value).hexdigest() != info['sha256']:
            raise ValueError('Changed package file')
    result = reconstruct(json.loads((root/'evidence.json').read_text())); generated = outputs(result)
    for name, value in generated.items():
        if hashlib.sha256(value.encode()).hexdigest() != manifest['expected_outputs'][name]:
            raise ValueError('Reconstructed output hash differs')
        if args.verify and (root/'expected'/name).read_text() != value: raise ValueError('Changed expected output')
    target = args.output.resolve()
    if target == root or root in target.parents and target.parts[len(root.parts)] == 'expected':
        raise ValueError('Do not overwrite evidence')
    target.mkdir(parents=True, exist_ok=True)
    if any((target/name).exists() or (target/name).is_symlink() for name in generated):
        raise ValueError('Refusing to overwrite existing reconstruction')
    for name, value in generated.items(): (target/name).write_text(value)
    print(json.dumps(dict(result_source='aggregate_replay_only', experiments=3, source_groups=14,
        fixed_bootstrap_draws=3000, stdlib_only=True, new_training=False, independent_confirmation=False,
        full_training_reproduction=False, submission_ready=False)))


if __name__ == '__main__': main()
