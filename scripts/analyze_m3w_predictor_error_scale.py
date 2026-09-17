"""Explain fixed primary errors from verified rows, without selecting a new metric."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_development_evaluation import content_digest
from scripts.evaluate_m3w_forecast_supplement import verify_completed_evaluation


def decomposition(rows):
    eligible = [r for r in rows if r['baseline_ade'] is not None]
    if not eligible:
        raise ValueError('Complete-path ADE labels required')
    b = np.array([r['baseline_ade'] for r in eligible])
    c = np.array([r['arms']['uncontrolled']['ade'] for r in eligible])
    scale = np.array([r['scale'] for r in eligible])
    scenes = np.array([r['physical_scene'] for r in eligible])
    unique, counts = np.unique(scenes, return_counts=True)
    weights = np.zeros(len(eligible))
    for scene, count in zip(unique, counts):
        weights[scenes == scene] = 1 / (len(unique)*count)
    if not np.isfinite([b, c, scale]).all() or (scale <= 0).any():
        raise ValueError('Finite matched errors and positive past scales required')
    # This is the frozen transform's numerical floor, not a physical distance.
    at_floor = np.isclose(scale, .001, rtol=1e-7, atol=0)
    harm = np.maximum(c-b, 0)
    result = {'complete_agent_queries': len(b), 'physical_scene_count': len(unique),
        'aggregation': 'equal_physical_scene', 'primary_baseline_ade': float(weights@b),
        'primary_candidate_ade': float(weights@c), 'net_excess': float(weights@(c-b)),
        'positive_harm': float(weights@harm), 'bins': {}, 'per_recording_native_only': {}}
    for name, mask in (('numerical_scale_floor', at_floor), ('above_numerical_floor', ~at_floor)):
        contribution = float(weights[mask]@(c-b)[mask])
        harm_mass = float(weights[mask]@harm[mask])
        result['bins'][name] = {'count': int(mask.sum()), 'query_fraction': float(mask.mean()),
            'net_excess_contribution_to_primary_ade': contribution,
            'positive_harm_contribution_to_primary_ade': harm_mass,
            'fraction_of_all_positive_harm': harm_mass/result['positive_harm'] if result['positive_harm'] > 0 else None}
    for recording in sorted({r['recording_id'] for r in eligible}):
        mask = np.array([r['recording_id'] == recording for r in eligible])
        native_b, native_c = b[mask]*scale[mask], c[mask]*scale[mask]
        result['per_recording_native_only'][recording] = {
            'count': int(mask.sum()), 'baseline_ade': float(native_b.mean()), 'candidate_ade': float(native_c.mean()),
            'gain_pct': float(100*(native_b.mean()-native_c.mean())/native_b.mean()) if native_b.mean() > 0 else None,
            'candidate_positive_harm_native_mean': float(np.maximum(native_c-native_b, 0).mean()),
            'coordinate_claim': 'dataset_local_unverified_not_pooled_with_other_recordings'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    heartbeat = json.loads((args.study_dir/'runner_heartbeat.json').read_text())
    if heartbeat.get('state') != 'requested_seeds_complete' or heartbeat['seeds'] != [17, 29, 43]:
        raise ValueError('Full registered study required; no cherry-picked seed diagnosis')
    results, sources = {}, {}
    for seed in heartbeat['seeds']:
        directory = args.study_dir/f'seed{seed}_development'
        verify_completed_evaluation(directory, file_digest)
        identity = json.loads((directory/'run_identity.json').read_text())
        candidate = f'seed{seed}_ridge_conservative'
        rows = []
        for recording in ('ucy_students01', 'ucy_students03'):
            key = [candidate, recording]
            stem = content_digest(key)
            cache, receipt = directory/(stem+'.rows.json'), directory/(stem+'.receipt.json')
            digest = file_digest(cache)
            if json.loads(receipt.read_text()) != {'key': key, 'run_sha256': content_digest(identity), 'cache_sha256': digest}:
                raise ValueError('Primary cache identity mismatch')
            sources[str(cache)] = digest
            rows.extend(json.loads(cache.read_text()))
        results[str(seed)] = decomposition(rows)
    report = {'result_source': 'fresh_run_analysis_of_hash_verified_completed_evaluation_rows',
        'protocol_sha256': heartbeat['protocol_sha256'], 'source_sha256': sources, 'seeds': results,
        'code_sha256': file_digest(Path(__file__)), 'interpretation': 'descriptive_decomposition_not_causal_attribution',
        'metric_or_threshold_changed': False, 'independent_confirmation': False, 'new_deployment': False}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'error_scale.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['# Fixed-Result Error-Scale Decomposition', '',
        'All registered seeds, unchanged predictions/labels/metric. Equal physical-scene weighting; raw errors stay recording-local.', '',
        '| Seed | Primary net excess | Floor-scale queries % | Positive harm attributable to floor-scale rows % | Floor net-excess contribution | Other rows contribution |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for seed, r in results.items():
        a, b = r['bins']['numerical_scale_floor'], r['bins']['above_numerical_floor']
        fraction = a['fraction_of_all_positive_harm']
        lines.append(f"| {seed} | {r['net_excess']:.6g} | {100*a['query_fraction']:.3f} | {100*fraction if fraction is not None else 'undefined'} | {a['net_excess_contribution_to_primary_ade']:.6g} | {b['net_excess_contribution_to_primary_ade']:.6g} |")
    lines += ['', 'The two signed net-excess contributions sum to the original primary error difference. The 0.001 cutoff is the already-used numerical normalization floor, not a new physical threshold or a sample-exclusion rule.',
        'This locates error mass, not the causal reason a network fails. It cannot prove that changing normalization improves an independently evaluated method. Better native-coordinate results are sensitivity evidence, not a replacement primary result.',
        'Future labels are used only for this post-run diagnosis. No model, support rule, loss, threshold or deployment choice is modified. No scene-level confidence interval is inferred from one physical development site.', '']
    (args.output_dir/'error_scale.md').write_text('\n'.join(lines))
    print(json.dumps({s: r['bins'] for s, r in results.items()}))


if __name__ == '__main__':
    main()
