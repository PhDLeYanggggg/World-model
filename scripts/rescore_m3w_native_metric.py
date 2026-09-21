"""Versioned source-only readout after the delegated evaluation amendment."""
import argparse
import json
from pathlib import Path
import platform
import sys

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.data_unification.m3w_causal_recordings import BASELINES
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_native_metrics import (
    native_errors, paired_scene_metrics, select_complement_baseline,
)
from src.evaluation.m3w_source_population import scene_mean, safe_gain

CONFIG = 'configs/m3w_native_metric_v1.json'
OLD = 'outputs/publication_readiness_2026_09/'


def compute():
    evidence = {}

    def check(path, expected=None):
        sha = file_digest(ROOT/path)
        if expected is not None and sha != expected:
            raise ValueError('Changed bound input: '+path)
        evidence[path] = sha

    def read(path):
        return json.loads((ROOT/path).read_text())

    reg = read(CONFIG)
    for path in (CONFIG, reg['decision'], 'scripts/rescore_m3w_native_metric.py',
                 'src/evaluation/m3w_native_metrics.py', 'tests/test_m3w_native_metrics.py'):
        check(path)
    for path, sha in reg['bindings'].items():
        check(path, sha)
    parent = read(OLD+'source_population_v1/audit.json')
    for path, sha in parent['evidence_hashes'].items():
        check(path, sha)
    archive = parent['row_archive']
    check(archive['path'], archive['sha256'])
    with np.load(ROOT/archive['path'], allow_pickle=False) as z:
        a = {k:z[k] for k in z.files}
    if len(a['scale']) != 175756 or sorted(set(a['sites'])) != reg['sites']:
        raise ValueError('Changed admitted population')
    native_ade, native_fde = a['ade']*a['scale'][:, None], a['fde']*a['scale'][:, None]
    manifest = read('data/stage_cvpr2027_experiments/sdd_auxiliary_v1/inputs/manifest.json')
    folder = ROOT/'data/stage_cvpr2027_experiments/sdd_auxiliary_v1/inputs'
    target_parts = []
    # Rebuild native costs from original paired trajectories, not just aggregate reports.
    for rec in manifest['records']:
        key = rec['recording']
        if key.split('/')[0] not in reg['sites']:
            continue
        ids = np.flatnonzero(a['recordings'] == key)
        g, y, valid, scale, keys = [np.load(folder/key/(k+'.npy'), mmap_mode='r', allow_pickle=False)
                                  for k in ('geometry', 'target', 'valid', 'scale', 'query_keys')]
        if len(ids) != rec['rows']:
            raise ValueError('Recording membership mismatch')
        np.testing.assert_array_equal(a['frames'][ids], keys[:, 0])
        np.testing.assert_array_equal(a['tracks'][ids], [f'{key}:{int(v)}' for v in keys[:, 1]])
        for k in range(len(BASELINES)):
            ade, fde = native_errors(g[:, 308+24*k:308+24*(k+1)].reshape(-1, 12, 2), y, valid, scale)
            np.testing.assert_allclose(ade, native_ade[ids, k], rtol=1e-12, atol=1e-10, equal_nan=True)
            np.testing.assert_allclose(fde, native_fde[ids, k], rtol=1e-12, atol=1e-10, equal_nan=True)
        stationary = a['static_history'][ids] & (a['valid_count'][ids] == 12)
        target_parts.append(np.array(y[stationary]))
    print(json.dumps(dict(state='native_costs_reconstructed', rows=len(a['scale']), recordings=33)), flush=True)

    def score(m, r, sites):
        return paired_scene_metrics(m, r, sites, expected_scenes=reg['sites'], dataset=reg['dataset'],
            coordinate_unit=reg['coordinate_unit'], bootstrap_resamples=reg['bootstrap_resamples'],
            seed=reg['bootstrap_seed'])

    cohorts = {}
    for name, mask in [('supported_masked', np.ones(len(a['scale']), bool)),
                       ('complete_future', a['valid_count'] == 12)]:
        e, f, sites = native_ade[mask], native_fde[mask], a['sites'][mask]
        choice, selection = select_complement_baseline(e, sites, reg['sites'])
        selected_e, selected_f = e[np.arange(len(e)), choice], f[np.arange(len(f)), choice]
        methods = {}
        for k, baseline in enumerate(BASELINES):
            methods[baseline] = dict(ADE_vs_CV=score(e[:, k], e[:, 1], sites),
                FDE_vs_CV=score(f[:, k], f[:, 1], sites),
                ADE_vs_complement_selected=score(e[:, k], selected_e, sites))
        # Oracle choice uses ADE labels only; use that SAME candidate's FDE.
        support = np.isfinite(e).all(1)
        oracle = np.zeros(len(e), int)
        oracle[support] = np.argmin(e[support], axis=1)
        methods['oracle_diagnostic_not_deployable'] = dict(
            ADE_vs_CV=score(e[np.arange(len(e)), oracle], e[:, 1], sites),
            FDE_vs_CV=score(f[np.arange(len(f)), oracle], f[:, 1], sites),
            ADE_vs_complement_selected=score(e[np.arange(len(e)), oracle], selected_e, sites))
        methods['complement_selected_baseline'] = dict(ADE_vs_CV=score(selected_e, e[:, 1], sites),
            FDE_vs_CV=score(selected_f, f[:, 1], sites), selection=selection)
        cohorts[name] = dict(methods=methods, unknown_ADE_rows=int((~support).sum()),
                             old_metric_unchanged=parent['cohorts']['complete' if name == 'complete_future' else name])

    neural_parent = read(OLD+'source_crossfit_v1/report.json')
    check('configs/m3w_source_crossfit_v1.json', neural_parent['identity']['registration_sha256'])
    for path, sha in read('configs/m3w_source_crossfit_v1.json')['bindings'].items():
        check(path, sha)
    for trial in neural_parent['trials']:
        fold = trial['identity']['fold']
        if set(fold['training_sites']) != set(reg['sites'])-{trial['site']} or fold['fitted_parent'] is not None:
            raise ValueError('Changed neural source exclusion')
        for prefix in ('checkpoint', 'prediction', 'parent'):
            check(trial[prefix+'_path'], trial[prefix+'_sha256'])

    static = a['static_history'] & (a['valid_count'] == 12)
    target = np.concatenate(target_parts)
    sites, scale = a['sites'][static], a['scale'][static]
    np.testing.assert_array_equal(native_ade[static], np.repeat(native_ade[static, 1, None], len(BASELINES), axis=1))
    # Historical source IDs include a closed-site offset. Read counts, not its arrays.
    support = read(OLD+'source_site_probe_v1/support_audit.json')
    main_count = read(OLD+'source_start_probe_v1/input_checks.json')['main_rows']
    closed_count = next(v['rows'] for v in support['sites'] if v['site'] == 'bookstore')
    names = [r['recording'].split('/')[0] for r in manifest['records']]
    if names[:7] != ['bookstore']*7 or 'bookstore' in names[7:] or static.sum() != 15430:
        raise ValueError('Historical source offset assumptions no longer valid')
    expected_ids = np.arange(main_count+closed_count, main_count+closed_count+int(static.sum()))
    seeds = {}
    neural_costs = []
    for entry in neural_parent['oof_labels']:
        check(entry['path'], entry['sha256'])
        with np.load(ROOT/entry['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], expected_ids)
            p = z['prediction']
            ade, fde = native_errors(p, target, np.ones(target.shape[:2], bool), scale)
            np.testing.assert_allclose(ade/scale, z['ade'], rtol=1e-12, atol=1e-9)
            np.testing.assert_allclose(fde/scale, z['fde'], rtol=1e-12, atol=1e-9)
            np.testing.assert_allclose(native_ade[static, 1]/scale, z['cv_ade'], rtol=1e-12, atol=1e-9)
        easy = a['event'][static] == 0
        seeds[str(entry['seed'])] = dict(ADE_vs_CV=score(ade, native_ade[static, 1], sites),
            FDE_vs_CV=score(fde, native_fde[static, 1], sites),
            zero_CV_easy_ADE=score(ade[easy], native_ade[static, 1][easy], sites[easy]),
            old_normalized_gain_percent=safe_gain(scene_mean(ade/scale, sites),
                                                   scene_mean(a['ade'][static, 1], sites)))
        neural_costs.append(ade)
    if sorted(map(int, seeds)) != reg['seeds']:
        raise ValueError('All registered seeds required')
    neural = dict(status='fresh_run_rescore_of_cached_verified_predictions', rows=int(static.sum()),
        population='original_static_history_complete_labels_only_not_full_population',
        full_population_neural_evaluation='not_run_no_matched_full_population_OOF_predictions_in_this_readout',
        all_seeds=seeds, mean_per_seed_errors_ADE=score(np.mean(neural_costs, axis=0), native_ade[static, 1], sites),
        aggregation='mean_seed_errors_not_prediction_ensemble', new_inference=False, new_training=False)
    # Refuse a source change during the readout, including metadata-only dependencies.
    for path, sha in evidence.items():
        if file_digest(ROOT/path) != sha:
            raise ValueError('Dependency changed during readout: '+path)
    return reg, dict(result_source='fresh_run_native_scoring_cached_verified_inputs_predictions',
        authority=reg['authority'], new_primary='native_ADE_equal_scene_mean_relative_gain',
        indexed_rows=len(a['scale']), source_sites=reg['sites'], source_recordings=33,
        verified_source_arrays=198, reconstructed_baseline_query_pairs=len(a['scale'])*len(BASELINES),
        cohorts=cohorts, existing_neural_subset=neural, evidence_hashes=evidence,
        old_reports_modified=False, data_roles_changed=False, new_training=False,
        new_inference=False, original_val_test_rows=0, main_outer_external_rows=0,
        independent_confirmation=False, deployment=False, stage5c_executed=False, smc_enabled=False)


def render(r):
    lines = ['# Native-Coordinate Source Readout', '', '## Material Passport', '',
        'Fresh paired score computation; cached/hash-verified inputs and predictions.',
        'Post-hoc evaluation amendment, four previously explored SDD training sites.',
        'No training, new inference, independent test, threshold search or deployment.',
        'SDD annotation pixels, 8 observed / 12 predicted steps, K=1; no meters/seconds.', '',
        '| Cohort | Method | Equal-scene ADE gain vs CV (%) | Scene bootstrap 95% interval | Worst scene (%) |',
        '| --- | --- | ---: | --- | ---: |']
    for cohort, c in r['cohorts'].items():
        for name, m in c['methods'].items():
            v = m['ADE_vs_CV']
            lines.append(f"| {cohort} | {name} | {v['equal_scene_gain_percent']} | {v['scene_bootstrap_ci95']} | {v['worst_scene_gain_percent']} |")
    lines += ['', 'All scene-native ADE/FDE, tail errors and unchanged old metrics are in analysis.json.',
        'Oracle uses future ADE labels and is not an executable policy. Its FDE uses the same ADE-selected candidate.',
        'Complement baseline is selected only on other source sites, not this site.', '',
        '## Existing Neural Predictions: Different, Explicit Subset', '',
        'All twelve old source-crossfit fits / three seeds are retained. Their cached',
        'predictions cover 15,430 static-history windows, not the full source population.',
        'Future labels are used for scoring only; no excluded labels were opened.', '',
        '| Seed | Native equal-scene ADE gain (%) | Old normalized gain (%) |',
        '| --- | ---: | ---: |']
    for seed, v in r['existing_neural_subset']['all_seeds'].items():
        lines.append(f"| {seed} | {v['ADE_vs_CV']['equal_scene_gain_percent']} | {v['old_normalized_gain_percent']} |")
    lines += ['', 'Zero-CV easy rows report absolute pixel harm; percentage safety is undefined, not passed.',
        'No best seed or baseline is promoted from this readout. The four-scene bootstrap',
        'describes reused source sites and cannot establish independent generalization.',
        'The new score does not fix old negative probes, lineage concerns or calibration.',
        'A matched full-population neural experiment and native-scale training/risk registration',
        'are still required. Original validation/test, main/external and bookstore readouts stay closed.',
        'Stage5C/SMC disabled. The research goal and submission readiness remain unmet.', '',
        'Reproduce: `.venv-pytorch/bin/python scripts/rescore_m3w_native_metric.py --verify`.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    reg, report = compute()
    folder = ROOT/reg['report_directory']
    path, md = folder/'analysis.json', folder/'results.md'
    if args.verify:
        if json.loads(path.read_text()) != report or md.read_text() != render(report):
            raise ValueError('Completed readout differs from exact replay')
        print(json.dumps(dict(exact_replay=True, analysis_sha256=file_digest(path))), flush=True)
        return
    if path.exists() or md.exists():
        raise FileExistsError('Use --verify; do not overwrite completed evidence')
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    md.write_text(render(report))
    print(json.dumps(dict(completed=True, analysis_sha256=file_digest(path),
                          indexed_rows=report['indexed_rows'])), flush=True)


if __name__ == '__main__':
    main()
