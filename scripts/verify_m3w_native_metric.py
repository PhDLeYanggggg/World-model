"""Independent reducer for the fixed native-coordinate amendment readout."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.evaluation.m3w_experiment_contract import file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    directory = ROOT/'outputs/publication_readiness_2026_09/native_metric_v1'
    result = json.loads((directory/'analysis.json').read_text())
    for path, sha in result['evidence_hashes'].items():
        if file_digest(ROOT/path) != sha:
            raise ValueError('Changed readout dependency: '+path)
    old = json.loads((ROOT/'outputs/publication_readiness_2026_09/source_population_v1/audit.json').read_text())
    with np.load(ROOT/old['row_archive']['path'], allow_pickle=False) as z:
        a = {k:z[k] for k in z.files}
    roster = result['source_sites']
    checks = []

    def verify_table(table, model, reference, scenes):
        gains = []
        for scene in roster:
            use = (scenes == scene) & np.isfinite(reference)
            rec = table['by_scene'][scene]
            assert rec['rows'] == int(use.sum())
            if not use.any():
                assert rec['gain_percent'] is None
                continue
            mean_m = sum(float(v) for v in model[use])/int(use.sum())
            mean_r = sum(float(v) for v in reference[use])/int(use.sum())
            np.testing.assert_allclose([rec['model_error'], rec['reference_error'], rec['absolute_harm']],
                                       [mean_m, mean_r, mean_m-mean_r], rtol=1e-11, atol=1e-10)
            if mean_r == 0:
                assert rec['gain_percent'] is None
            else:
                gain = 100*(mean_r-mean_m)/mean_r
                np.testing.assert_allclose(rec['gain_percent'], gain, rtol=1e-10, atol=1e-9)
                gains.append(gain)
        if len(gains) == len(roster):
            np.testing.assert_allclose(table['equal_scene_gain_percent'], sum(gains)/len(gains), atol=1e-9)
            draws = np.random.default_rng(38113).integers(0, len(roster), size=(3000, len(roster)))
            boot = np.array([sum(gains[i] for i in row)/len(roster) for row in draws])
            np.testing.assert_allclose(table['scene_bootstrap_ci95'], np.quantile(boot, [.025, .975]), atol=1e-9)
        else:
            assert table['equal_scene_gain_percent'] is table['scene_bootstrap_ci95'] is None
        checks.append(len(roster))

    names = list(old['cohorts']['complete']['baseline_metrics'])
    for label, cohort in result['cohorts'].items():
        take = np.ones(len(a['scale']), bool) if label == 'supported_masked' else a['valid_count'] == 12
        sites, scale = a['sites'][take], a['scale'][take]
        e = np.einsum('nk,n->nk', a['ade'][take], scale)
        f = np.einsum('nk,n->nk', a['fde'][take], scale)
        selected = np.zeros(len(sites), int)
        for site in roster:
            training_scores = []
            for other in roster:
                if site == other:
                    continue
                rows = (sites == other) & np.isfinite(e[:, 1])
                mean = np.sum(e[rows], axis=0)/rows.sum()
                training_scores.append(mean/mean[1])
            k = int(np.sum(training_scores, axis=0).argmin())
            selected[sites == site] = k
            rec = cohort['methods']['complement_selected_baseline']['selection'][site]
            assert rec['selected_index'] == k and site not in rec['selection_scenes']
        oracle = np.zeros(len(e), int)
        valid = np.isfinite(e[:, 1])
        oracle[valid] = np.argmin(e[valid], axis=1)
        ref = e[np.arange(len(e)), selected]
        for name, metrics in cohort['methods'].items():
            choice = (oracle if name == 'oracle_diagnostic_not_deployable' else
                      selected if name == 'complement_selected_baseline' else
                      np.full(len(e), names.index(name)))
            verify_table(metrics['ADE_vs_CV'], e[np.arange(len(e)), choice], e[:, 1], sites)
            verify_table(metrics['FDE_vs_CV'], f[np.arange(len(f)), choice], f[:, 1], sites)
            if 'ADE_vs_complement_selected' in metrics:
                verify_table(metrics['ADE_vs_complement_selected'], e[np.arange(len(e)), choice], ref, sites)
    static = a['static_history'] & (a['valid_count'] == 12)
    sites, scale = a['sites'][static], a['scale'][static]
    parent = json.loads((ROOT/'outputs/publication_readiness_2026_09/source_crossfit_v1/report.json').read_text())
    costs = []
    for rec in parent['oof_labels']:
        with np.load(ROOT/rec['path'], allow_pickle=False) as z:
            ade, fde = z['ade']*scale, z['fde']*scale
        report = result['existing_neural_subset']['all_seeds'][str(rec['seed'])]
        verify_table(report['ADE_vs_CV'], ade, a['ade'][static, 1]*scale, sites)
        verify_table(report['FDE_vs_CV'], fde, a['fde'][static, 1]*scale, sites)
        easy = a['event'][static] == 0
        verify_table(report['zero_CV_easy_ADE'], ade[easy], a['ade'][static, 1][easy]*scale[easy], sites[easy])
        costs.append(ade)
    verify_table(result['existing_neural_subset']['mean_per_seed_errors_ADE'], np.mean(costs, axis=0),
                 a['ade'][static, 1]*scale, sites)
    report = dict(result_source='fresh_run_independent_scalar_reduction_cached_verified_inputs',
        analysis_sha256=file_digest(directory/'analysis.json'),
        verifier_sha256=file_digest(Path(__file__)), dependency_hashes_checked=len(result['evidence_hashes']),
        metric_tables_checked=len(checks), scene_reductions_checked=sum(checks),
        bootstrap_resamples_per_defined_table=3000, all_checks_passed=True,
        new_training=False, new_inference=False, independent_confirmation=False)
    target = directory/'verification.json'
    if args.verify:
        if json.loads(target.read_text()) != report:
            raise ValueError('Verification replay mismatch')
    else:
        if target.exists():
            raise FileExistsError('Use --verify for completed verification')
        target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
