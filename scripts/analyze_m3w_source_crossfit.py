"""Fixed training-side candidate utility; OOF is not independent confirmation."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_source_crossfit import load_config, load_data, CrossfitCorpus
from scripts.run_m3w_source_continuation import immutable_json
import numpy as np
import torch
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_recording_diagnostic import recording_resamples, paired_gain_interval
from src.world_model.m3w_source_crossfit import cost_labels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg = load_config(args.registration)
    public, private = ROOT/reg['reports'], ROOT/reg['output']
    report = json.loads((public/'report.json').read_text())
    assert report['models'] == 12 and report['optimizer_updates'] == 120000
    data = CrossfitCorpus(load_data(Path(reg['data_registration'])))
    ids = np.flatnonzero(data.source_sites != 'bookstore') + data.nmain
    loc = ids-data.nmain
    sites, records = data.source_sites[loc], data.source_records[loc]
    seed_errors, seed_fde, seed_gain = [], [], []
    for receipt in report['oof_labels']:
        path = ROOT/receipt['path']
        assert file_digest(path) == receipt['sha256']
        with np.load(path, allow_pickle=False) as a:
            np.testing.assert_array_equal(a['ids'], ids)
            labels = cost_labels(a['prediction'], data.target[loc], a['cost_scale'])
            for key, value in labels.items():
                np.testing.assert_array_equal(a[key], value)
            seed_errors.append(a['ade'].copy()); seed_fde.append(a['fde'].copy())
            seed_gain.append(a['signed_gain'].copy())
    errors = np.asarray(seed_errors)
    cv = np.linalg.norm(data.target[loc].astype(float), axis=-1).mean(1)
    mean_error = errors.mean(0)
    native = data.native_scale[loc]
    site_rows = []
    for site in reg['inner_sites']:
        mask = sites == site
        _, inv, counts = recording_resamples(records[mask], reg['bootstrap_resamples'], reg['bootstrap_seed'])
        ci = paired_gain_interval(cv[mask], mean_error[mask], cv[mask], inv, counts)
        fits = [t for t in report['trials'] if t['site'] == site]
        cutoff = fits[0]['identity']['training_hard_cut']
        hard = mask & (cv >= cutoff); easy = mask & (cv == 0)
        site_rows.append(dict(site=site, rows=int(mask.sum()), recordings=len(set(records[mask])),
            agents=len(set(data.source_tracks[loc][mask])),
            train_gain_percent_mean=float(np.mean([t['training']['gain_percent'] for t in fits])),
            oof_ade=float(mean_error[mask].mean()), cv_ade=float(cv[mask].mean()),
            oof_gain_percent=ci['point_percent'], conditional_recording_ci95=ci['conditional_recording_ci95'],
            seed_gain_percent=[float(100*(1-e[mask].sum()/cv[mask].sum())) for e in errors],
            hard_gain_percent=float(100*(1-mean_error[hard].sum()/cv[hard].sum())) if hard.any() else None,
            easy_pixel_harm=float((mean_error[easy]*native[easy]).mean()) if easy.any() else None,
            easy_percentage_degradation=None))
    s_cv = np.array([r['cv_ade'] for r in site_rows]); s_err = np.array([r['oof_ade'] for r in site_rows])
    sampled = np.random.default_rng(reg['bootstrap_seed']).integers(4, size=(reg['bootstrap_resamples'], 4))
    values = 100*(1-s_err[sampled].mean(1)/s_cv[sampled].mean(1))
    controls = json.loads((ROOT/reg['reference_training_report']).read_text())
    reference_errors = []
    reference_hashes = {}
    for seed in reg['seeds']:
        trial = next(t for t in controls['trials'] if t['seed'] == seed and t['schedule'] == 'cosine')
        final = trial['milestones'][-1]
        path = ROOT/final['prediction_path']
        assert file_digest(path) == final['prediction_sha256']
        reference_hashes[str(path.relative_to(ROOT))] = final['prediction_sha256']
        with np.load(path, allow_pickle=False) as a:
            np.testing.assert_array_equal(a['ids'], ids)
            reference_errors.append(np.linalg.norm(a['prediction'].astype(float)-data.target[loc], axis=-1).mean(1))
    reference = np.asarray(reference_errors)
    zero = cv == 0
    analysis = dict(result_source='fresh_run_training_only_crossfit_analysis',
        report_sha256=file_digest(public/'report.json'), rows=len(ids), sites=4,
        recordings=len(set(records)), scoped_agents=len(set(data.source_tracks[loc])),
        seed_aggregation='mean_per_query_errors_not_forecast_ensemble',
        primary_equal_site_gain_percent=float(100*(1-s_err.mean()/s_cv.mean())),
        conditional_four_site_ci95=np.quantile(values, [.025, .975]).tolist(),
        bootstrap_resamples=reg['bootstrap_resamples'],
        uncertainty_limit='four_explored_sites_shared_fold_training_not_independent_confirmation',
        pooled_window_gain_percent=float(100*(1-mean_error.sum()/cv.sum())),
        pooled_seed_gain_percent=[float(100*(1-e.sum()/cv.sum())) for e in errors],
        native_pixel_ade=float((mean_error*native).mean()),
        native_pixel_cv_ade=float((cv*native).mean()),
        easy_rows=int(zero.sum()), easy_pixel_harm=float((mean_error[zero]*native[zero]).mean()),
        easy_percentage_degradation=None, easy_reason='zero_error_CV_denominator',
        mean_per_seed_binary_oracle_gain_percent=float(np.mean([100*(1-np.minimum(e,cv).sum()/cv.sum()) for e in errors])),
        oracle_is_deployable=False, sites_detail=site_rows,
        descriptive_in_sample_reference_gain_percent=float(100*(1-reference.mean(0).sum()/cv.sum())),
        reference_hashes=reference_hashes,
        positive_reference_to_nonpositive_oof_fraction=float(((reference < cv) & (errors >= cv)).mean()),
        reference_oof_gain_sign_disagreement=float(((reference < cv) != (errors < cv)).mean()),
        comparison_confound='smaller_training_population_site_shift_and_normalizer_change_not_isolated_optimism_effect',
        risk_head_trained=False, outer_rows_scored=0, main_rows_scored=0,
        main_primary_changed=False, sealed_roles_opened=False, new_deployment=False,
        stage5c_executed=False, smc_enabled=False)
    immutable_json(public/'analysis.json', analysis)
    with (public/'site_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(site_rows[0]))
        writer.writeheader(); writer.writerows(site_rows)
    lines = ['# Training-Side Candidate Cross-Fitting', '', '## Material Passport', '',
        'Twelve cold-start models, three seeds, four inner source sites, 120,000 updates.',
        'Bookstore and all main roles excluded from every fit and inference.',
        'OOF targets are training supervision, not independent confirmation or a deployed risk head.', '',
        '| Inner held site | Rows | Mean training gain (%) | OOF gain (%) | Conditional recording 95% CI | Easy pixel harm |',
        '| --- | ---: | ---: | ---: | --- | ---: |']
    for row in site_rows:
        lines.append(f"| {row['site']} | {row['rows']} | {row['train_gain_percent_mean']:+.6f} | {row['oof_gain_percent']:+.6f} | {row['conditional_recording_ci95']} | {row['easy_pixel_harm']:.8f} |")
    lines += ['', f"Primary equal-site gain: {analysis['primary_equal_site_gain_percent']:+.6f}%.",
        f"Conditional four-site interval: {analysis['conditional_four_site_ci95']}.",
        f"Window-weighted sensitivity: {analysis['pooled_window_gain_percent']:+.6f}%.",
        f"Per-seed window gains: {analysis['pooled_seed_gain_percent']}.",
        f"Mean per-seed binary future-oracle gain: {analysis['mean_per_seed_binary_oracle_gain_percent']:.6f}% (not a policy).", '',
        'The cached full-four-site in-sample reference is descriptive only. Smaller inner',
        'training sets, site shift and normalization change prevent a causal attribution',
        'of the entire gap to training-error optimism. No favorable site/seed is selected.',
        'Bootstrap uncertainty is conditional on four explored sites and overlapping fits.',
        'Easy percentage is undefined, not a safety pass. Annotation pixels/past-normalized',
        'raw-frame task only; no metric/seconds/foundation claim. Stage5C/SMC remain off.', '']
    (public/'results.md').write_text('\n'.join(lines))
    os.environ['MPLCONFIGDIR'] = str(private/'plot_runtime')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with matplotlib.rc_context({'svg.hashsalt':'m3w-source-crossfit-v1'}):
        fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
        x = np.arange(4)
        ax.bar(x-.18, [r['train_gain_percent_mean'] for r in site_rows], .36, label='Training complement', color='#437a91')
        ax.bar(x+.18, [r['oof_gain_percent'] for r in site_rows], .36, label='Inner held site', color='#b95759')
        ax.axhline(0, color='#444444', linewidth=.8)
        ax.set_xticks(x, reg['inner_sites']); ax.set_ylabel('ADE gain over stationary CV (%)')
        ax.set_title('Candidate cost transfer inside training sites: three-seed means')
        ax.legend(); fig.savefig(public/'comparison.svg', metadata={'Date':None})
        fig.savefig(private/'plot_runtime'/'comparison.png', dpi=150); plt.close(fig)
    print(json.dumps({k:v for k,v in analysis.items() if k not in ('sites_detail','reference_hashes')}, indent=2))


if __name__ == '__main__':
    main()
