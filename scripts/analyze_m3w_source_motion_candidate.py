"""Registered matched loss diagnostic; future oracle is not a deployment policy."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_source_motion_candidate import load_config, context
from scripts.run_m3w_source_continuation import immutable_json
from src.world_model.m3w_source_crossfit import cost_labels
from src.evaluation.m3w_crossfit_cost_diagnostic import decompose_costs
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_recording_diagnostic import recording_resamples, paired_gain_interval
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg = load_config(args.registration)
    public, private = ROOT/reg['reports'], ROOT/reg['output']
    _, control, data = context(reg)
    new = json.loads((public/'report.json').read_text())
    assert new['models'] == 12 and new['optimizer_updates'] == 120000
    ids = np.flatnonzero(data.source_sites != 'bookstore') + data.nmain
    loc = ids-data.nmain; sites = data.source_sites[loc]; records = data.source_records[loc]
    cv = np.linalg.norm(data.target[loc].astype(float), axis=-1).mean(1)
    costs, sources = {}, {}
    for name, report in [('unconditional_control',control), ('motion_loss',new)]:
        errors = []
        sources[name] = []
        for receipt in report['oof_labels']:
            path = ROOT/receipt['path']; assert file_digest(path) == receipt['sha256']
            with np.load(path, allow_pickle=False) as a:
                np.testing.assert_array_equal(a['ids'], ids)
                labels = cost_labels(a['prediction'], data.target[loc], a['cost_scale'])
                for key, value in labels.items(): np.testing.assert_array_equal(a[key], value)
                errors.append(labels['ade'])
            sources[name].append(receipt)
        costs[name] = np.asarray(errors)
    site_names = reg['sites']
    baseline = np.array([cv[sites == s].mean() for s in site_names])
    draws = np.random.default_rng(reg['bootstrap_seed']).integers(len(site_names),
        size=(reg['bootstrap_resamples'], len(site_names)))
    def interval(reference, candidate):
        point = float(100*(reference.mean()-candidate.mean())/baseline.mean())
        sampled = 100*(reference[draws].mean(1)-candidate[draws].mean(1))/baseline[draws].mean(1)
        return dict(point_pp=point, conditional_site_ci95=np.quantile(sampled, [.025,.975]).tolist())
    site_costs, site_oracles, summary, details = {}, {}, {}, []
    for name, errors in costs.items():
        mean = errors.mean(0); oracle = np.minimum(errors, cv[None,:]).mean(0)
        site_costs[name] = np.array([mean[sites == s].mean() for s in site_names])
        site_oracles[name] = np.array([oracle[sites == s].mean() for s in site_names])
        summary[name] = dict(decompose_costs(cv, errors, sites, data.native_scale[loc]),
            all_gain=interval(baseline,site_costs[name]),
            oracle_gain=interval(baseline,site_oracles[name]),
            pooled_seed_gain=[float(100*(1-e.sum()/cv.sum())) for e in errors],
            native_pixel_ade=float((mean*data.native_scale[loc]).mean()),
            primary_is_original_equal_site_ADE=True)
        for site in site_names:
            mask = sites == site
            cutoff = next(t['identity']['training_hard_cut'] for t in new['trials'] if t['site'] == site)
            hard = mask & (cv >= cutoff); easy = mask & (cv == 0); moving = mask & (cv > 0)
            _, inv, counts = recording_resamples(records[mask], reg['bootstrap_resamples'], reg['bootstrap_seed'])
            ci = paired_gain_interval(cv[mask], mean[mask], cv[mask], inv, counts)
            details.append(dict(arm=name, site=site, rows=int(mask.sum()),
                all_gain=ci['point_percent'], conditional_recording_ci95=ci['conditional_recording_ci95'],
                nonzero_target_gain=float(100*(1-mean[moving].sum()/cv[moving].sum())),
                oracle_gain=float(100*(1-oracle[mask].sum()/cv[mask].sum())),
                hard_gain=float(100*(1-mean[hard].sum()/cv[hard].sum())),
                easy_pixel_harm=float((mean[easy]*data.native_scale[loc][easy]).mean()),
                per_seed_gain=[float(100*(1-e[mask].sum()/cv[mask].sum())) for e in errors]))
    analysis = dict(result_source='fresh_run_fixed_training_only_analysis_with_cached_verified_control',
        report_sha256=file_digest(public/'report.json'), sources=sources, models_new=12, models_control=12,
        new_updates=120000, rows=len(ids), sites=4, recordings=len(set(records)),
        scoped_agents=len(set(data.source_tracks[loc])), summary=summary, site_details=details,
        motion_minus_control_all=interval(site_costs['unconditional_control'],site_costs['motion_loss']),
        motion_minus_control_oracle=interval(site_oracles['unconditional_control'],site_oracles['motion_loss']),
        bootstrap_resamples=reg['bootstrap_resamples'],
        uncertainty_scope='four_explored_sites_shared_fold_fits_conditional_not_confirmation',
        intervention='only_zero_target_ADE_gradients_removed_fixed_full_batch_denominator',
        oracle_is_deployable=False, new_policy_trained=False, new_deployment=False,
        all_eval_rows_retained=True, outer_rows_scored=0, main_rows_scored=0,
        main_primary_changed=False, stage5c_executed=False, smc_enabled=False)
    immutable_json(public/'analysis.json', analysis)
    with (public/'site_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0])); writer.writeheader(); writer.writerows(details)
    lines = ['# Matched Zero-Target Gradient Intervention', '', '## Material Passport', '',
        'Twelve new cold-start fits, 120,000 updates, three seeds; twelve controls cached_verified.',
        'Same source folds, inputs, preprocessing, scale, initial seeds, sampler and terminal budget.',
        'All held queries retained. No future label is an inference input. No learned deployment gate.', '',
        '| Arm | Equal-site all gain | Conditional site 95% CI | Oracle gain (diagnostic) | Easy pixel harm |',
        '| --- | ---: | --- | ---: | ---: |']
    for name, row in summary.items():
        lines.append(f"| {name} | {row['all_gain']['point_pp']:+.6f}% | {row['all_gain']['conditional_site_ci95']} | {row['oracle_gain']['point_pp']:.6f}% | {row['subsets']['zero_target']['native_pixel_mean_excess']:.6f} |")
    lines += ['', 'Matched contrasts:', '', json.dumps({k:analysis[k] for k in ['motion_minus_control_all','motion_minus_control_oracle']}, indent=2), '',
        'Oracle values use future labels and cannot be deployed. Zero-target easy percentage is undefined.',
        'Shared training folds and explored sites prevent independent confirmation. This loss weighting',
        'control is not a novel architecture or evidence of useful causal selection by itself.',
        'Pixels/past-normalized raw frames only. No metric/seconds/true-3D/foundation claim. Stage5C/SMC off.', '']
    (public/'results.md').write_text('\n'.join(lines))
    os.environ['MPLCONFIGDIR'] = str(private/'plot_runtime')
    os.environ['XDG_CACHE_HOME'] = str(private/'plot_runtime'/'fontcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with matplotlib.rc_context({'svg.hashsalt':'m3w-motion-loss-v1'}):
        fig, axes = plt.subplots(1,2, figsize=(10,4), layout='constrained')
        for i, name in enumerate(costs):
            rows = [r for r in details if r['arm'] == name]
            for ax, field in zip(axes, ['all_gain','oracle_gain']):
                ax.bar(np.arange(4)+(i-.5)*.36, [r[field] for r in rows], .36, label=name,
                       color=['#437a91','#b95759'][i])
                ax.set_xticks(np.arange(4),site_names,rotation=20); ax.axhline(0,color='#444',linewidth=.7)
                ax.set_ylabel('ADE gain over stationary CV (%)')
        axes[0].set_title('Actual ungated candidate'); axes[1].set_title('Future oracle: not deployable')
        axes[0].legend(fontsize=8)
        fig.savefig(public/'comparison.svg', metadata={'Date':None})
        fig.savefig(private/'plot_runtime'/'comparison.png',dpi=150); plt.close(fig)
    print(json.dumps({k:v for k,v in analysis.items() if k not in ('sources','site_details','summary')}, indent=2))


if __name__ == '__main__':
    main()
