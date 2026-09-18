"""Fixed all-arm readout, with shared explored-site uncertainty and no selection."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_source_pretrained_temporal import load_config, context, ARMS
from scripts.run_m3w_source_continuation import immutable_json
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_recording_diagnostic import recording_resamples, paired_gain_interval
from src.world_model.m3w_source_crossfit import cost_labels
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    args = p.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg = load_config(args.registration); data = context(reg)
    public = ROOT/reg['reports']; report = json.loads((public/'report.json').read_text())
    assert report['models'] == 36 and report['optimizer_updates'] == 360000
    ids = np.flatnonzero(data.source_sites != 'bookstore')+data.nmain; loc = ids-data.nmain
    cv = np.linalg.norm(data.target[loc].astype(float), axis=-1).mean(1)
    native = data.native_scale[loc]; sites = data.source_sites[loc]; records = data.source_records[loc]
    seed_errors = {}; fdes = {}
    for arm in ARMS:
        costs = []; final_errors = []
        for seed in reg['seeds']:
            receipt = next(t for t in report['oof_labels'] if t['arm'] == arm and t['seed'] == seed)
            path = ROOT/receipt['path']; assert file_digest(path) == receipt['sha256']
            with np.load(path, allow_pickle=False) as a:
                np.testing.assert_array_equal(ids, a['ids'])
                labels = cost_labels(a['prediction'], data.target[loc], a['cost_scale'])
                for k, value in labels.items(): np.testing.assert_array_equal(value, a[k])
                costs.append(a['ade'].copy()); final_errors.append(a['fde'].copy())
        seed_errors[arm] = np.asarray(costs); fdes[arm] = np.asarray(final_errors)
    draws = np.random.default_rng(reg['bootstrap_seed']).integers(4, size=(reg['bootstrap_resamples'], 4))
    s_cv = np.array([cv[sites == s].mean() for s in reg['sites']])
    summary = {}; per_site_means = {}
    for arm in ARMS:
        errors = seed_errors[arm]; mean = errors.mean(0)
        s_error = np.array([mean[sites == s].mean() for s in reg['sites']]); per_site_means[arm] = s_error
        gains = 100*(1-s_error[draws].mean(1)/s_cv[draws].mean(1))
        details = []
        for site in reg['sites']:
            m = sites == site; fits = [t for t in report['trials'] if t['arm'] == arm and t['site'] == site]
            cutoff = fits[0]['identity']['training_hard_cut']; hard = m & (cv >= cutoff)
            _, inv, counts = recording_resamples(records[m], reg['bootstrap_resamples'], reg['bootstrap_seed'])
            interval = paired_gain_interval(cv[m], mean[m], cv[m], inv, counts)
            details.append(dict(site=site, rows=int(m.sum()), records=len(set(records[m])),
                training_gain_percent=float(np.mean([t['training']['gain_percent'] for t in fits])),
                held_gain_percent=interval['point_percent'], conditional_recording_ci95=interval['conditional_recording_ci95'],
                seed_gains=[float(100*(1-e[m].sum()/cv[m].sum())) for e in errors],
                hard_gain_percent=float(100*(1-mean[hard].sum()/cv[hard].sum())) if hard.any() else None,
                easy_pixel_harm=float((mean[m & (cv == 0)]*native[m & (cv == 0)]).mean())))
        summary[arm] = dict(equal_site_gain_percent=float(100*(1-s_error.mean()/s_cv.mean())),
            conditional_four_site_ci95=np.quantile(gains, [.025, .975]).tolist(),
            window_gain_percent=float(100*(1-mean.sum()/cv.sum())),
            native_pixel_ade=float((mean*native).mean()), native_pixel_fde=float((fdes[arm].mean(0)*native).mean()),
            easy_pixel_harm=float((mean[cv == 0]*native[cv == 0]).mean()), easy_percentage_degradation=None,
            nonzero_gain_percent=float(100*(1-mean[cv > 0].sum()/cv[cv > 0].sum())),
            binary_oracle_equal_site_gain_percent=float(100*(1-np.mean([
                np.minimum(errors[:, sites == s], cv[sites == s]).mean() for s in reg['sites']])/s_cv.mean())),
            actual_nonzero_output_rate=float(np.mean([t['held']['actual_changed_rate'] for t in report['trials'] if t['arm'] == arm])),
            tail_ade95=float(np.quantile(mean*native, .95)), tail_ade99=float(np.quantile(mean*native, .99)),
            sites=details, positive_held_models=sum(t['held']['gain_percent'] > 0 for t in report['trials'] if t['arm'] == arm))
    contrasts = []
    for candidate, reference in [('sequence', 'geometry'), ('sequence', 'current'), ('current', 'geometry')]:
        difference = per_site_means[reference]-per_site_means[candidate]
        values = 100*difference[draws].mean(1)/s_cv[draws].mean(1)
        contrasts.append(dict(candidate=candidate, reference=reference,
            gain_difference_pp=float(100*difference.mean()/s_cv.mean()),
            conditional_four_site_ci95=np.quantile(values, [.025, .975]).tolist()))
    result = dict(result_source='fresh_run_fixed_all_arm_analysis', models=36, updates=360000,
        rows=len(ids), sites=4, recordings=len(set(records)), seeds=reg['seeds'], summary=summary, contrasts=contrasts,
        bootstrap_resamples=reg['bootstrap_resamples'], seed_aggregation='mean_errors_not_ensemble_forecast',
        uncertainty='four_explored_sites_shared_training_populations_not_confirmation',
        main_rows_scored=0, outer_rows_scored=0, new_policy=False, new_deployment=False,
        stage5c_executed=False, smc_enabled=False)
    immutable_json(public/'analysis.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
