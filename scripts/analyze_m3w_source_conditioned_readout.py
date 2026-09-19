"""Report all registered readout endpoints; no seed or threshold selection."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_source_conditioned_readout import registration
from scripts.run_m3w_source_importance_sampling import load_config, context
from scripts.run_m3w_source_crossfit import immutable_json
from src.world_model.m3w_source_crossfit import cost_labels
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    args = p.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg = registration(args.registration); base = load_config(Path(reg['source_registration']))
    _, data, _, _, _, ids, _, _ = context(base)
    control = json.loads((ROOT/reg['control_report']).read_text())
    current = json.loads((ROOT/reg['reports']/'report.json').read_text())
    target = data.target[ids-data.nmain].astype(float)
    sites, native = data.source_sites[ids-data.nmain], data.native_scale[ids-data.nmain]
    cv = np.linalg.norm(target, axis=-1).mean(1)
    cv_sites = np.array([cv[sites == s].mean() for s in base['sites']])
    draws = np.random.default_rng(base['bootstrap_seed']).integers(4, size=(base['bootstrap_resamples'], 4))
    summaries, site_errors = {}, {}
    for label, report in [('control', control), ('conditioned', current)]:
        for arm in base['arms']:
            ade, fde = [], []
            for seed in base['seeds']:
                item = next(x for x in report['oof_labels'] if x['arm'] == arm and x['seed'] == seed)
                path = ROOT/item['path']; assert file_digest(path) == item['sha256']
                with np.load(path, allow_pickle=False) as a:
                    np.testing.assert_array_equal(ids, a['ids'])
                    for name, values in cost_labels(a['prediction'], data.target[ids-data.nmain], a['cost_scale']).items():
                        np.testing.assert_array_equal(a[name], values)
                    ade.append(a['ade'].copy()); fde.append(a['fde'].copy())
            ade = np.asarray(ade); mean = ade.mean(0)
            site_mean = np.array([mean[sites == s].mean() for s in base['sites']])
            key = arm+'_'+label; site_errors[key] = site_mean
            interval = 100*(1-site_mean[draws].mean(1)/cv_sites[draws].mean(1))
            details = []
            for site in base['sites']:
                trials = [t for t in report['trials'] if t['site'] == site and t['arm'] == arm]
                mask = sites == site; hard = mask & (cv >= trials[0]['identity']['training_hard_cut'])
                details.append(dict(site=site, rows=int(mask.sum()),
                    gain_percent=float(100*(1-mean[mask].sum()/cv[mask].sum())),
                    hard_gain_percent=float(100*(1-mean[hard].sum()/cv[hard].sum())),
                    seed_gains=[float(100*(1-e[mask].sum()/cv[mask].sum())) for e in ade],
                    training_gains=[t['training']['gain_percent'] for t in trials]))
            summaries[key] = dict(result_source='fresh_run' if label == 'conditioned' else 'cached_verified_recomputed',
                equal_site_gain_percent=float(100*(1-site_mean.mean()/cv_sites.mean())),
                conditional_four_site_ci95=np.quantile(interval, [.025, .975]).tolist(),
                window_gain_percent=float(100*(1-mean.sum()/cv.sum())),
                native_pixel_ade=float((mean*native).mean()),
                native_pixel_fde=float((np.mean(fde, axis=0)*native).mean()),
                static_pixel_harm=float((mean[cv == 0]*native[cv == 0]).mean()),
                easy_percentage_degradation=None,
                nonzero_gain_percent=float(100*(1-mean[cv > 0].sum()/cv[cv > 0].sum())),
                binary_oracle_equal_site_gain_percent=float(100*(1-np.mean([
                    np.minimum(ade[:, sites == s], cv[sites == s]).mean() for s in base['sites']])/cv_sites.mean())),
                positive_held_models=sum(t['held']['gain_percent'] > 0 for t in report['trials'] if t['arm'] == arm),
                sites=details)
    contrasts = []
    for candidate, reference in [('geometry_conditioned', 'geometry_control'),
                                 ('centered_conditioned', 'centered_control'),
                                 ('centered_conditioned', 'geometry_conditioned')]:
        diff = site_errors[reference]-site_errors[candidate]
        values = 100*diff[draws].mean(1)/cv_sites[draws].mean(1)
        contrasts.append(dict(candidate=candidate, reference=reference,
            gain_difference_pp=float(100*diff.mean()/cv_sites.mean()),
            conditional_four_site_ci95=np.quantile(values, [.025, .975]).tolist()))
    previous = json.loads((ROOT/reg['control_report']).with_name('analysis.json').read_text())
    for arm in base['arms']:
        np.testing.assert_allclose(summaries[arm+'_control']['equal_site_gain_percent'],
            previous['summary'][arm+'_corrected']['equal_site_gain_percent'], atol=1e-10, rtol=0)
    diagnostics = [dict(trial=t['trial'], readout_gain=t['identity']['readout_gain'],
        training_gain_percent=t['training']['gain_percent'], held_gain_percent=t['held']['gain_percent'],
        logged_gradient_clipped_fraction=float(np.mean([x['gradient_norm'] > 5 for x in t['fit']['trace']])),
        logged_gradient_norm_range=[min(x['gradient_norm'] for x in t['fit']['trace']),
                                    max(x['gradient_norm'] for x in t['fit']['trace'])]) for t in current['trials']]
    result = dict(result_source='fresh_run_conditioned_heads_cached_verified_matched_controls',
        new_models=24, new_updates=240000, cached_control_models=24, rows=len(ids),
        bootstrap_resamples=base['bootstrap_resamples'], seed_aggregation='mean_errors_not_prediction_ensemble',
        uncertainty='conditional_four_explored_sites_shared_training_not_confirmation',
        summary=summaries, contrasts=contrasts, diagnostics=diagnostics,
        new_deployment=False, main_outer_rows_scored=0, stage5c_executed=False, smc_enabled=False)
    immutable_json(ROOT/reg['reports']/'analysis.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
