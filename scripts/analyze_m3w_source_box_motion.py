"""All fixed source-motion endpoints and paired site-conditional uncertainty."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_m3w_source_box_motion import registration
from scripts.run_m3w_source_importance_sampling import load_config, context
from scripts.run_m3w_source_crossfit import immutable_json
from src.world_model.m3w_source_crossfit import cost_labels
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--registration', type=Path, required=True)
    args = p.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg = registration(args.registration); base = load_config(Path(reg['source_registration']))
    _, data, _, _, _, ids, _, _ = context(base)
    current = json.loads((ROOT/reg['reports']/'report.json').read_text())
    control = json.loads((ROOT/reg['control_report']).read_text())
    target = data.target[ids-data.nmain].astype(float)
    sites, native = data.source_sites[ids-data.nmain], data.native_scale[ids-data.nmain]
    cv = np.linalg.norm(target, axis=-1).mean(1)
    cv_sites = np.array([cv[sites == s].mean() for s in base['sites']])
    draws = np.random.default_rng(base['bootstrap_seed']).integers(4, size=(base['bootstrap_resamples'], 4))
    summary, errors = {}, {}
    for source, report, arms in [('fresh_run', current, reg['arms']), ('cached_verified', control, base['arms'])]:
        for arm in arms:
            ade, fde = [], []
            for seed in base['seeds']:
                item = next(x for x in report['oof_labels'] if x['arm'] == arm and x['seed'] == seed)
                assert file_digest(ROOT/item['path']) == item['sha256']
                with np.load(ROOT/item['path'], allow_pickle=False) as a:
                    np.testing.assert_array_equal(ids, a['ids'])
                    for name, value in cost_labels(a['prediction'], data.target[ids-data.nmain], a['cost_scale']).items():
                        np.testing.assert_array_equal(a[name], value)
                    ade.append(a['ade'].copy()); fde.append(a['fde'].copy())
            ade = np.array(ade); mean = ade.mean(0)
            per_site = np.array([mean[sites == s].mean() for s in base['sites']]); errors[arm] = per_site
            interval = 100*(1-per_site[draws].mean(1)/cv_sites[draws].mean(1))
            detail = []
            for site in base['sites']:
                trials = [t for t in report['trials'] if t['site'] == site and t['arm'] == arm]
                mask = sites == site; hard = mask & (cv >= trials[0]['identity']['training_hard_cut'])
                detail.append(dict(site=site, rows=int(mask.sum()), gain_percent=float(100*(1-mean[mask].sum()/cv[mask].sum())),
                    hard_gain_percent=float(100*(1-mean[hard].sum()/cv[hard].sum())),
                    seed_gains=[float(100*(1-v[mask].sum()/cv[mask].sum())) for v in ade],
                    training_gains=[t['training']['gain_percent'] for t in trials]))
            summary[arm] = dict(result_source=source, equal_site_gain_percent=float(100*(1-per_site.mean()/cv_sites.mean())),
                conditional_four_site_ci95=np.quantile(interval, [.025, .975]).tolist(),
                window_gain_percent=float(100*(1-mean.sum()/cv.sum())), native_pixel_ade=float((mean*native).mean()),
                native_pixel_fde=float((np.mean(fde, axis=0)*native).mean()),
                static_pixel_harm=float((mean[cv == 0]*native[cv == 0]).mean()), easy_percentage_degradation=None,
                nonzero_gain_percent=float(100*(1-mean[cv > 0].sum()/cv[cv > 0].sum())),
                binary_oracle_equal_site_gain_percent=float(100*(1-np.mean([
                    np.minimum(ade[:, sites == s], cv[sites == s]).mean() for s in base['sites']])/cv_sites.mean())),
                positive_held_models=sum(t['held']['gain_percent'] > 0 for t in report['trials'] if t['arm'] == arm), sites=detail)
    contrasts = []
    for candidate, reference in [('motion', 'quality'), ('quality', 'geometry'), ('motion', 'geometry'), ('motion', 'centered')]:
        difference = errors[reference]-errors[candidate]
        values = 100*difference[draws].mean(1)/cv_sites[draws].mean(1)
        contrasts.append(dict(candidate=candidate, reference=reference, gain_difference_pp=float(100*difference.mean()/cv_sites.mean()),
                             conditional_four_site_ci95=np.quantile(values, [.025, .975]).tolist()))
    previous = json.loads((ROOT/reg['control_report']).with_name('analysis.json').read_text())
    for arm in base['arms']:
        np.testing.assert_allclose(summary[arm]['equal_site_gain_percent'],
            previous['summary'][arm+'_conditioned']['equal_site_gain_percent'], atol=1e-10, rtol=0)
    result = dict(result_source='fresh_run_motion_quality_heads_cached_verified_conditioned_controls', rows=len(ids),
        new_models=24, new_updates=240000, cached_reference_models=24, summary=summary, contrasts=contrasts,
        diagnostics=[dict(trial=t['trial'], training_gain=t['training']['gain_percent'], held_gain=t['held']['gain_percent'],
            clipping_fraction=float(np.mean([x['gradient_norm'] > 5 for x in t['fit']['trace']]))) for t in current['trials']],
        bootstrap_resamples=base['bootstrap_resamples'], seed_aggregation='mean_errors_not_prediction_ensemble',
        uncertainty='conditional_four_explored_sites_shared_training_not_confirmation',
        new_deployment=False, main_outer_rows_scored=0, stage5c_executed=False, smc_enabled=False)
    immutable_json(ROOT/reg['reports']/'analysis.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
