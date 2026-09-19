"""Evaluate every fixed importance-corrected head and its matched controls."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.run_m3w_source_importance_sampling import load_config, context, immutable_json
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_recording_diagnostic import recording_resamples, paired_gain_interval
from src.world_model.m3w_source_crossfit import cost_labels
from src.world_model.m3w_source_episode_sampler import episode_weights
from src.world_model.m3w_source_importance_sampling import uniform_risk_factors
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--registration',type=Path,required=True)
    args = p.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg = load_config(args.registration)
    _,data,cached,uniform,event,ids,groups,_ = context(reg)
    public = ROOT/reg['reports']; report = json.loads((public/'report.json').read_text())
    centered = json.loads((ROOT/reg['centered_control_report']).read_text())
    loc = ids-data.nmain; sites,records = data.source_sites[loc],data.source_records[loc]
    cv = np.linalg.norm(data.target[loc].astype(float),axis=-1).mean(1); native = data.native_scale[loc]
    cv_site = np.array([cv[sites==s].mean() for s in reg['sites']])
    draws = np.random.default_rng(reg['bootstrap_seed']).integers(4,size=(reg['bootstrap_resamples'],4))
    means,summary = {},{}
    arms = [('geometry_uniform','geometry',uniform),('centered_uniform','centered',centered),
            ('geometry_event','geometry',event),('centered_event','centered',event),
            ('geometry_corrected','geometry',report),('centered_corrected','centered',report)]
    for arm,model_arm,source in arms:
        errors,fdes = [],[]
        for seed in reg['seeds']:
            item = next(v for v in source['oof_labels'] if v['arm']==model_arm and v['seed']==seed)
            path = ROOT/item['path']; assert file_digest(path)==item['sha256']
            with np.load(path,allow_pickle=False) as a:
                np.testing.assert_array_equal(ids,a['ids'])
                for k,v in cost_labels(a['prediction'],data.target[loc],a['cost_scale']).items():
                    np.testing.assert_array_equal(v,a[k])
                errors.append(a['ade'].copy()); fdes.append(a['fde'].copy())
        errors = np.asarray(errors); mean = errors.mean(0)
        sm = np.array([mean[sites==s].mean() for s in reg['sites']]); means[arm] = sm
        values = 100*(1-sm[draws].mean(1)/cv_site[draws].mean(1)); details = []
        for site in reg['sites']:
            m = sites==site; fits = [t for t in source['trials'] if t['arm']==model_arm and t['site']==site]
            hard = m&(cv>=fits[0]['identity']['training_hard_cut'])
            _,inv,counts = recording_resamples(records[m],reg['bootstrap_resamples'],reg['bootstrap_seed'])
            interval = paired_gain_interval(cv[m],mean[m],cv[m],inv,counts)
            details.append(dict(site=site,rows=int(m.sum()),records=len(set(records[m])),
                training_gain_percent=float(np.mean([t['training']['gain_percent'] for t in fits])),
                held_gain_percent=interval['point_percent'],conditional_recording_ci95=interval['conditional_recording_ci95'],
                seed_gains=[float(100*(1-e[m].sum()/cv[m].sum())) for e in errors],
                hard_gain_percent=float(100*(1-mean[hard].sum()/cv[hard].sum())),
                easy_pixel_harm=float((mean[m&(cv==0)]*native[m&(cv==0)]).mean())))
        summary[arm] = dict(result_source='fresh_run' if source is report else 'cached_verified_recomputed',
            equal_site_gain_percent=float(100*(1-sm.mean()/cv_site.mean())),
            conditional_four_site_ci95=np.quantile(values,[.025,.975]).tolist(),
            window_gain_percent=float(100*(1-mean.sum()/cv.sum())),
            native_pixel_ade=float((mean*native).mean()),native_pixel_fde=float((np.mean(fdes,axis=0)*native).mean()),
            easy_pixel_harm=float((mean[cv==0]*native[cv==0]).mean()),easy_percentage_degradation=None,
            nonzero_gain_percent=float(100*(1-mean[cv>0].sum()/cv[cv>0].sum())),
            binary_oracle_equal_site_gain_percent=float(100*(1-np.mean([
                np.minimum(errors[:,sites==s],cv[sites==s]).mean() for s in reg['sites']])/cv_site.mean())),
            tail_ade95=float(np.quantile(mean*native,.95)),tail_ade99=float(np.quantile(mean*native,.99)),
            sites=details,positive_held_models=sum(t['held']['gain_percent']>0 for t in source['trials'] if t['arm']==model_arm))
    previous = json.loads((ROOT/reg['episode_control_report']).with_name('analysis.json').read_text())
    for arm in previous['summary']:
        for k in ('equal_site_gain_percent','window_gain_percent','easy_pixel_harm'):
            np.testing.assert_allclose(summary[arm][k],previous['summary'][arm][k],atol=1e-10,rtol=0)
    contrasts = []
    for a,b in [('geometry_corrected','geometry_uniform'),('centered_corrected','centered_uniform'),
                ('geometry_corrected','geometry_event'),('centered_corrected','centered_event'),
                ('centered_corrected','geometry_corrected')]:
        diff = means[b]-means[a]; values = 100*diff[draws].mean(1)/cv_site[draws].mean(1)
        contrasts.append(dict(candidate=a,reference=b,gain_difference_pp=float(100*diff.mean()/cv_site.mean()),
            conditional_four_site_ci95=np.quantile(values,[.025,.975]).tolist()))
    diagnosis = []
    for trial in report['trials']:
        train,_,_,_,scale = data.configure(trial['site'])
        p,_ = episode_weights(train,ids,groups); w = uniform_risk_factors(p)
        target = data.target[train-data.nmain]
        path = ROOT/trial['prediction_path']; assert file_digest(path)==trial['prediction_sha256']
        with np.load(path,allow_pickle=False) as a:
            np.testing.assert_array_equal(a['train_ids'],train)
            error = np.linalg.norm(a['train_prediction'].astype(float)-target,axis=-1).mean(1)
        floor = np.linalg.norm(target.astype(float),axis=-1).mean(1)
        corrected = float(p@((error/scale)*w)/p.sum()); uniform_loss = float(error.mean()/scale)
        np.testing.assert_allclose(corrected,uniform_loss,atol=1e-12,rtol=1e-12)
        diagnosis.append(dict(trial=trial['trial'],site=trial['site'],arm=trial['arm'],seed=trial['seed'],
            uniform_training_gain_percent=float(100*(1-error.mean()/floor.mean())),
            uncorrected_episode_training_gain_percent=float(100*(1-p@error/(p@floor))),
            corrected_expected_loss=corrected,uniform_expected_loss=uniform_loss,
            logged_gradient_clipped_fraction=float(np.mean([v['gradient_norm']>5 for v in trial['fit']['trace']]))))
    result = dict(result_source='fresh_run_importance_corrected_heads_cached_verified_controls',new_models=24,
        new_updates=240000,cached_control_models=48,rows=len(ids),sites=4,recordings=29,
        bootstrap_resamples=reg['bootstrap_resamples'],uncertainty='four_explored_sites_shared_training_conditional_not_confirmation',
        seed_aggregation='mean_errors_not_prediction_ensemble',summary=summary,contrasts=contrasts,diagnosis=diagnosis,
        main_outer_rows_scored=0,new_deployment=False,stage5c_executed=False,smc_enabled=False)
    immutable_json(public/'analysis.json',result)
    print(json.dumps(dict(summary={k:v['equal_site_gain_percent'] for k,v in summary.items()},contrasts=contrasts),indent=2))


if __name__ == '__main__': main()
