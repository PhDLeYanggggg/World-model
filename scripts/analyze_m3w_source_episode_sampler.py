"""Read every fixed equal-episode fit under the unchanged original metric."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_m3w_source_episode_sampler import load_config,setup,immutable_json
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_recording_diagnostic import recording_resamples,paired_gain_interval
from src.world_model.m3w_source_crossfit import cost_labels
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    reg=load_config(args.registration);parent,data,cached,control=setup(reg)
    public=ROOT/reg['reports'];report=json.loads((public/'report.json').read_text())
    ids=cached.ids;loc=ids-data.nmain;sites=data.source_sites[loc];records=data.source_records[loc]
    cv=np.linalg.norm(data.target[loc].astype(float),axis=-1).mean(1);native=data.native_scale[loc]
    cv_site=np.array([cv[sites==s].mean() for s in reg['sites']])
    draws=np.random.default_rng(reg['bootstrap_seed']).integers(4,size=(reg['bootstrap_resamples'],4))
    means={};summary={}
    centered_control=json.loads((ROOT/reg['centered_control_report']).read_text())
    for arm,model_arm,source in [('geometry_uniform','geometry',control),
            ('centered_uniform','centered',centered_control),
            ('geometry_event','geometry',report),('centered_event','centered',report)]:
        errors=[];fdes=[]
        for seed in reg['seeds']:
            item=next(v for v in source['oof_labels'] if v['arm']==model_arm and v['seed']==seed)
            path=ROOT/item['path'];assert file_digest(path)==item['sha256']
            with np.load(path,allow_pickle=False) as a:
                np.testing.assert_array_equal(ids,a['ids'])
                labels=cost_labels(a['prediction'],data.target[loc],a['cost_scale'])
                for k,v in labels.items():np.testing.assert_array_equal(v,a[k])
                errors.append(a['ade'].copy());fdes.append(a['fde'].copy())
        errors=np.asarray(errors);mean=errors.mean(0)
        sm=np.array([mean[sites==s].mean() for s in reg['sites']]);means[arm]=sm
        values=100*(1-sm[draws].mean(1)/cv_site[draws].mean(1))
        details=[]
        for site in reg['sites']:
            m=sites==site;fits=[t for t in source['trials'] if t['arm']==model_arm and t['site']==site]
            cutoff=fits[0]['identity']['training_hard_cut'];hard=m&(cv>=cutoff)
            _,inv,counts=recording_resamples(records[m],reg['bootstrap_resamples'],reg['bootstrap_seed'])
            interval=paired_gain_interval(cv[m],mean[m],cv[m],inv,counts)
            details.append(dict(site=site,rows=int(m.sum()),records=len(set(records[m])),
                training_gain_percent=float(np.mean([t['training']['gain_percent'] for t in fits])),
                held_gain_percent=interval['point_percent'],conditional_recording_ci95=interval['conditional_recording_ci95'],
                seed_gains=[float(100*(1-e[m].sum()/cv[m].sum())) for e in errors],
                hard_gain_percent=float(100*(1-mean[hard].sum()/cv[hard].sum())),
                easy_pixel_harm=float((mean[m&(cv==0)]*native[m&(cv==0)]).mean())))
        summary[arm]=dict(result_source='fresh_run' if source is report else 'cached_verified_recomputed',
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
    previous=json.loads((ROOT/reg['centered_control_report']).with_name('analysis.json').read_text())
    for alias,arm in [('geometry_uniform','geometry'),('centered_uniform','centered')]:
        for k in ('equal_site_gain_percent','window_gain_percent','easy_pixel_harm'):
            np.testing.assert_allclose(summary[alias][k],previous['summary'][arm][k],atol=1e-10,rtol=0)
    contrasts=[]
    for a,b in [('geometry_event','geometry_uniform'),('centered_event','centered_uniform'),
                ('centered_event','geometry_event')]:
        diff=means[b]-means[a];values=100*diff[draws].mean(1)/cv_site[draws].mean(1)
        contrasts.append(dict(candidate=a,reference=b,gain_difference_pp=float(100*diff.mean()/cv_site.mean()),
            conditional_four_site_ci95=np.quantile(values,[.025,.975]).tolist()))
    result=dict(result_source='fresh_run_episode_balanced_heads_cached_verified_uniform_controls',
        new_models=24,new_updates=240000,cached_control_models=24,rows=len(ids),sites=4,recordings=29,
        bootstrap_resamples=2000,uncertainty='four_explored_sites_shared_training_conditional_not_confirmation',
        seed_aggregation='mean_errors_not_prediction_ensemble',summary=summary,contrasts=contrasts,
        main_outer_rows_scored=0,new_deployment=False,stage5c_executed=False,smc_enabled=False)
    immutable_json(public/'analysis.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
