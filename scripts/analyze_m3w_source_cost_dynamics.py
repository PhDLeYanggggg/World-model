"""Fixed forecast-cost contrasts; all sites/seeds retained, no policy search."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_m3w_source_cost_dynamics import DynamicsCorpus, load_config
import numpy as np
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write


def site_ratio_interval(model,reference,*,resamples=2000,seed=432):
    model,reference=map(lambda v:np.asarray(v,dtype=float),(model,reference))
    if (model.ndim!=1 or model.shape!=reference.shape or len(model)<2
            or not np.isfinite(model).all() or not np.isfinite(reference).all()
            or np.any(model<0) or np.any(reference<=0)):
        raise ValueError('Paired finite nonnegative errors per physical site required')
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(model),(resamples,len(model)))
    draws=100*(1-model[indices].mean(1)/reference[indices].mean(1))
    return dict(model_error=float(model.mean()),reference_error=float(reference.mean()),
        gain_percent=100*float(1-model.mean()/reference.mean()),
        conditional_ci95=np.quantile(draws,[.025,.975]).tolist(),physical_sites=len(model),
        resamples=resamples,independent_confirmation=False,overlapping_training_folds=True)


def blocked_error_interval(model,reference,groups,*,resamples=2000,seed=433):
    model,reference,groups=map(np.asarray,(model,reference,groups))
    if (model.ndim!=2 or model.shape!=reference.shape or model.shape[1]!=len(groups)
            or groups.ndim!=1 or not np.isfinite(model).all() or not np.isfinite(reference).all()):
        raise ValueError('Aligned seed losses and video blocks required')
    m,r=model.mean(0),reference.mean(0)
    unique,inverse=np.unique(groups,return_inverse=True)
    sums=np.bincount(inverse,weights=m);refs=np.bincount(inverse,weights=r)
    counts=np.bincount(inverse)
    value=dict(rows=len(groups),blocks=len(unique),model_error=float(m.mean()),reference_error=float(r.mean()),
        absolute_gain=float((r-m).mean()),gain_percent=100*float(1-m.mean()/r.mean()) if r.mean()>0 else None,
        independent_confirmation=False)
    if len(unique)<2:
        return dict(value,available=False)
    rng=np.random.default_rng(seed);idx=rng.integers(0,len(unique),(resamples,len(unique)))
    absolute=(refs[idx]-sums[idx]).sum(1)/counts[idx].sum(1)
    denominator=refs[idx].sum(1);positive=denominator>0
    relative=100*(1-sums[idx][positive].sum(1)/denominator[positive])
    return dict(value,available=True,absolute_conditional_ci95=np.quantile(absolute,[.025,.975]).tolist(),
        relative_conditional_ci95=np.quantile(relative,[.025,.975]).tolist() if len(relative) else None,
        bootstrap_zero_reference_draws=int((~positive).sum()),resamples=resamples)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--registration',type=Path,required=True)
    args=parser.parse_args();reg=load_config(args.registration);reports=ROOT/reg['reports'];out=ROOT/reg['output']
    rp=reports/'report.json';report=json.loads(rp.read_text())
    if report['completed_models']!=60 or report['optimizer_updates']!=120000:
        raise ValueError('Full fixed trajectory matrix required')
    data=DynamicsCorpus(reg)
    if report['identity']['source_assignment_sha256']!=data.assignment_hash:
        raise ValueError('Source population changed')
    lookup={(t['objective'],t['arm'],t['site'],t['seed']):t for t in report['trials']}
    errors={};rows=[];losses=[];site_summaries=[];videos=[];bound_audits=[]
    for t in report['trials']:
        if file_digest(ROOT/t['prediction_path'])!=t['prediction_sha256']:
            raise ValueError('Saved predictions changed')
        with np.load(ROOT/t['prediction_path'],allow_pickle=False) as p:
            ids=p['held_indices'].copy();prediction=p['prediction'].astype(float);gate=p['gate'].copy()
        expected=np.flatnonzero(data.source_sites==t['site'])+data.nmain
        np.testing.assert_array_equal(ids,expected)
        local=ids-data.nmain;y=data.target[local].astype(float);scale=data.native_scale[local]
        cv=np.linalg.norm(y,axis=-1).mean(1);cvfde=np.linalg.norm(y[:,-1],axis=-1)
        hard=cv>=t['identity']['training_hard_cut'];easy=cv==0
        radius=data.radius[local].astype(float)
        # A containing ball accounts for cached float32 rotation roundoff.
        bound=radius*np.linalg.svd(data.rotation[local].astype(float),compute_uv=False)[:,0]
        bound=np.where(data.support[local],bound,0.)
        norm=np.linalg.norm(prediction,axis=-1)
        if np.any(norm>bound[:,None]+1e-5*np.maximum(bound[:,None],1)):
            raise ValueError('Prediction exceeds registered observed-context bound')
        if np.count_nonzero(prediction[~data.support[local]]):
            raise ValueError('Unsupported context failed to retain baseline')
        oracle=np.maximum(np.linalg.norm(y,axis=-1)-bound[:,None],0).mean(1)
        bound_audits.append(dict(trial=t['trial'],rows=len(ids),unsupported=int((~data.support[local]).sum()),
            output_bound_pass=True,closed_containing_ball_oracle_ade=float(oracle.mean()),cv_ade=float(cv.mean()),
            oracle_future_informed_not_model=True))
        for mode,use in [('uncontrolled',np.ones(len(ids),bool)),('fixed_probability_gate',gate)]:
            chosen=np.where(use[:,None,None],prediction,0)
            ade=np.linalg.norm(chosen-y,axis=-1).mean(1);fde=np.linalg.norm(chosen[:,-1]-y[:,-1],axis=-1)
            e=t['evaluation'][mode]
            np.testing.assert_allclose([ade.mean(),fde.mean(),cv.mean()],[e['ade'],e['fde'],e['cv_ade']],rtol=1e-12)
            errors[t['objective'],t['arm'],t['site'],t['seed'],mode]=(ade,cv,ids)
            rows.append(dict(trial=t['trial'],objective=t['objective'],arm=t['arm'],site=t['site'],seed=t['seed'],mode=mode,
                ade=e['ade'],fde=e['fde'],cv_ade=e['cv_ade'],gain_percent=e['gain_percent'],
                native_pixel_ade=float((ade*scale).mean()),native_pixel_fde=float((fde*scale).mean()),
                native_pixel_cv_ade=float((cv*scale).mean()),native_pixel_cv_fde=float((cvfde*scale).mean()),
                hard_rows=int(hard.sum()),hard_gain_percent=100*float(1-ade[hard].mean()/cv[hard].mean()) if hard.any() and cv[hard].mean()>0 else None,
                easy_rows=int(easy.sum()),easy_absolute_harm=float(ade[easy].mean()) if easy.any() else None,
                easy_native_pixel_harm=float((ade[easy]*scale[easy]).mean()) if easy.any() else None,
                easy_nonzero_predictions=int((ade[easy]>0).sum()),intervention_rate=float(use.mean()),
                oracle_binary_gain_percent=100*float(1-np.minimum(ade,cv).mean()/cv.mean()),
                training_ade=t['training_ade'],training_cv_ade=t['training_cv_ade'],
                training_gain_percent=100*(1-t['training_ade']/t['training_cv_ade'])))
            for recording in sorted(set(data.source_records[local])):
                m=data.source_records[local]==recording
                videos.append(dict(trial=t['trial'],site=t['site'],recording=recording,mode=mode,rows=int(m.sum()),
                    ade=float(ade[m].mean()),cv_ade=float(cv[m].mean()),
                    gain_percent=100*float(1-ade[m].mean()/cv[m].mean()) if cv[m].mean()>0 else None,
                    intervention_rate=float(use[m].mean())))
        losses.extend(dict(trial=t['trial'],**loss) for loss in t['fit']['losses'])
    for objective in reg['objectives']:
        for arm in reg['arms']:
            for mode in ('uncontrolled','fixed_probability_gate'):
                for site in reg['sites']:
                    cells=[errors[objective,arm,site,seed,mode] for seed in reg['seeds']]
                    ps,refs=np.stack([c[0] for c in cells]),np.stack([c[1] for c in cells]);ids=cells[0][2]-data.nmain
                    selected=[r for r in rows if (r['objective'],r['arm'],r['mode'],r['site'])==(objective,arm,mode,site)]
                    mean={k:float(np.mean([r[k] for r in selected])) for k in
                        ('ade','cv_ade','fde','native_pixel_ade','native_pixel_fde','hard_gain_percent',
                         'easy_absolute_harm','easy_native_pixel_harm','easy_nonzero_predictions',
                         'intervention_rate','oracle_binary_gain_percent','training_gain_percent')}
                    tracks=data.source_tracks[ids]
                    agent_m=np.array([ps[:,tracks==g].mean() for g in np.unique(tracks)])
                    agent_r=np.array([refs[:,tracks==g].mean() for g in np.unique(tracks)])
                    site_summaries.append(dict(objective=objective,arm=arm,mode=mode,site=site,means=mean,
                        video_interval=blocked_error_interval(ps,refs,data.source_records[ids],resamples=reg['bootstrap_resamples']),
                        agent_balanced_model_ade=float(agent_m.mean()),agent_balanced_cv_ade=float(agent_r.mean()),
                        positive_seed_fits=sum(r['gain_percent']>0 for r in selected)))
    summary=[];contrasts=[]
    for objective in reg['objectives']:
        for arm in reg['arms']:
            for mode in ('uncontrolled','fixed_probability_gate'):
                cells=[next(s for s in site_summaries if (s['objective'],s['arm'],s['mode'],s['site'])==(objective,arm,mode,site)) for site in reg['sites']]
                result=site_ratio_interval([s['means']['ade'] for s in cells],[s['means']['cv_ade'] for s in cells])
                summary.append(dict(objective=objective,arm=arm,mode=mode,**result,
                    positive_sites=sum(s['video_interval']['gain_percent']>0 for s in cells),
                    positive_seed_fits=sum(s['positive_seed_fits'] for s in cells),
                    equal_site_easy_absolute_harm=float(np.mean([s['means']['easy_absolute_harm'] for s in cells])),
                    equal_site_intervention_rate=float(np.mean([s['means']['intervention_rate'] for s in cells])),
                    easy_percentage_degradation=None,
                    agent_sensitivity=site_ratio_interval([s['agent_balanced_model_ade'] for s in cells],[s['agent_balanced_cv_ade'] for s in cells])))
    # Only uncontrolled image contrasts isolate image content. Same-arm objective
    # comparisons share the frozen gate when a gated contrast is reported.
    def site_errors(objective,arm,mode):
        return [next(s['means']['ade'] for s in site_summaries if (s['objective'],s['arm'],s['mode'],s['site'])==(objective,arm,mode,site)) for site in reg['sites']]
    for objective in reg['objectives']:
        contrasts.append(dict(kind='RGB_vs_mask',objective=objective,mode='uncontrolled',
            **site_ratio_interval(site_errors(objective,'past_rgb','uncontrolled'),site_errors(objective,'mask_only','uncontrolled'))))
    for arm in reg['arms']:
        for mode in ('uncontrolled','fixed_probability_gate'):
            contrasts.append(dict(kind='ADE_vs_log_ADE',arm=arm,mode=mode,
                **site_ratio_interval(site_errors('ade',arm,mode),site_errors('log_ade',arm,mode))))
    result=dict(result_source='fresh_run_analysis_of_cached_verified_fixed_forecasts',report_sha256=file_digest(rp),
        analysis_script_sha256=file_digest(Path(__file__)),summary=summary,contrasts=contrasts,sites=site_summaries,
        bound_audits=bound_audits,main_primary_changed=False,sealed_roles_opened=False,new_deployment=False,
        source_internal_stationary_cohort_only=True,independent_confirmation=False,
        gate_is_uncalibrated_diagnostic=True,gated_RGB_comparison_confound='different_same_arm_classifier_gates',
        easy_percentage_undefined_not_passed=True,stage5c_executed=False,smc_enabled=False)
    json_write(reports/'analysis.json',result)
    for name,items in [('fit_metrics.csv',rows),('video_metrics.csv',videos),('loss_trace.csv',losses)]:
        with (reports/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(items[0]),lineterminator='\n');w.writeheader();w.writerows(items)
    lines=['# Source Trajectory Cost Alignment Results','',
        'Fixed source-fit diagnostic, not the main benchmark or independent confirmation.',
        'Parent-normalized ADE averaged within physical site, then equally across sites. Average seed losses, not ensemble paths.',
        'The fixed0.9 gate is not calibrated safety. All sites and seeds are retained.','',
        '| Objective | Input | Mode | ADE | CV ADE | Gain vs CV (%) [conditional95% site interval] | Positive sites / fits | Mean site easy harm | Mean site intervention |',
        '| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: |']
    for s in summary:
        lo,hi=s['conditional_ci95']
        lines.append(f"| {s['objective']} | {s['arm']} | {s['mode']} | {s['model_error']:.6f} | {s['reference_error']:.6f} | {s['gain_percent']:+.6f} [{lo:+.6f}, {hi:+.6f}] | {s['positive_sites']}/5; {s['positive_seed_fits']}/15 | {s['equal_site_easy_absolute_harm']:.6f} | {s['equal_site_intervention_rate']:.4f} |")
    lines+=['','## Matched Contrasts','',
        '| Comparison | Input/objective | Mode | Gain over matched control (%) [conditional95% site interval] |',
        '| --- | --- | --- | --- |']
    for c in contrasts:
        lo,hi=c['conditional_ci95'];name=c.get('arm',c.get('objective'))
        lines.append(f"| {c['kind']} | {name} | {c['mode']} | {c['gain_percent']:+.6f} [{lo:+.6f}, {hi:+.6f}] |")
    lines+=['','Gated RGB/mask comparisons do not isolate pixels because their frozen classifier gates differ.',
        'Same-arm loss comparisons share exactly the same gate. No gate, model or checkpoint is selected from these scores.',
        'Easy CV error is exactly zero: relative degradation is undefined, not automatically a pass. See absolute harms and native-pixel errors.',
        'Context bounds are not a physical validity certificate. Closed-ball and binary oracles use future labels for diagnosis only.',
        'Source +144rawframes is not seconds-equivalent to main native8-to12. Pixel coordinates are not metric.',
        'Five exposed source sites and overlapping training folds yield conditional sensitivity only. No deployment, Stage5C or SMC.','']
    (reports/'results.md').write_text('\n'.join(lines))
    cache=out/'plot_runtime';cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache/'matplotlib'));os.environ.setdefault('XDG_CACHE_HOME',str(cache/'xdg'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt']='m3w-source-cost-dynamics-v1'
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,mode,title in zip(axes,['uncontrolled','fixed_probability_gate'],['Uncontrolled trajectories','Fixed probability gate (diagnostic)']):
        ss=[s for s in summary if s['mode']==mode]
        ax.barh(range(4),[s['gain_percent'] for s in ss],color=['#357289','#a45351','#357289','#a45351'])
        ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(4),[s['objective']+' / '+s['arm'] for s in ss]);ax.invert_yaxis()
        ax.set_xlabel('Source equal-site ADE gain over CV (%)');ax.set_title(title);ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Actual trajectory-cost objectives: fixed three-seed source-site comparison')
    fig.text(.5,.01,'All stationary source queries retained. Negative is worse. Not independent confirmation or deployment.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.96));fig.savefig(reports/'trajectory_gain.svg',metadata={'Date':None});fig.savefig(cache/'trajectory_gain.png',dpi=130);plt.close(fig)
    svg=reports/'trajectory_gain.svg';svg.write_text('\n'.join(x.rstrip() for x in svg.read_text().splitlines())+'\n')
    print(json.dumps({'summary':summary,'contrasts':contrasts},indent=2))


if __name__=='__main__':
    main()
