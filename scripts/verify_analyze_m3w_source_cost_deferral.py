"""Matched training-only proposal/deferral diagnosis; no held labels scored."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_m3w_source_cost_deferral import load_config, build_data, training_metrics
import numpy as np
import torch
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_recording_diagnostic import error_summary
from src.world_model.m3w_offline_visual_data import json_write


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    reg=load_config(args.registration)
    private,public=ROOT/reg['output'],ROOT/reg['reports']
    report=json.loads((public/'report.json').read_text())
    replay=json.loads((public/'replay.json').read_text())
    assert report['completed_branches']==6 and report['additional_updates']==48000
    assert replay['identity']==report['identity'] and replay['all_exact'] and len(set(replay['replayed']))==24
    data,ids,weights,scale,parents,controls=build_data(reg)
    loc=ids-data.nmain
    target,native=data.target[loc].astype(float),data.native_scale[loc]
    cv=np.linalg.norm(target,axis=-1).mean(1)
    zero,moving=cv==0,cv>0
    hard_cut=float(np.quantile(cv,.9)); hard=cv>=hard_cut
    paths={private/'identity.json',public/'report.json',public/'replay.json',public/'input_checks.json'}
    previous=json.loads((ROOT/reg['control_report']).with_name('analysis.json').read_text())
    for name,digest in previous['verification']['artifact_hashes'].items():
        assert file_digest(ROOT/name)==digest
        paths.add(ROOT/name)
    paths.add(ROOT/reg['control_report'])
    dense,control_rows={},[]
    for seed,control in controls.items():
        for m in control['milestones']:
            with np.load(ROOT/m['prediction_path'],allow_pickle=False) as a:
                np.testing.assert_array_equal(a['ids'],ids)
                prediction=a['prediction'].astype(float)
            dense[seed,m['step']]=np.linalg.norm(prediction-target,axis=-1).mean(1)
            fde=np.linalg.norm(prediction[:,-1]-target[:,-1],axis=-1)
            metrics=error_summary(dense[seed,m['step']],fde,cv,native,np.any(prediction!=0,axis=(1,2)),hard)
            control_rows.append(dict(variant='dense_control',seed=seed,step=m['step'],mode='proposal',**metrics))
    rows,sites,calibration,pairs=[],[],[],[]
    last_states={}
    for trial in report['trials']:
        checkpoint=ROOT/trial['checkpoint_path']
        assert file_digest(checkpoint)==trial['checkpoint_sha256']
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['step']==10000 and state['identity']==trial['identity'] and state['config']==reg['training']
        assert state['variant']==trial['variant'] and state['normalizer']==scale
        assert state['draw_counts'].sum()==640000 and trial['parameters']==44897
        assert all(torch.isfinite(v).all() for v in state['model'].values())
        last_states[trial['seed'],trial['variant']]=state
        paths.update([checkpoint,private/'trials'/f"{trial['trial']}.json"])
        for m in trial['milestones']:
            for kind in ('checkpoint','prediction','receipt'):
                path=ROOT/m[kind+'_path']; assert file_digest(path)==m[kind+'_sha256']; paths.add(path)
            with np.load(ROOT/m['prediction_path'],allow_pickle=False) as a:
                np.testing.assert_array_equal(a['ids'],ids)
                proposal,score=a['proposal'].astype(float),a['score'].copy()
            assert proposal.shape==target.shape and np.isfinite(proposal).all() and np.isfinite(score).all()
            bound=data.radius[loc]*np.linalg.svd(data.rotation[loc],compute_uv=False)[:,0]
            assert np.all(np.linalg.norm(proposal,axis=-1)<=bound[:,None]*(1+1e-5)+1e-8)
            assert not np.any(proposal[~data.support[loc]])
            recomputed=training_metrics(proposal,score,target,native,hard_cut,scale,trial['variant'])
            assert recomputed==m['metrics']
            use=score>0
            emitted=np.where(use[:,None,None],proposal,0.)
            assert not np.any(emitted[~use])
            error=np.linalg.norm(proposal-target,axis=-1).mean(1)
            gain=(cv-error)/scale
            current=torch.load(ROOT/m['checkpoint_path'],map_location='cpu',weights_only=False)
            control_m=next(x for x in controls[trial['seed']]['milestones'] if x['step']==m['step'])
            control_cp=torch.load(ROOT/control_m['checkpoint_path'],map_location='cpu',weights_only=False)
            np.testing.assert_array_equal(current['draw_counts'],control_cp['draw_counts'])
            assert torch.equal(current['sampler_rng'],control_cp['sampler_rng'])
            assert torch.equal(current['torch_rng'],control_cp['torch_rng'])
            if m['step']==2000:
                assert not score.any()
                for name,value in parents[trial['seed']][1]['model'].items():
                    assert torch.equal(value,current['model'][name])
            for mode,prediction in [('proposal',proposal),('hard_action',emitted)]:
                ade=np.linalg.norm(prediction-target,axis=-1).mean(1)
                fde=np.linalg.norm(prediction[:,-1]-target[:,-1],axis=-1)
                changed=np.any(prediction!=0,axis=(1,2))
                metrics=error_summary(ade,fde,cv,native,changed,hard)
                base=dense[trial['seed'],m['step']]
                row=dict(variant=trial['variant'],seed=trial['seed'],step=m['step'],mode=mode,**metrics,
                    gain_over_dense_pp=float(100*(base.sum()-ade.sum())/cv.sum()),
                    requested_rate=float(use.mean()) if mode=='hard_action' else 1.,
                    mean_soft_gate=recomputed['mean_soft_gate'],expected_action_gain_percent=recomputed['expected_action_gain_percent'],
                    zero_action_rows=int((~changed).sum()),all_baseline=bool(not changed.any()),
                    clip_fraction=state['clipped_updates']/8000 if m['step']==10000 else None)
                control_easy=float((base[zero]*native[zero]).mean())
                row['no_greater_easy_harm_than_dense']=bool(row['easy_pixel_harm']<=control_easy)
                row['training_signal_not_research_gate']=bool(row['gain_percent']>0 and row['gain_over_dense_pp']>0
                    and (row['moving_gain_percent']>0 or row['hard_gain_percent']>0)
                    and row['no_greater_easy_harm_than_dense'])
                rows.append(row)
                for site in sorted(set(data.source_sites[loc])):
                    mask=data.source_sites[loc]==site
                    sites.append(dict(variant=trial['variant'],seed=trial['seed'],step=m['step'],mode=mode,site=site,
                        rows=int(mask.sum()),gain_percent=float(100*(1-ade[mask].sum()/cv[mask].sum())),
                        actual_changed_rate=float(changed[mask].mean()),native_pixel_ade=float((ade[mask]*native[mask]).mean())))
            calibration.append(dict(variant=trial['variant'],seed=trial['seed'],step=m['step'],
                interpretation='in_sample_signed_cost_fit' if trial['variant']=='cost_supervised' else 'unconstrained_action_logit_not_gain_regression',
                score_rmse=float(np.sqrt(np.mean((score-gain)**2))),
                constant_training_mean_rmse=float(np.std(gain)),zero_predictor_rmse=float(np.sqrt(np.mean(gain**2))),
                score_mae=float(np.abs(score-gain).mean()),mean_label=float(gain.mean()),mean_score=float(score.mean()),
                score_std=float(score.std()),label_std=float(gain.std()),independent_calibration=False))
    for seed in reg['seeds']:
        a,b=[last_states[seed,v] for v in reg['variants']]
        control=torch.load(ROOT/controls[seed]['checkpoint_path'],map_location='cpu',weights_only=False)
        np.testing.assert_array_equal(a['draw_counts'],b['draw_counts'])
        np.testing.assert_array_equal(a['draw_counts'],control['draw_counts'])
        assert torch.equal(a['sampler_rng'],b['sampler_rng']) and torch.equal(a['sampler_rng'],control['sampler_rng'])
        generator=torch.Generator(); generator.set_state(parents[seed][1]['sampler_rng'])
        draws=parents[seed][1]['draw_counts'].copy()
        for _ in range(8000):
            batch=torch.multinomial(torch.as_tensor(weights,dtype=torch.float64),64,replacement=True,generator=generator).numpy()
            np.add.at(draws,batch,1)
        np.testing.assert_array_equal(draws,a['draw_counts'])
        assert torch.equal(generator.get_state(),a['sampler_rng'])
        pairs.append(dict(seed=seed,all_milestones_matched=True,three_way_draw_stream=True,
            mean_draws_per_row=float(draws.mean()),minimum_draws=int(draws.min()),maximum_draws=int(draws.max())))
    hashes={str(p.relative_to(ROOT)):file_digest(p) for p in sorted(paths)}
    proc=subprocess.run([sys.executable,'scripts/run_m3w_source_cost_deferral.py','--registration',str(args.registration)],
                        cwd=ROOT,capture_output=True,text=True,check=True)
    event=[json.loads(x) for x in proc.stdout.splitlines() if x.startswith('{')][-1]
    assert event['state']=='completed_resume_verified' and event['new_updates']==event['new_branches']==0
    assert hashes=={str(p.relative_to(ROOT)):file_digest(p) for p in sorted(paths)}
    summaries=[]
    for variant in reg['variants']+['dense_control']:
        for step in reg['training']['milestones']:
            for mode in (['proposal'] if variant=='dense_control' else ['proposal','hard_action']):
                selected=[r for r in rows+control_rows if r['variant']==variant and r['step']==step and r['mode']==mode]
                assert len(selected)==3
                value=dict(variant=variant,step=step,mode=mode)
                for field in ('gain_percent','moving_gain_percent','hard_gain_percent','easy_pixel_harm','actual_changed_rate',
                              'binary_oracle_gain_percent','native_pixel_ade','tail_ade95','tail_ade99'):
                    value[field]=float(np.mean([r[field] for r in selected]))
                value['gain_seed_range']=[min(r['gain_percent'] for r in selected),max(r['gain_percent'] for r in selected)]
                if variant!='dense_control':
                    value['all_baseline_seeds']=sum(r['all_baseline'] for r in selected)
                    value['training_signal_seeds']=sum(r['training_signal_not_research_gate'] for r in selected)
                    value['gain_over_dense_pp']=float(np.mean([r['gain_over_dense_pp'] for r in selected]))
                summaries.append(value)
    analysis=dict(result_source='fresh_run_training_only_analysis_cached_verified_controls',
        report_sha256=file_digest(public/'report.json'),summaries=summaries,
        exact_milestone_replays=24,paired_streams=pairs,unchanged_artifacts=len(paths),artifact_hashes=hashes,
        completed_resume=event,training_rows=len(ids),zero_target_rows=int(zero.sum()),nonzero_rows=int(moving.sum()),
        hard_rows=int(hard.sum()),training_cost_scale=scale,training_hard_cut=hard_cut,
        held_rows_scored=0,main_rows_scored=0,independent_confirmation=False,new_deployment=False,
        main_primary_changed=False,sealed_roles_opened=False,stage5c_executed=False,smc_enabled=False)
    json_write(public/'analysis.json',analysis)
    write_csv(public/'training_metrics.csv',rows)
    write_csv(public/'dense_control_metrics.csv',control_rows)
    write_csv(public/'training_site_metrics.csv',sites)
    write_csv(public/'signed_cost_fit.csv',calibration)
    write_csv(public/'batch_trace.csv',[dict(trial=t['trial'],**point) for t in report['trials'] for point in t['fit']['trace']])
    lines=['# Training-Only Cost Deferral Results','','## Material Passport','',
        'Six fresh continuations,48000 new updates; three dense controls and all ancestors cached_verified.',
        'All15430 training rows. No held/main forecasts, threshold selection or deployment.',
        'Three-seed mean and range describe optimization, not generalization uncertainty.','',
        '| Variant | Step | Output | Gain vsCV (%) [seed range] | Moving gain (%) | Hard gain (%) | Zero-target pixel harm | Actual intervention | Training-signal seeds |',
        '| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for s in summaries:
        low,high=s['gain_seed_range']
        lines.append(f"| {s['variant']} | {s['step']} | {s['mode']} | {s['gain_percent']:+.6f} [{low:+.6f},{high:+.6f}] | {s['moving_gain_percent']:+.6f} | {s['hard_gain_percent']:+.6f} | {s['easy_pixel_harm']:.8f} | {s['actual_changed_rate']:.3%} | {s.get('training_signal_seeds','n/a')} |")
    lines += ['','## Checks and Limits','',
        f"24exact prediction/score replays,3matched streams with dense controls,all milestones verified. {len(paths)} artifacts unchanged on completed resume,zero updates.",
        'Hard action returns the exact baseline at score<=0. Expected soft-action risk is a training surrogate, not deterministic trajectory accuracy or a calibrated probability.',
        'Easy percentage is undefined at zero baselineerror. Training-signal counts do not pass the original easy gate or an independent evaluation gate. All-baseline collapse does not count as predictive success.',
        'The new head has44897parameters versus44864for the reused dense head. A synthetic dense-control execution is exact; the extra33gate parameters are the intentional architecture change.',
        'Future errors supply loss targets only. Cost-supervised targets are detached, but computed from a jointly changing candidate; their in-sample fit is not independently calibrated risk.',
        'Offline annotation inputs and source stride12/+144rawframes; no meters/seconds, human intentiongold, true3D or foundation claim. Stage5C/SMC remainoff.','']
    (public/'results.md').write_text('\n'.join(lines))
    cache=private/'plot_runtime';cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache/'matplotlib'));os.environ.setdefault('XDG_CACHE_HOME',str(cache/'xdg'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt']='m3w-source-cost-deferral-v1'
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.6))
    colors={'expected_cost':'#357289','cost_supervised':'#a45351','dense_control':'#6a7852'}
    for variant in reg['variants']+['dense_control']:
        for mode in (['proposal'] if variant=='dense_control' else ['proposal','hard_action']):
            cells=[s for s in summaries if s['variant']==variant and s['mode']==mode]
            x=[s['step'] for s in cells]
            label=variant+(' / hard' if mode=='hard_action' else ' / proposal')
            for ax,field in zip(axes,['gain_percent','easy_pixel_harm','actual_changed_rate']):
                ax.plot(x,[s[field] for s in cells],marker='o',markersize=3,
                        linestyle='-' if mode=='hard_action' or variant=='dense_control' else '--',color=colors[variant],label=label)
    for ax,title in zip(axes,['Full training gain vsCV (%)','Zero-target harm (annotation pixels)','Actual changed-prediction rate']):
        ax.set_ylabel(title);ax.set_xlabel('Global optimizer step');ax.set_xticks([2000,4000,6000,10000],['2k','4k','6k','10k'])
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2);ax.axhline(0,color='black',linewidth=.7,linestyle=':')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,fontsize=8)
    fig.suptitle('Exact-baseline action versus candidate learning: training-only comparison')
    fig.text(.5,.012,'Three-seed means. No held-source score, risk calibration, model selection or deployment.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.81))
    fig.savefig(public/'training_curves.svg',metadata={'Date':None});fig.savefig(cache/'training_curves.png',dpi=150);plt.close(fig)
    svg=public/'training_curves.svg';svg.write_text('\n'.join(x.rstrip() for x in svg.read_text().splitlines())+'\n')
    print(json.dumps(dict(verification=dict(exact_replays=24,unchanged_artifacts=len(paths),completed_resume=event),
                         final=[s for s in summaries if s['step']==10000]),indent=2))


if __name__=='__main__':
    main()
