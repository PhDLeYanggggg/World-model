"""Fixed visual contrasts, explicit prior controls and no held-label calibration."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_m3w_source_visual_start import VisualCorpus
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_motion_start_probe import paired_agent_interval
from src.world_model.m3w_offline_visual_data import json_write


def paired_brier_parts(y,p,reference):
    y,p,reference=map(lambda v:np.asarray(v,dtype=float),(y,p,reference))
    if y.ndim!=1 or y.shape!=p.shape or y.shape!=reference.shape or not len(y):
        raise ValueError('Aligned nonempty probability vectors required')
    def terms(v):
        mean=float(v.mean()); rate=float(y.mean())
        return (mean-rate)**2,float(np.var(v)-2*np.mean((v-mean)*(y-rate)))
    p_bias,p_variation=terms(p); r_bias,r_variation=terms(reference)
    lift=float(np.mean((reference-y)**2-(p-y)**2))
    np.testing.assert_allclose(lift,r_bias-p_bias+r_variation-p_variation,atol=1e-12)
    return dict(mean_shift_contribution=r_bias-p_bias,
        within_site_variation_contribution=r_variation-p_variation,total_lift=lift,
        held_label_diagnostic_not_inference_calibration=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True); args=parser.parse_args()
    reg=json.loads(args.registration.read_text()); reports=ROOT/reg['reports']; out=ROOT/reg['output']
    rp=reports/'report.json'; report=json.loads(rp.read_text()); assert report['complete']
    data=VisualCorpus(reg)
    assert all(report['identity'][k]==v for k,v in data.identity.items())
    source_prior=float(data.y[data.nmain:].mean())
    lookup={(t['arm'],t['schedule'],t['seed'],t['fold']):t for t in report['trials']}
    predictions={}
    for t in report['trials']:
        assert file_digest(ROOT/t['prediction_path'])==t['prediction_sha256']
        with np.load(ROOT/t['prediction_path'],allow_pickle=False) as a:
            predictions[t['trial']]=(a['held_indices'].copy(),a['probability'].copy())
    rows=[]; losses=[]; decompositions=[]; contrasts=[]
    for t in report['trials']:
        ids,p=predictions[t['trial']]
        control=lookup['mask_only',t['schedule'],t['seed'],t['fold']]
        control_ids,control_p=predictions[control['trial']]
        np.testing.assert_array_equal(control_ids,ids)
        for e in t['evaluation']:
            select=data.folds[ids]==e['fold']; y=data.y[ids][select]; p1=p[select]; c1=control_p[select]
            prior=np.full(len(y),e['reference_prior']); own=np.full(len(y),t['training_weighted_prior'])
            source=np.full(len(y),source_prior)
            score=lambda v:float(np.mean((v-y)**2))
            parts=paired_brier_parts(y,p1,c1)
            row=dict(trial=t['trial'],arm=t['arm'],schedule=t['schedule'],seed=t['seed'],
                held_site=['ETH','Hotel'][e['fold']],rows=len(y),agents=e['agents'],brier=e['brier'],
                main_prior_lift=e['brier_lift'],mask_lift=score(c1)-score(p1),own_prior_lift=score(own)-score(p1),
                source_prior_lift=score(source)-score(p1),auroc=e['auroc'],auprc=e['auprc'],ece=e['ece'],
                log_loss=e['log_loss'],training_weighted_brier=t['training_weighted_brier'],
                mask_mean_shift=parts['mean_shift_contribution'],mask_within_site=parts['within_site_variation_contribution'])
            rows.append(row)
            decompositions.append(dict(trial=t['trial'],fold=e['fold'],vs_mask=parts,
                vs_main_prior=paired_brier_parts(y,p1,prior),vs_own_prior=paired_brier_parts(y,p1,own),
                vs_source_prior=paired_brier_parts(y,p1,source)))
        for loss in t['fit']['losses']:
            losses.append(dict(trial=t['trial'],**loss))
    for arm in reg['arms']:
        for schedule in reg['schedules']:
            for fold in (0,1):
                ps=[]; masks=[]; priors=[]; owns=[]; sources=[]
                for seed in reg['seeds']:
                    t=lookup[arm,schedule,seed,-1 if schedule=='source_only' else fold]
                    ids,p=predictions[t['trial']]; selected=data.folds[ids]==fold
                    ps.append(p[selected]); c=lookup['mask_only',schedule,seed,t['fold']]
                    masks.append(predictions[c['trial']][1][selected])
                    e=next(v for v in t['evaluation'] if v['fold']==fold)
                    priors.append(np.full(selected.sum(),e['reference_prior']))
                    owns.append(np.full(selected.sum(),t['training_weighted_prior']))
                    sources.append(np.full(selected.sum(),source_prior))
                y=data.y[:data.nmain][data.folds==fold]; groups=data.stationary_tracks[data.folds==fold]
                selected_rows=[r for r in rows if r['arm']==arm and r['schedule']==schedule and r['held_site']==['ETH','Hotel'][fold]]
                row=dict(arm=arm,schedule=schedule,fold=fold,site=['ETH','Hotel'][fold],
                    means={k:float(np.mean([r[k] for r in selected_rows])) for k in
                        ('brier','main_prior_lift','mask_lift','own_prior_lift','source_prior_lift','auroc','auprc','ece','mask_mean_shift','mask_within_site')},
                    positive_mask_seeds=sum(r['mask_lift']>0 for r in selected_rows),
                    mask_interval=paired_agent_interval(y,np.array(ps),np.array(masks),groups),
                    own_prior_interval=paired_agent_interval(y,np.array(ps),np.array(owns),groups),
                    source_prior_interval=paired_agent_interval(y,np.array(ps),np.array(sources),groups),
                    main_prior_interval=paired_agent_interval(y,np.array(ps),np.array(priors),groups))
                contrasts.append(row)
    for filename,items in [('fit_metrics.csv',rows),('loss_trace.csv',losses)]:
        with (reports/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(items[0]),lineterminator='\n'); writer.writeheader(); writer.writerows(items)
    rgb=[v for v in contrasts if v['arm']=='past_rgb']
    result=dict(result_source='fresh_run_analysis_of_fixed_predictions',report_sha256=file_digest(rp),
        source_prior=source_prior,contrasts=contrasts,decompositions=decompositions,
        bidirectional_mean_rgb_over_mask_schedules=[s for s in reg['schedules'] if all(v['means']['mask_lift']>0 for v in rgb if v['schedule']==s)],
        bidirectional_mean_rgb_over_own_prior_schedules=[s for s in reg['schedules'] if all(v['means']['own_prior_lift']>0 for v in rgb if v['schedule']==s)],
        independent_confirmation=False,main_primary_changed=False,new_forecasting_lift=False,
        no_held_label_calibration=True,no_deployment=True)
    json_write(reports/'analysis.json',result)
    lines=['# Matched Visual Start-Information Results','',
        'All means average seed losses, not prediction ensembles. Positive lift means lower Brier, not ADE/FDE.',
        'The primary contrast is RGB versus the same-schedule coverage control. All sites are exposed fit roles.','',
        '| Arm | Schedule | Held site | Brier | Lift vs mask | Lift vs own train prior | Lift vs source prior | AUROC | ECE |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for v in contrasts:
        m=v['means']
        lines.append(f"| {v['arm']} | {v['schedule']} | {v['site']} | {m['brier']:.6f} | {m['mask_lift']:.6f} | {m['own_prior_lift']:.6f} | {m['source_prior_lift']:.6f} | {m['auroc']:.6f} | {m['ece']:.6f} |")
    lines+=['','## RGB Incremental Uncertainty','',
        'The intervals below are conditional agent-balanced contrasts; row means above have a different estimand.',
        'Five ETH and26HotelIDs,overlapping windows and two exposed sites do not establish independent scene generalization.','',
        '| Schedule | Held site | Agent-balanced RGB lift [95% interval] | Row-mean bias-shift term | Row-mean varying-prediction term | Positive seeds |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for v in rgb:
        b=v['mask_interval']; m=v['means']; lo,hi=b['descriptive_ci95']
        lines.append(f"| {v['schedule']} | {v['site']} | {b['agent_balanced_lift']:.6f} [{lo:.6f}, {hi:.6f}] | {m['mask_mean_shift']:.6f} | {m['mask_within_site']:.6f} | {v['positive_mask_seeds']}/3 |")
    lines+=['','Brier decomposition uses held labels only to explain fixed predictions,never to recalibrate or select.',
        'No forecast policy,physical-time,metric,true3D,foundation,Stage5C or SMC claim.','']
    (reports/'results.md').write_text('\n'.join(lines))
    cache=out/'plot_runtime'; cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache/'matplotlib')); os.environ.setdefault('XDG_CACHE_HOME',str(cache/'xdg'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt']='m3w-source-visual-start-v1'
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharex=True,sharey=True)
    for fold,ax in enumerate(axes):
        for i,schedule in enumerate(reg['schedules']):
            v=next(v for v in rgb if v['fold']==fold and v['schedule']==schedule)
            ax.barh(i,v['means']['mask_lift'],height=.6,color=['#355c8c','#00856a','#b14450'][i],alpha=.8)
            values=[r['mask_lift'] for r in rows if r['arm']=='past_rgb' and r['schedule']==schedule and r['held_site']==v['site']]
            ax.scatter(values,np.full(3,i),c='black',s=15,zorder=3)
        ax.axvline(0,color='black',lw=.8); ax.set_title(['ETH (5 agent IDs)','Hotel (26 agent IDs)'][fold])
        ax.set_xlabel('RGB Brier lift over matched coverage control')
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='x',alpha=.15)
    axes[0].set_yticks(range(3),['Main only','SDD only','Mixed']); axes[0].invert_yaxis()
    fig.suptitle('Past-image contribution: fixed three-seed probability probe')
    fig.text(.5,.01,'Positive is better; bars average seed losses, points are seeds. Not a trajectory or deployment result.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.96)); fig.savefig(reports/'visual_probability_lift.svg',metadata={'Date':None})
    fig.savefig(cache/'visual_probability_lift.png',dpi=130); plt.close(fig)
    svg=reports/'visual_probability_lift.svg'; svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('contrasts','decompositions')},indent=2))


if __name__=='__main__':
    main()
