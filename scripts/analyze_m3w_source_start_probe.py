"""Descriptive complete-matrix analysis; no model or decision threshold selection."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write


def brier_decomposition(y,p,reference_prior):
    y,p=np.asarray(y,dtype=float),np.asarray(p,dtype=float)
    if y.shape!=p.shape or y.ndim!=1 or not len(y):
        raise ValueError('Aligned nonempty row scores required')
    rate=float(y.mean()); mean=float(p.mean()); variance=float(np.var(p))
    covariance=float(np.mean((p-mean)*(y-rate)))
    calibration=(reference_prior-rate)**2-(mean-rate)**2
    variation=2*covariance-variance
    observed=float(np.mean((reference_prior-y)**2-(p-y)**2))
    np.testing.assert_allclose(calibration+variation,observed,rtol=1e-10,atol=1e-12)
    return dict(held_rate=rate,mean_probability=mean,probability_variance=variance,
        probability_label_covariance=covariance,mean_shift_contribution=calibration,
        within_site_variation_contribution=variation,total_brier_lift=observed,
        held_label_diagnostic_only=True,not_an_inference_calibration=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    args=parser.parse_args(); reg=json.loads(args.registration.read_text()); reports=ROOT/reg['reports']
    path=reports/'report.json'; report=json.loads(path.read_text()); inputs=json.loads((reports/'input_checks.json').read_text())
    assert report['complete'] and len(report['trials'])==45
    parent=json.loads((ROOT/reg['parent_registration']).read_text())
    main_reg=json.loads((ROOT/parent['main_registration']).read_text()); main_dir=ROOT/main_reg['output']/'inputs'
    manifest=json.loads((main_dir/'data_manifest.json').read_text())
    assert file_digest(main_dir/'data_manifest.json')==report['identity']['main_manifest_sha256']
    for name in ('geometry.npy','targets.npy','folds.npy'):
        assert file_digest(main_dir/name)==manifest['arrays'][name]
    geometry=np.load(main_dir/'geometry.npy',mmap_mode='r',allow_pickle=False)
    main_ids=np.flatnonzero(np.all(geometry[:,:16]==0,axis=1))
    y_all=np.any(np.load(main_dir/'targets.npy',mmap_mode='r',allow_pickle=False)[main_ids]!=0,axis=(1,2)).astype(int)
    folds=np.load(main_dir/'folds.npy',allow_pickle=False)[main_ids]
    trial_rows=[]; loss_rows=[]; contrasts=[]; decomposition=[]
    source_rate=inputs['source_class_counts'][1]/sum(inputs['source_class_counts'])
    for t in report['trials']:
        assert file_digest(ROOT/t['prediction_path'])==t['prediction_sha256']
        with np.load(ROOT/t['prediction_path'],allow_pickle=False) as a:
            ids,p=a['held_indices'],a['probability']
        for e in t['evaluation']:
            rate=e['positive_rate']; source_prior_brier=rate*(1-source_rate)**2+(1-rate)*source_rate**2
            mask=folds[ids]==e['fold']
            parts=brier_decomposition(y_all[ids][mask],p[mask],e['reference_prior'])
            np.testing.assert_allclose(parts['total_brier_lift'],e['brier_lift'],atol=1e-12)
            decomposition.append(dict(trial=t['trial'],family=t['family'],schedule=t['schedule'],
                fold=e['fold'],**parts))
            trial_rows.append(dict(trial=t['trial'],family=t['family'],schedule=t['schedule'],seed=t['seed'],
                held_site=['ETH','Hotel'][e['fold']],rows=e['rows'],agents=e['agents'],
                positive_rate=rate,reference_prior=e['reference_prior'],brier=e['brier'],
                brier_lift=e['brier_lift'],auroc=e['auroc'],auprc=e['auprc'],ece=e['ece'],log_loss=e['log_loss'],
                training_weighted_brier=t['training_weighted_brier'],clipping_fraction=t['held_feature_clipping_fraction'],
                source_constant_prior_brier=source_prior_brier,source_constant_prior_lift=source_prior_brier-e['brier']))
        for row in t['fit'].get('losses',[]):
            loss_rows.append(dict(trial=t['trial'],**row))
    for name,items in report['summary'].items():
        for e in items:
            contrasts.append(dict(arm=name,site=['ETH','Hotel'][e['fold']],
                mean_brier=e['mean_brier'],mean_brier_lift=e['mean_brier_lift'],
                mean_lift_vs_main_only=e['mean_lift_vs_main_only'],mean_auroc=e['mean_auroc'],
                mean_auprc=e['mean_auprc'],mean_ece=e['mean_ece'],positive_seeds=e['positive_seeds'],
                vs_prior_agent_balanced_lift=e['vs_prior_agent_interval']['agent_balanced_lift'],
                vs_prior_agent_low=e['vs_prior_agent_interval']['descriptive_ci95'][0],
                vs_prior_agent_high=e['vs_prior_agent_interval']['descriptive_ci95'][1],
                vs_main_agent_balanced_lift=e['vs_main_agent_interval']['agent_balanced_lift'],
                vs_main_agent_low=e['vs_main_agent_interval']['descriptive_ci95'][0],
                vs_main_agent_high=e['vs_main_agent_interval']['descriptive_ci95'][1]))
    for name,rows in [('fit_metrics.csv',trial_rows),('loss_trace.csv',loss_rows),('contrasts.csv',contrasts)]:
        with (reports/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    diagnostic=dict(result_source='fresh_run_analysis_of_fixed_predictions',report_sha256=file_digest(path),
        source_start_rate=source_rate,comparison_not_used_to_select_or_fit=True,
        posthoc_source_constant_prior_comparison=True,
        bidirectionally_positive_arms=[name for name,items in report['summary'].items() if all(e['mean_brier_lift']>0 for e in items)],
        bidirectionally_source_helpful_arms=[name for name,items in report['summary'].items() if all(e['mean_lift_vs_main_only']>0 for e in items)],
        model_warnings=[dict(trial=t['trial'],warnings=t['warnings']) for t in report['trials'] if t['warnings']],
        fit_seconds_by_family={f:sum(t['fit']['seconds'] for t in report['trials'] if t['family']==f) for f in reg['models']},
        contrasts=contrasts,brier_decomposition=decomposition,sealed_roles_opened=False,new_forecasting_lift=False)
    json_write(reports/'analysis.json',diagnostic)
    lines=['# Fixed-Matrix Probability Contrasts','',
        'Positive Brier lift means lower squared probability error. It is not trajectory improvement.',
        'Intervals give equal weight to held agents; main point estimates give equal weight to rows.',
        'These are different descriptive estimands. Neither is a new-site confirmation interval.','',
        '| Arm | Held site | Brier lift vs main prior | Agent-balanced lift vs prior [95% interval] | Lift vs main-only | Agent-balanced lift vs main [95% interval] |',
        '| --- | --- | ---: | --- | ---: | --- |']
    for e in contrasts:
        lines.append(f"| {e['arm']} | {e['site']} | {e['mean_brier_lift']:.6f} | {e['vs_prior_agent_balanced_lift']:.6f} [{e['vs_prior_agent_low']:.6f}, {e['vs_prior_agent_high']:.6f}] | {e['mean_lift_vs_main_only']:.6f} | {e['vs_main_agent_balanced_lift']:.6f} [{e['vs_main_agent_low']:.6f}, {e['vs_main_agent_high']:.6f}] |")
    lines+=['','Source constant-prior scores in fit_metrics.csv are post hoc analytic diagnostics, not another selected classifier.',
        'No independence claim for overlapping windows, contemporaneous agents or repeated fits.',
        'No pooled best arm, intervention, threshold search or independent confirmation.','']
    (reports/'contrasts.md').write_text('\n'.join(lines))
    plot_cache=ROOT/reg['output']/'plot_runtime'; plot_cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(plot_cache/'matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME',str(plot_cache/'xdg'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt']='m3w-source-start-v1'
    names=list(report['summary']); colors={'logistic':'#355c8c','extra_trees':'#00856a','mlp':'#b14450'}
    fig,axes=plt.subplots(1,2,figsize=(11,5),sharey=True)
    for fold,ax in enumerate(axes):
        for i,name in enumerate(names):
            values=[t['brier_lift'] for t in trial_rows if t['family']+'_'+t['schedule']==name and t['held_site']==['ETH','Hotel'][fold]]
            color=colors[next(f for f in colors if name.startswith(f+'_'))]
            ax.barh(i,np.mean(values),color=color,alpha=.75,height=.65)
            ax.scatter(values,np.full(len(values),i),c='black',s=9,zorder=3)
        ax.axvline(0,color='black',lw=.8); ax.set_title(['ETH: 81 windows / 5 agents','Hotel: 284 windows / 26 agents'][fold])
        ax.set_xlabel('Brier lift vs main training prior (>0 is better)')
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='x',alpha=.18)
    axes[0].set_yticks(range(len(names)),[n.replace('_',' ') for n in names]); axes[0].invert_yaxis()
    fig.suptitle('Fixed source-support comparison: probability, not forecasting')
    fig.text(.5,.01,'Bars average three seed losses; points are seeds. Exposed fit sites only; no independent confirmation.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.96)); fig.savefig(reports/'probability_transfer.svg',metadata={'Date':None})
    fig.savefig(plot_cache/'probability_transfer.png',dpi=130); plt.close(fig)
    svg=reports/'probability_transfer.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(json.dumps({k:v for k,v in diagnostic.items() if k not in ('contrasts','brier_decomposition')},indent=2))


if __name__=='__main__':
    main()
