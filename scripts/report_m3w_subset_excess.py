"""Deterministic development report; no post-readout model selection."""
import json
from pathlib import Path
import re
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_subset_excess as run
from scripts.report_m3w_fixed_floor_tail import ci


def main():
    d=json.loads((run.PUBLIC/'summary.json').read_text());training=[]
    for ref in json.loads((run.PUBLIC/'training_freeze.json').read_text())['groups']:
        assert run.base.artifact(ROOT/ref['path'])==ref
        doc=json.loads((ROOT/ref['path']).read_text())
        training.extend(dict(group=Path(ref['path']).parent.name,arm=a,**v) for a,v in doc['fits'].items())
    totals={a:dict(heads=sum(r['arm']==a for r in training),updates=sum(r['step'] for r in training if r['arm']==a),
        training_queries=sum(r['supervised_queries'] for r in training if r['arm']==a),
        singleton_queries=sum(r['singleton_queries'] for r in training if r['arm']==a),
        loss_declined_heads=sum(r['trace'][-1]['monitor'][a]<r['trace'][0]['monitor'][a] for r in training if r['arm']==a),
        unknown_draws=sum(r['unknown_rows_sampled'] for r in training if r['arm']==a)) for a in run.api.ARMS}
    run.base.immutable_json(run.PUBLIC/'training_summary.json',totals)
    contrast='subset_aggregate_rank_vs_subset_pointwise_rank';joint='subset_aggregate_joint_vs_subset_pointwise_joint'
    lines=['# Anchored Subset Risk Training','','## Evidence Role','',
        'fresh_run: 216 new risk heads, 432,000 updates and outcome readout. cached_verified: predictors, floor, utility and previous controls.',
        'Twelve opened source-training development localities; 4producer/4controller/2risk-fit/2held roles, three forecasting seeds.',
        'Independent selection/calibration/confirmation remain closed. No held checkpoint or threshold selection. No deployment change.',
        'Image-local detector silver, obs8/pred12 raw-frame stride12. No metric, seconds, human-gold, physical-safety, true3D or foundation claim. No Stage5C or SMC.','',
        '## Registered Development Contrasts','',
        '| Contrast | ADE gain % | Selected harm reduction pp | All-reference harm reduction pp | Intervention difference pp |',
        '|---|---:|---:|---:|---:|']
    for name,m in d['paired'].items():
        lines.append('| '+name+' | '+' | '.join(ci(m[k]) for k in ('ADE_gain_percent','selected_harm_reduction_pp','all_reference_harm_reduction_pp','intervention_difference_pp'))+' |')
    lines+=['','Rank contrasts use the same original per-query intervention count. Joint contrasts do not match intervention counts.',
        'Ten inherited zero-action views leave selected-risk undefined. Fixed-denominator total harm is zero under abstention, not a substitute2%risk certificate. The old primary remains incomplete.',
        '','## Policy Results','','| Policy | ADE/floor % | Hard/floor % | Easy/CV % | FDE/floor % | Intervention % | Selected harm % | All-reference harm % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in d['summary'].items():
        lines.append('| '+name+' | '+' | '.join(ci(m[k]) for k in ('all_gain_floor','hard_gain_floor','easy_gain_CV','FDE_gain_floor'))+' | '+
            ' | '.join(ci(m[k],100) for k in ('intervention_rate','selected_positive_harm_ratio','positive_harm_over_all_floor'))+' |')
    lines+=['','## Coverage, Risk and Tail','','| Policy | Violating /216 | Undefined | Abstaining | Worst easy gain % | P95/floor | Unknown interventions/view |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,m in d['summary'].items():
        w=d['worst_views'][name]
        lines.append(f"| {name} | {w['risk_violating_views']} | {w['undefined_selected_risk_views']} | {w['abstaining_views']} | {w['worst_easy_gain_CV']:.6f} | "+ci(m['p95_ratio_to_floor'])+' | '+ci(m['unknown_interventions'])+' |')
    lines+=['','## Held Risk Prediction','','| Arm | Metric | Estimate and locality95%CI |','|---|---|---:|']
    for arm,metrics in d['quality'].items():
        for key,value in metrics.items():lines.append('| '+arm+' | '+key+' | '+ci(value)+' |')
    lines+=['','Held subset losses average nonempty known subsets; fitting loss includes empty subsets as zero. Their raw magnitudes are not directly comparable.',
        'Subset coverage is built from all causal rows before unknown labels are excluded from supervised loss. Unknown rows stay in inference/action counts.',
        'Controller admission is a fixed past-only proxy from a separately fitted source group, not an exact deployed floor-relative mask.',
        'The individual anchor is shared. Within-subset mean-square error versus square-of-mean error is the only matched objective difference.',
        'The latter is algebraically no larger for a fixed model. That inequality alone is not learned improvement; compare models on the same held metric.',
        '','## Fitting and Gates','',*[f'- {a}: {v}' for a,v in totals.items()],'',*[f'- {k}: {v}' for k,v in d['gates'].items()],
        '','Fitting-query totals repeat data across heads and are not independent samples. CIs use3000locality draws after averaging dependent views; they are development CIs.',
        'Four score outputs are not separately identified/calibrated expected moments. No new collision or physical-safety model was trained.',
        'Retained query cohorts are not a completeness guarantee for every visible agent; missing future labels are not assumed missing at random.',
        'Full legacy tests and cold raw reconstruction are not_run; scoped verification does not substitute independent generalization.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    detail=['# Locality and Seed Breakdown','','| Locality | Matched-rank ADE gain % | Joint ADE gain % | Fixed-denominator rank harm reduction pp |','|---|---:|---:|---:|']
    for site,v in d['paired'][contrast]['ADE_gain_percent']['by_site'].items():
        j=d['paired'][joint]['ADE_gain_percent']['by_site'][site];h=d['paired'][contrast]['all_reference_harm_reduction_pp']['by_site'][site]
        detail.append(f'| {site} | {v:.6f} | {j:.6f} | {h:.6f} |')
    detail+=['','| Seed | Subset pointwise joint ADE/floor % | Subset aggregate joint ADE/floor % |','|---|---:|---:|']
    for seed,m in d['by_seed'].items():detail.append('| '+seed+' | '+ci(m['subset_pointwise_joint']['all_gain_floor'])+' | '+ci(m['subset_aggregate_joint']['all_gain_floor'])+' |')
    detail+=['','All sources remain on the roster. No favorable subset is promoted to an independent claim.']
    (run.PUBLIC/'locality_seed.md').write_text('\n'.join(detail)+'\n')
    resources={}
    for phase in ('pilot','train','fit_replay','decide','evaluate','replay','replay_evaluate'):
        path=run.PRIVATE/(phase+'.log')
        if not path.exists():continue
        text=path.read_text();t=re.search(r'([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys',text)
        memory=re.search(r'(\d+)\s+maximum resident set size',text)
        resources[phase]=dict(process_complete=bool(t and memory),seconds=float(t[1]) if t else None,
            peak_RSS_bytes=int(memory[1]) if memory else None)
    run.base.immutable_json(run.PUBLIC/'resources.json',resources)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    policies=['cached_pointwise_joint','cached_query_joint','subset_pointwise_joint','subset_aggregate_joint','subset_pointwise_rank','subset_aggregate_rank']
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    for i,p in enumerate(policies):
        value=d['summary'][p]['all_gain_floor'];lo,hi=value['ci95'];point=value['point']
        ax.errorbar(point,i,xerr=[[point-lo],[hi-point]],fmt='o',capsize=4,color='#177c75' if 'aggregate' in p else '#60666b')
    ax.set_yticks(range(len(policies)),[p.replace('_',' ') for p in policies]);ax.set_ylim(len(policies)-.5,-.5)
    ax.axvline(0,color='#ad4444',linestyle='--');ax.set_xlabel('ADE gain over protected floor (%)')
    ax.set_title('Anchored subset supervision: accuracy is not a risk certificate')
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(run.PUBLIC/'objective_comparison.png',dpi=150,metadata={'Software':'M3W anchored subset experiment'});plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,metric in zip(axes,('anchor','auxiliary_pointwise','auxiliary_aggregate')):
        for arm,color in [('subset_pointwise','#60666b'),('subset_aggregate','#177c75')]:
            traces=[r['trace'] for r in training if r['arm']==arm];steps=[r['step'] for r in traces[0]]
            assert all([r['step'] for r in t]==steps for t in traces)
            values=np.array([[r['monitor'][metric]/t[0]['monitor'][metric] for r in t] for t in traces])
            lo,median,hi=np.quantile(values,[.25,.5,.75],axis=0)
            ax.plot(steps,median,label=arm.replace('subset_',''),color=color);ax.fill_between(steps,lo,hi,color=color,alpha=.15)
        ax.set_title(metric.replace('_',' '));ax.set_xlabel('Optimizer updates');ax.set_ylabel('Loss / initial loss');ax.legend(fontsize=8)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Fitting losses: median and quartiles across dependent heads, not CIs')
    fig.savefig(run.PUBLIC/'training_curves.png',dpi=150,metadata={'Software':'M3W anchored subset experiment'});plt.close(fig)
    print(json.dumps(dict(paired=d['paired'],gates=d['gates'],training=totals)))


if __name__=='__main__':main()
