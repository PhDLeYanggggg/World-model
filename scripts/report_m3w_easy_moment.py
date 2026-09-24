"""Lightweight tables from verified fixed easy-moment analyses."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/easy_moment_v1'


def main():
    raw=(PUBLIC/'analysis.json').read_bytes();a=json.loads(raw);sha=hashlib.sha256(raw).hexdigest()
    for name in ('verification.json','independent_verification.json'):
        r=json.loads((PUBLIC/name).read_text())
        assert r['all_checks_passed'] and r['analysis_sha256']==sha
    rows=[]
    for name,s in a['summary'].items():
        worst=max(0.,max(-r['subsets']['positive_easy']['worst_scene_gain_percent'] for r in s['seeds'].values()))
        rows.append(dict(policy=name,ADE_gain_percent=s['ADE']['equal_scene_gain_percent'],
            ADE_CI_low=s['ADE']['scene_bootstrap_ci95'][0],ADE_CI_high=s['ADE']['scene_bootstrap_ci95'][1],
            FDE_gain_percent=s['FDE']['equal_scene_gain_percent'],hard_gain_percent=s['subsets']['hard']['equal_scene_gain_percent'],
            worst_site_seed_easy_degradation_percent=worst,
            mean_switch_rate_percent=100*sum(r['selected'] for r in s['seeds'].values())/(3*a['rows']),
            zero_CV_harmed_total=sum(r['zero_CV_harmed'] for r in s['seeds'].values())))
    with (PUBLIC/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Conditional Easy-Moment Results','','Development only; equal-site relative ADE gain over CV. Three seeds, four exposed sites.',
        'No metric/seconds, independent safety, confirmation or deployment claim. All fixed policies retained.','',
        '| Policy | ADE gain % [CI95] | Hard gain % | Worst easy degradation % | Switch % | Zero-CV harmed, summed seeds |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['policy']} | {r['ADE_gain_percent']:.4f} [{r['ADE_CI_low']:.4f}, {r['ADE_CI_high']:.4f}] | {r['hard_gain_percent']:.4f} | {r['worst_site_seed_easy_degradation_percent']:.4f} | {r['mean_switch_rate_percent']:.3f} | {r['zero_CV_harmed_total']} |")
    lines+=['','## Paired Contrasts','','Nominal exploratory physical-site bootstrap intervals; no multiple-comparison claim.','',
        '| Contrast | ADE difference pp | CI95 |','|---|---:|---|']
    for key,r in a['contrasts'].items():lines.append(f"| {key} | {r['all']['mean_gain_difference_pp']:.4f} | {r['all']['ci95_pp']} |")
    (PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    loss=['# Fitting Losses','','Four bounded moment targets; draw-weighted fitting MSE, not validation loss or forecast ADE.',
        'No early stopping or outer-outcome selection. All 36 fits use 128 trees.','',
        '| View | Action | First MSE (16 trees) | Final MSE (128 trees) | Fit seconds |','|---|---|---:|---:|---:|']
    for r in a['fits']:
        f=r['fit'];assert f['complete'] and f['trees']==128 and f['sampled_rows']==768000
        loss.append(f"| {r['view']} | {r['action']} | {f['trace'][0]['fitting_mean_mse']:.8f} | {f['trace'][-1]['fitting_mean_mse']:.8f} | {f['seconds']:.3f} |")
    seconds=sum(r['fit']['seconds'] for r in a['fits'])
    loss+=['',f'Summed fitting-loop time: {seconds:.3f} seconds, not total wall time. Per-target traces are in analysis.json.']
    (PUBLIC/'training_losses.md').write_text('\n'.join(loss)+'\n')
    compact=dict(analysis_sha256=sha,result_source=a['result_source'],rows=a['rows'],complete_rows=a['complete_rows'],
        new_risk_forests=36,new_forecasters=0,seed_count=3,physical_site_count=4,fitting_seconds=seconds,
        fitting_seconds_are_not_wall_time=True,policies=rows,contrasts=a['contrasts'],
        independent_confirmation=False,deployment=False,stage5c_executed=False,smc_enabled=False)
    (PUBLIC/'compact_results.json').write_text(json.dumps(compact,indent=2)+'\n')
    detailed_tables(a)
    rejection_diagnostics(a)
    figure(rows)
    print(json.dumps(compact,indent=2))


def detailed_tables(a):
    site_rows=[]
    for policy,s in a['summary'].items():
        for seed,r in s['seeds'].items():
            for subset,metrics in [('all',r['ADE']),*r['subsets'].items()]:
                for site,m in metrics['by_scene'].items():
                    site_rows.append(dict(policy=policy,seed=seed,subset=subset,site=site,**m))
    fields=list(dict.fromkeys(k for r in site_rows for k in r))
    with (PUBLIC/'site_seed_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(site_rows)
    lines=['# Easy-Moment Reliability on Complete Labels','',
        'Post-fit descriptive diagnostics, not threshold selection or calibration. Each row uses',
        'only complete-path outcome labels; inference and intervention never use that mask.',
        'Predicted and observed numerators here are positive easy-weighted harm sums.',
        'The observed ratio is not net easy degradation: it does not subtract improvements.',
        'Small/zero denominators and missing futures preclude a population safety claim.','',
        '| View/action | Policy | Complete interventions | Predicted positive harm / easy denominator (%) | Observed positive harm / easy denominator (%) |',
        '|---|---|---:|---:|---:|']
    quality=[]
    for key,r in a['conditional_quality'].items():
        for policy,s in r['selected'].items():
            ratios=[]
            for kind in ('predicted','actual'):
                denominator=s[kind+'_easy_denominator_sum']
                ratios.append(100*s[kind+'_easy_harm_sum']/denominator if denominator>0 else None)
            quality.append(dict(view_action=key,policy=policy,**s,
                predicted_positive_easy_harm_ratio_percent=ratios[0],
                observed_positive_easy_harm_ratio_percent=ratios[1]))
            if policy in ('strict_stop','joint_easy_moment','product_easy_marginals'):
                display=['undefined' if v is None else f'{v:.4f}' for v in ratios]
                lines.append(f"| {key} | {policy} | {s['count']} | {display[0]} | {display[1]} |")
    with (PUBLIC/'conditional_reliability.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(quality[0]));w.writeheader();w.writerows(quality)
    (PUBLIC/'conditional_reliability.md').write_text('\n'.join(lines)+'\n')


def rejection_diagnostics(a):
    import numpy as np
    cache=ROOT/'data/stage_cvpr2027_experiments/easy_moment_v1'
    records=[]
    for item in a['decision_archives']:
        path=ROOT/item['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        key=item['view'];seed=int(key.rsplit('seed',1)[1])
        with np.load(path,allow_pickle=False) as z:
            ids=z['ids']
            for action in ('damped_velocity_005','transformer','eqmotion'):
                q=z[action+'__moments'];d=z[action+'__distance']
                net=z[action+'__net_stop'];joint=z[action+'__joint_easy_moment']
                receipt=next(r for r in a['fits'] if r['view']==key and r['action']==action)
                cutoff=receipt['identity']['easy_cut'];denominator=q[:,1]*cutoff
                ratio=np.divide(q[:,0]*d,denominator,out=np.full(len(ids),np.nan),where=denominator>0)
                op=cache/'outcomes'/f'{action}_seed{seed}.npz'
                binding=next(r for r in a['outcome_archives'] if ROOT/r['path']==op)
                assert hashlib.sha256(op.read_bytes()).hexdigest()==binding['sha256']
                with np.load(op,allow_pickle=False) as o:
                    gain=o['cv'][ids]-o['candidate_ade'][ids]
                    hard=o['hard'][ids];complete=o['complete'][ids]
                useful=net&hard&complete&(gain>0)
                r=ratio[net&np.isfinite(ratio)]
                records.append(dict(view=key,action=action,net_count=int(net.sum()),
                    joint_count=int(joint.sum()),blocked_net_count=int((net&~joint).sum()),
                    predicted_zero_easy_denominator=int((net&(denominator==0)).sum()),
                    net_joint_risk_ratio_median=float(np.median(r)) if len(r) else None,
                    net_joint_risk_ratio_p10=float(np.quantile(r,.1)) if len(r) else None,
                    net_joint_exceeds_product_count=int((net&(q[:,0]>q[:,2]*q[:,3])).sum()),
                    complete_net_beneficial_hard=int(useful.sum()),
                    complete_joint_beneficial_hard=int((useful&joint).sum()),
                    complete_net_beneficial_hard_gain_sum=float(gain[useful].sum()),
                    complete_joint_beneficial_hard_gain_sum=float(gain[useful&joint].sum())))
    with (PUBLIC/'rejection_diagnostics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    summary={}
    for action in ('damped_velocity_005','transformer','eqmotion'):
        subset=[r for r in records if r['action']==action]
        summary[action]={k:sum(r[k] for r in subset) for k in (
            'net_count','joint_count','blocked_net_count','predicted_zero_easy_denominator',
            'net_joint_exceeds_product_count','complete_net_beneficial_hard',
            'complete_joint_beneficial_hard','complete_net_beneficial_hard_gain_sum',
            'complete_joint_beneficial_hard_gain_sum')}
    (PUBLIC/'rejection_diagnostics.json').write_text(json.dumps(dict(
        scope='post_readout_descriptive_diagnosis_not_selection',
        counts_unit='query_seed_instances_not_independent_samples',
        future_labels_used_only_for_diagnostic_hard_gain=True,
        policy_changed=False,summary=summary),indent=2)+'\n')


def figure(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    actions=['damped_velocity_005','transformer','eqmotion']
    policies=['net_stop','strict_stop','joint_easy_moment','product_easy_marginals']
    colors=['#7b8186','#377db5','#248a71','#bf7540']
    fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
    lookup={r['policy']:r for r in rows}
    for col,(action,title) in enumerate(zip(actions,['Simple damping','Transformer','EqMotion'])):
        values=[lookup[action+'__'+p] for p in policies]
        gains=np.array([r['ADE_gain_percent'] for r in values])
        lo=np.array([r['ADE_CI_low'] for r in values]);hi=np.array([r['ADE_CI_high'] for r in values])
        axes[0,col].bar(range(4),gains,color=colors,width=.65)
        axes[0,col].vlines(range(4),lo,hi,color='#222222',lw=1.2)
        axes[0,col].plot(range(4),gains,'o',color='#222222',ms=3)
        axes[0,col].axhline(0,color='#666666',lw=.7)
        axes[0,col].set_title(title,fontsize=12)
        axes[0,col].set_xticks([])
        easy=[r['worst_site_seed_easy_degradation_percent'] for r in values]
        bars=axes[1,col].bar(range(4),easy,color=colors,width=.65)
        axes[1,col].axhline(2,color='#aa3333',ls='--',lw=1.2)
        axes[1,col].set_ylim(0,max(3,max(easy)*1.22))
        axes[1,col].bar_label(bars,labels=[f'{x:.2f}' for x in easy],padding=3,fontsize=9,
                            bbox=dict(facecolor='white',edgecolor='none',pad=.5))
        axes[1,col].set_xticks(range(4),['Net','Strict','Joint','Product'])
        for row in range(2):
            axes[row,col].spines[['top','right']].set_visible(False)
            axes[row,col].set_axisbelow(True)
            axes[row,col].grid(axis='y',alpha=.18)
    axes[0,0].set_ylabel('Equal-site ADE gain over CV (%)')
    axes[1,0].set_ylabel('Worst-site/seed easy degradation (%)')
    fig.suptitle('Fixed policies on four development-exposed SDD sites\nTop: paired-site CI95; bottom: observed easy harm, not certified risk',fontsize=12)
    fig.savefig(PUBLIC/'fixed_policy_tradeoff.svg',metadata={'Date':None})
    preview=ROOT/'data/stage_cvpr2027_experiments/easy_moment_v1/tradeoff_preview.png'
    fig.savefig(preview,dpi=130)
    plt.close(fig)


if __name__=='__main__':main()
