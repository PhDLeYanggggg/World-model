"""All fixed allocation results, not a selected deployment winner."""
import csv
import json
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/easy_allocation_risk_scaled_v1'


def main():
    raw=(PUBLIC/'analysis.json').read_bytes();a=json.loads(raw);sha=hashlib.sha256(raw).hexdigest()
    for name in ('aggregate_replay.json','independent_verification.json'):
        r=json.loads((PUBLIC/name).read_text());assert r['all_checks_passed'] and r['analysis_sha256']==sha
    d=json.loads((PUBLIC/'decision_replay.json').read_text())
    assert d['all_checks_passed'] and d['decision_manifest_sha256']==a['decision_manifest_sha256']
    rows=[];sites=[]
    for key,r in a['summary'].items():
        worst=max(0.,max(-s['subsets']['positive_easy']['worst_scene_gain_percent'] for s in r['seeds'].values()))
        rows.append(dict(policy=key,ADE_gain_percent=r['ADE']['equal_scene_gain_percent'],
            ADE_CI_low=r['ADE']['scene_bootstrap_ci95'][0],ADE_CI_high=r['ADE']['scene_bootstrap_ci95'][1],
            FDE_gain_percent=r['FDE']['equal_scene_gain_percent'],hard_gain_percent=r['subsets']['hard']['equal_scene_gain_percent'],
            worst_site_seed_easy_degradation_percent=worst,
            switch_percent=100*sum(s['selected'] for s in r['seeds'].values())/(len(a['seeds'])*a['rows']),
            zero_CV_harmed=sum(s['zero_CV_harmed'] for s in r['seeds'].values()),
            unknown_selected=sum(s['selected_unknown'] for s in r['seeds'].values()),
            incomplete_selected=sum(s['selected_incomplete'] for s in r['seeds'].values())))
        for seed,s in r['seeds'].items():
            for subset,m in [('all',s['ADE']),*s['subsets'].items()]:
                for site,v in m['by_scene'].items():sites.append(dict(policy=key,seed=seed,subset=subset,site=site,**v))
    for name,values in [('results.csv',rows),('site_seed_results.csv',sites)]:
        with (PUBLIC/name).open('w',newline='') as f:
            fields=list(dict.fromkeys(k for v in values for k in v))
            w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(values)
    lines=['# Fixed Query Allocation Results','',
        'Development-only, native annotation pixels, obs8/pred12. Seed means are means of errors, not an ensemble.',
        'CI95: 3,000 paired physical-site resamples, only four exposed sites. Not independent confirmation.',
        'Zero-CV/unknown/incomplete counts sum query/seed instances. Every policy is retained.','',
        '| Policy | ADE gain % [CI95] | Hard gain % | Worst easy degradation % | Switch % | Zero-CV harmed |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['policy']} | {r['ADE_gain_percent']:.4f} [{r['ADE_CI_low']:.4f}, {r['ADE_CI_high']:.4f}] | {r['hard_gain_percent']:.4f} | {r['worst_site_seed_easy_degradation_percent']:.4f} | {r['switch_percent']:.3f} | {r['zero_CV_harmed']} |")
    lines+=['','## Paired Contrasts','','Nominal development contrasts, no multiple-comparison or population claim.','',
        '| Predictor / contrast | Difference pp | CI95 pp |','|---|---:|---|']
    for action,contrasts in a['contrasts'].items():
        for key,r in contrasts.items():lines.append(f"| {action} / {key} | {r['all']['mean_gain_difference_pp']:.6f} | {r['all']['ci95_pp']} |")
    lines+=['','## Query Geometry and Numerical Support','',
        '| Predictor | Query/seed instances | Unmatched | Selected solver failures | Population / unary / joint failures | Nonadditive queries | Changed joint/unary queries |',
        '|---|---:|---:|---:|---|---:|---:|']
    for action,r in a['allocation'].items():
        lines.append(f"| {action} | {r['query_seed_instances']} | {r['unmatched']} | {r['failed_selected']} | {r['failed_population']} / {r['failed_unary']} / {r['failed_joint']} | {r['nonadditive_queries']} | {r['changed_joint_unary_queries']} |")
    (PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    compact=dict(result_source=a['result_source'],analysis_sha256=sha,rows=a['rows'],sites=a['sites'],seeds=a['seeds'],
        new_training=False,query_action_seed_instances=sum(r['query_seed_instances'] for r in a['allocation'].values()),
        policies=rows,contrasts=a['contrasts'],allocation=a['allocation'],
        independent_calibration=False,independent_confirmation=False,deployment=False,stage5c_executed=False,smc_enabled=False)
    (PUBLIC/'compact_results.json').write_text(json.dumps(compact,indent=2)+'\n')
    figure(rows)
    print(json.dumps(dict(analysis_sha256=sha,policies=len(rows),site_seed_rows=len(sites),all_verified=True)))


def figure(rows):
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='easy-allocation-v1'
    import matplotlib.pyplot as plt
    actions=['damped_velocity_005','transformer','eqmotion']
    policies=['strict_stop','pointwise','aggregate_selected','aggregate_population','aggregate_joint']
    colors=['#697882','#3884bd','#46a085','#ac782c','#aa5470']
    lookup={r['policy']:r for r in rows};fig,axes=plt.subplots(2,3,figsize=(13,6),layout='constrained')
    for j,title in enumerate(['Simple damping','Transformer','EqMotion']):
        v=[lookup[actions[j]+'__'+p] for p in policies]
        axes[0,j].bar(range(5),[r['ADE_gain_percent'] for r in v],color=colors)
        axes[0,j].vlines(range(5),[r['ADE_CI_low'] for r in v],[r['ADE_CI_high'] for r in v],color='#222222',lw=1.)
        axes[0,j].set_title(title);axes[0,j].set_xticks([])
        easy=[r['worst_site_seed_easy_degradation_percent'] for r in v]
        b=axes[1,j].bar(range(5),easy,color=colors)
        axes[1,j].bar_label(b,labels=[f'{x:.2f}' for x in easy],padding=3,fontsize=9,
                            bbox=dict(facecolor='white',edgecolor='none',pad=.5))
        axes[1,j].axhline(2,color='#b13d42',ls='--',lw=1.)
        axes[1,j].set_ylim(0,max(3,max(easy)*1.25))
        axes[1,j].set_xticks(range(5),['Strict','Point','Selected','Population','Joint'],rotation=20)
        for i in range(2):
            axes[i,j].spines[['top','right']].set_visible(False);axes[i,j].set_axisbelow(True)
            axes[i,j].grid(axis='y',alpha=.2)
    axes[0,0].set_ylabel('Equal-site ADE gain over CV (%)')
    axes[1,0].set_ylabel('Worst site/seed easy degradation (%)')
    fig.suptitle('Frozen forecasts and risk estimates: fixed allocation controls\nFour development-exposed SDD sites; empirical harm is not certified risk',fontsize=12)
    path=PUBLIC/'allocation_tradeoff.svg';fig.savefig(path,metadata={'Date':None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    fig.savefig(ROOT/'data/stage_cvpr2027_experiments/easy_allocation_risk_scaled_v1/preview.png',dpi=130)
    plt.close(fig)


if __name__=='__main__':main()
