"""Report both registered query-risk objectives without selecting a winner post hoc."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_query_excess_refit as run
from scripts.report_m3w_fixed_floor_tail import ci


def main():
    d=json.loads((run.PUBLIC/'summary.json').read_text())
    freeze=json.loads((run.PUBLIC/'training_freeze.json').read_text());training=[]
    for ref in freeze['groups']:
        assert run.base.artifact(ROOT/ref['path'])==ref
        doc=json.loads((ROOT/ref['path']).read_text())
        for arm,v in doc['fits'].items():
            training.append(dict(group=Path(ref['path']).parent.name,arm=arm,**v))
    totals={a:dict(heads=sum(r['arm']==a for r in training),updates=sum(r['step'] for r in training if r['arm']==a),
        training_queries=sum(r['supervised_queries'] for r in training if r['arm']==a),
        singleton_queries=sum(r['singleton_queries'] for r in training if r['arm']==a),
        loss_declined_heads=sum(r['trace'][-1]['monitor'][a]<r['trace'][0]['monitor'][a] for r in training if r['arm']==a),
        unknown_draws=sum(r['unknown_rows_sampled'] for r in training if r['arm']==a)) for a in ('pointwise','query')}
    run.base.immutable_json(run.PUBLIC/'training_summary.json',totals)
    lines=['# Matched Query-Excess Training Results','','## Material Passport','',
        'Fresh paired risk-head training and outcome evaluation; cached_verified forecasters, floor and utility.',
        '108 paired groups, 216 new heads, 432,000 updates. Twelve opened development localities; three forecasting seeds.',
        'Independent selection/calibration/confirmation remain closed. No outcome-selected checkpoints or thresholds.','',
        '## Registered Primary','',
        'Query versus pointwise ranking uses the same frozen-parent count in every current query. These rank arms are diagnostics, not risk-certified policies.',
        'Selected positive-harm reduction (percentage points): **'+ci(d['paired']['query_rank_vs_pointwise_rank']['harm_reduction_pp'])+'**.',
        'Equal-count ADE gain (%): **'+ci(d['paired']['query_rank_vs_pointwise_rank']['ADE_gain_percent'])+'**.','',
        '| Policy | ADE / floor gain % | Hard / floor gain % | Easy / CV gain % | FDE / floor gain % | Intervention % | Selected harm % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p,m in d['summary'].items():
        lines.append('| '+p+' | '+' | '.join(ci(m[k]) for k in ('all_gain_floor','hard_gain_floor','easy_gain_CV','FDE_gain_floor'))+' | '+ci(m['intervention_rate'],100)+' | '+ci(m['selected_positive_harm_ratio'],100)+' |')
    lines+=['','## Paired Contrasts','','| Contrast | ADE gain % | Harm reduction pp | Intervention difference pp |','|---|---:|---:|---:|']
    for p,m in d['paired'].items():
        lines.append('| '+p+' | '+' | '.join(ci(m[k]) for k in ('ADE_gain_percent','harm_reduction_pp','intervention_difference_pp'))+' |')
    lines+=['','The joint-policy contrasts use the same nominal predicted budget, not matched intervention counts. Only the registered ranking contrast is count matched.','',
        '## Risk, Missing Labels and Tail','','| Policy | Violating views /216 | Undefined views | Worst easy gain % | P95 ratio / floor | Unknown actions per view |',
        '|---|---:|---:|---:|---:|---:|']
    for p,m in d['summary'].items():
        w=d['worst_views'][p]
        lines.append(f"| {p} | {w['risk_violating_views']} | {w['undefined_risk_views']} | {w['worst_easy_gain_CV']:.6f} | "+ci(m['p95_ratio_to_floor'])+' | '+ci(m['unknown_interventions'])+' |')
    lines+=['','## Held Query Prediction','','| Objective | Pointwise all MSE | Pointwise easy MSE | Query all MSE | Query easy MSE | Singleton fraction |','|---|---:|---:|---:|---:|---:|']
    for p,m in d['quality'].items():
        lines.append('| '+p+' | '+' | '.join(ci(m[k]) for k in ('pointwise_all_MSE','pointwise_easy_MSE','query_all_MSE','query_easy_MSE','singleton_fraction'))+' |')
    lines+=['','Both objectives are evaluated on both losses. Aggregate MSE is algebraically no larger than individual MSE on a fixed model; that inequality is not a learned improvement. Compare models within the same metric.','',
        '## Training and Gates','',*[f'- {a}: {v}' for a,v in totals.items()],'',*[f'- {k}: {v}' for k,v in d['gates'].items()],
        '','The four outputs are signed-risk score bases, not identified calibrated moments. Query means are supervised on known labels only, while all causal rows remain in inference.',
        'Unknown outcomes stay unknown; incomplete fixed-roster risk summaries do not become zero risk.',
        'Three thousand locality-bootstrap draws follow averaging of dependent producer/fit/seed views. No IID-window or independent-confirmation claim.',
        'These are retained query cohorts, not a completeness guarantee for all visible agents. No pairwise collision term was learned.',
        'Image-local detector silver, obs8/pred12 raw-frame stride12. No metric, seconds, human-gold, physical-safety, true3D or foundation claim. No Stage5C or SMC.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    detail=['# Locality and Seed Breakdown','','| Locality | Query rank vs pointwise ADE gain % | Query joint vs pointwise ADE gain % |','|---|---:|---:|']
    for s,v in d['paired']['query_rank_vs_pointwise_rank']['ADE_gain_percent']['by_site'].items():
        q=d['paired']['query_joint_vs_pointwise_joint']['ADE_gain_percent']['by_site'][s]
        detail.append(f'| {s} | {v:.6f} | {q:.6f} |')
    detail+=['','| Seed | Pointwise joint ADE/floor % | Query joint ADE/floor % |','|---|---:|---:|']
    for seed,m in d['by_seed'].items():detail.append('| '+seed+' | '+ci(m['pointwise_joint']['all_gain_floor'])+' | '+ci(m['query_joint']['all_gain_floor'])+' |')
    detail+=['','All locality and seed summaries remain development evidence. No favorable subset is removed.']
    (run.PUBLIC/'locality_seed.md').write_text('\n'.join(detail)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    for i,p in enumerate(('parent_joint','pointwise_joint','query_joint','pointwise_rank','query_rank')):
        v=d['summary'][p]['all_gain_floor'];lo,hi=v['ci95'];pt=v['point']
        ax.errorbar(pt,i,xerr=[[pt-lo],[hi-pt]],fmt='o',capsize=4,color='#197b75' if p.startswith('query') else '#62666a')
    ax.set_yticks(range(5),['Frozen parent joint','Pointwise-trained joint','Query-trained joint','Pointwise rank (matched count)','Query rank (matched count)'])
    ax.set_ylim(4.5,-.5);ax.axvline(0,color='#af4242',linestyle='--')
    ax.set_xlabel('ADE gain over protected floor (%)');ax.set_title('Matched training objectives: accuracy and risk are distinct')
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(run.PUBLIC/'objective_comparison.png',dpi=150,metadata={'Software':'M3W matched query-risk experiment'});plt.close(fig)
    print(json.dumps(dict(primary=d['paired']['query_rank_vs_pointwise_rank'],gates=d['gates'],training=totals)))


if __name__=='__main__':main()
