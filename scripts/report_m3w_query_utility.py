"""All frozen query-allocation arms, including risk failures and undefined views."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_query_utility as run
from scripts.report_m3w_fixed_floor_tail import ci


def main():
    d=json.loads((run.PUBLIC/'summary.json').read_text())
    lines=['# Frozen Query Utility Allocation Results','','## Material Passport','',
        'Fresh causal allocation and outcome readout; cached_verified estimators, source roles and forecasts. No new training.',
        'Twelve opened development localities, three forecasting seeds and 108 paired groups. Independent roles remain closed.',
        'Joint utility and independent admission have identical switch counts in each current query. Uniform admission is not rate matched; top-k is a risk-unconstrained diagnostic.','',
        '## Registered Primary','',
        'Joint utility versus independent ADE gain (%): **'+ci(d['paired']['independent']['ADE_gain_percent'])+'**.',
        'Predicted feasibility is not observed risk control. All safety gates below must be retained.','',
        '| Policy | ADE / floor gain % | Hard / floor gain % | Easy / CV gain % | FDE / floor gain % | Intervention % | Selected positive harm % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p,m in d['summary'].items():
        lines.append('| '+p+' | '+' | '.join(ci(m[k]) for k in ('all_gain_floor','hard_gain_floor','easy_gain_CV','FDE_gain_floor'))+' | '+ci(m['intervention_rate'],100)+' | '+ci(m['selected_positive_harm_ratio'],100)+' |')
    lines+=['','## Same-Count Contrasts','',
        '| Joint versus control | ADE gain % | Harm reduction pp | Count difference pp |','|---|---:|---:|---:|']
    for p,m in d['paired'].items():
        lines.append('| '+p+' | '+' | '.join(ci(m[k]) for k in ('ADE_gain_percent','positive_harm_reduction_pp','intervention_difference_pp'))+' |')
    lines+=['','## Worst Views, Missing Labels and Tail Error','',
        '| Policy | Worst easy gain % | Risk violations /216 | Undefined views | P95 / floor P95 | Unknown interventions /view | Complete ADE gain % | Partial ADE gain % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for p,m in d['summary'].items():
        w=d['worst_views'][p]
        lines.append(f"| {p} | {w['worst_easy_gain_CV']:.4f} | {w['risk_violating_views']} | {w['undefined_risk_views']} | "+' | '.join(ci(m[k]) for k in ('p95_ratio_to_floor','unknown_interventions','complete_gain_floor','partial_gain_floor'))+' |')
    lines+=['','## Locality and Seed Evidence','','| Locality | Joint versus independent ADE gain % |','|---|---:|']
    for site,value in d['paired']['independent']['ADE_gain_percent']['by_site'].items():
        lines.append(f'| {site} | {value:.6f} |')
    lines+=['','| Forecaster seed | Independent ADE / floor gain % | Joint ADE / floor gain % |','|---|---:|---:|']
    for seed,m in d['by_seed'].items():
        lines.append('| '+seed+' | '+ci(m['independent']['all_gain_floor'])+' | '+ci(m['joint_utility']['all_gain_floor'])+' |')
    lines+=['','## Solver and Gates','',*[f'- {k}: {v}' for k,v in d['solver'].items()],'',
        *[f'- {k}: {v}' for k,v in d['gates'].items()],'',
        'Expected utility totals use per-head fitting-only normalized scores over repeated query views, not a unique-population benefit estimate.',
        'The four neural components are signed-risk score bases, not separately identified calibrated moments. Aggregate predicted excess constraints can be satisfied while actual selected harm fails.',
        'There is no pairwise interaction penalty in this contrast. This is a joint budget allocation test, not proof of nonadditive interaction or full-visible-scene safety.',
        'Bootstrap uses 3,000 draws over 12 locality means, after averaging dependent producer/fit/seed views. No IID-window, multiple-comparison-adjusted or independent confirmation claim.',
        'Image-local detector silver; obs8/pred12 rawstride12. No metric, seconds, human-gold, physical-safety, true3D or foundation claims. Stage5C and SMC remain disabled.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.4),layout='constrained')
    policies=['independent','joint_utility','utility_topk','query_uniform']
    for i,p in enumerate(policies):
        v=d['summary'][p]['all_gain_floor'];lo,hi=v['ci95'];pt=v['point']
        ax.errorbar(pt,i,xerr=[[pt-lo],[hi-pt]],fmt='o',capsize=4,color='#197b75' if p=='joint_utility' else '#62666a')
    ax.set_yticks(range(4),['Independent','Joint utility','Top-k (no risk)','Whole-query uniform'])
    ax.set_ylim(3.5,-.5);ax.axvline(0,color='#af4242',linestyle='--')
    ax.set_xlabel('ADE gain over protected floor (%)');ax.set_title('Frozen allocation: mean gains do not certify risk')
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(run.PUBLIC/'allocation_comparison.png',dpi=150,metadata={'Software':'M3W query utility allocation'});plt.close(fig)
    print(json.dumps(dict(primary=d['paired']['independent'],gates=d['gates'],solver=d['solver'])))


if __name__=='__main__':main()
