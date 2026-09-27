"""Fixed-roster risk report; empty coverage and component ambiguity stay explicit."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_fixed_floor_excess as run
from scripts.report_m3w_fixed_floor_tail import ci


def render(d):
    lines=['# Fixed-Floor Signed-Excess Results','','## Material Passport','',
        'Fresh108 risk heads and216,000updates; cached_verified108 matched MSE controls, utility/floor and forecasting models.',
        'Twelve development-exposed localities, three forecaster seeds,3,000 locality-bootstrap draws. No new trajectory training or independent test.','',
        '## Prespecified Primary','',
        'Matched-count MSE minus new selected harm ratio (percentage points): **'+ci(d['paired']['mse_matched_count']['positive_harm_reduction_pp'])+'**.',
        'New ADE gain at equal current-query count (%): **'+ci(d['paired']['mse_matched_count']['ADE_gain_percent'])+'**.',
        'Counts match within locality/recording/frame including unknown-label rows, never across later queries.','',
        '| Policy | ADE / floor gain % | ADE / CV gain % | Hard / floor gain % | Easy / CV gain % | FDE / floor gain % | Intervention % | Selected harm % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for p,m in d['summary'].items():
        lines.append('| '+p+' | '+' | '.join(ci(m[k]) for k in ('all_gain_floor','all_gain_CV','hard_gain_floor','easy_gain_CV','FDE_gain_floor'))+' | '+ci(m['intervention_rate'],100)+' | '+ci(m['selected_positive_harm_ratio'],100)+' |')
    lines+=['','Undefined is not zero risk; the fixed roster is not dropped. Count-matched MSE is a ranking diagnostic, not a certified2% policy.','',
        '| New versus control | ADE gain % | Harm reduction pp | Intervention difference pp |','|---|---:|---:|---:|']
    for p,m in d['paired'].items(): lines.append('| '+p+' | '+' | '.join(ci(m[k]) for k in ('ADE_gain_percent','positive_harm_reduction_pp','intervention_difference_pp'))+' |')
    lines+=['','## Worst Views and Label Sensitivity','',
        '| Policy | Worst easy gain % | Risk violations /216 | Undefined views | P95 / floor | Complete ADE gain % | Partial ADE gain % | Unknown interventions/view |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for p,m in d['summary'].items():
        w=d['worst_views'][p]
        lines.append(f"| {p} | {w['worst_easy_gain_CV']:.4f} | {w['risk_violating_views']} | {w['undefined_risk_views']} | "+' | '.join(ci(m[k]) for k in ('p95_ratio_to_floor','complete_gain_floor','partial_gain_floor','unknown_interventions'))+' |')
    lines+=['','## Fit Versus Held Signed-Score Errors','',
        '| Objective | Fit all MSE | Held all MSE | Fit easy MSE | Held easy MSE | Selected predicted all excess | Selected observed all excess |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p,m in d['quality'].items():
        lines.append('| '+p+' | '+' | '.join(ci(m[k]) for k in ('all_fit_MSE','all_normalized_excess_MSE','easy_fit_MSE','easy_normalized_excess_MSE','all_selected_predicted_excess','all_selected_observed_excess'))+' |')
    lines+=['','Scores/errors use fitting-only mean-CV-cost normalization. Fitting values are repeated as descriptive paired context references, not held measurements or independent fit units.',
        'New output components are unidentified score bases. Do not interpret their ratio as calibrated reference/harm moments. A negative predicted excess is an empirical decision, not an upper confidence bound.','',
        '## Three Seeds','',
        '| Forecaster seed | New gain / floor % | MSE gain / floor % | New intervention % | New selected harm % |',
        '|---|---:|---:|---:|---:|']
    for seed,m in d['by_seed'].items():
        lines.append('| '+seed+' | '+ci(m['excess']['all_gain_floor'])+' | '+ci(m['mse']['all_gain_floor'])+' | '+ci(m['excess']['intervention_rate'],100)+' | '+ci(m['excess']['selected_positive_harm_ratio'],100)+' |')
    lines+=['','## Gates','',*[f'- {k}: {v}' for k,v in d['gates'].items()],'',
        'Per-locality values for every metric are in summary.json. Bootstrap12source means after averaging dependent seed/producer/fit views; overlapping windows are not independent.',
        'All intervals are unadjusted development uncertainty, not confirmation or simultaneous safety. No independent roles, deployment, Stage5C or SMC.',
        'Image-local detector silver; obs8/pred12 rawstride12, not historicalt50, metric, seconds, human gold, physical safety, true3D or foundation.']
    return '\n'.join(lines)+'\n'


def main():
    d=json.loads((run.PUBLIC/'summary.json').read_text()); (run.PUBLIC/'results.md').write_text(render(d))
    freeze=json.loads((run.PUBLIC/'decision_freeze.json').read_text()); traces=[]
    for ref in freeze['heads']:
        assert run.artifact(ROOT/ref['path'])==ref
        r=json.loads((ROOT/ref['path']).read_text()); traces.append(r['fit'])
    t=dict(new_heads=len(traces),updates=sum(x['step'] for x in traces),fit_seconds=sum(x['seconds'] for x in traces),
        unknown_supervised_draws=sum(x['unknown_rows_sampled'] for x in traces),
        declining_fixed_loss_heads=sum(x['trace'][-1]['fixed_training_excess_MSE']<x['trace'][0]['fixed_training_excess_MSE'] for x in traces),
        mean_initial_fixed_loss=float(np.mean([x['trace'][0]['fixed_training_excess_MSE'] for x in traces])),
        mean_final_fixed_loss=float(np.mean([x['trace'][-1]['fixed_training_excess_MSE'] for x in traces])),
        cached_controls=108,new_forecasters=0)
    run.immutable_json(run.PUBLIC/'training_summary.json',t)
    (run.PUBLIC/'training_report.md').write_text('# Training Completion\n\n## Material Passport\n\nFresh risk-head training, cached matched controls. Training loss is not downstream success.\n\n'+
        '\n'.join(f'- {k}: {v}' for k,v in t.items())+'\n\nFixed2,000updates, native CPU4, workers0, checkpoint500. Pilot included, no held checkpoint selection.\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['MSE','Direct excess','MSE: matched count']; policies=['mse','excess','mse_matched_count']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,key,mult,title in ((axes[0],'all_gain_floor',1,'Incremental ADE gain (%)'),(axes[1],'selected_positive_harm_ratio',100,'Selected positive harm (%)')):
        for i,p in enumerate(policies):
            m=d['summary'][p][key]
            if m['point'] is None:
                ax.text(.05,i,'undefined',transform=ax.get_yaxis_transform(),va='center'); continue
            pt=m['point']*mult; lo,hi=np.asarray(m['ci95'])*mult
            ax.errorbar(pt,i,xerr=[[pt-lo],[hi-pt]],fmt='o',capsize=3,color='#167d73' if p=='excess' else '#555555')
        ax.set_yticks(range(3),labels); ax.set_ylim(2.5,-.5); ax.set_title(title)
        ax.axvline(0 if key=='all_gain_floor' else 2,color='#ba4340',ls='--',lw=1); ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Fixed-floor signed loss: development evidence, not certified safety')
    fig.savefig(run.PUBLIC/'risk_gain_comparison.png',dpi=150,metadata={'Software':'M3W fixed-floor signed loss'})
    plt.close(fig); print(json.dumps(dict(training=t,gates=d['gates'],primary=d['paired']['mse_matched_count'])))


if __name__=='__main__': main()
