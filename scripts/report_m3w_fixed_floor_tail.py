"""Matched loss/readout reports, including count-controlled negative evidence."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_fixed_floor_tail as run


def ci(r,mult=1):
    if r['point'] is None or r['ci95'] is None: return 'undefined'
    return f"{mult*r['point']:.4f} [{mult*r['ci95'][0]:.4f}, {mult*r['ci95'][1]:.4f}]"


def report(d):
    rows=['# Fixed-Floor Tail-Weighted Risk Results','','## Material Passport','',
        'Fresh_run:216 Torch risk heads,432,000 updates, frozen inference/decisions, held readout and3,000 locality-bootstrap draws.',
        'Cached_verified: forecasting models, protected damping, ridge utility and input preprocessing.',
        'Twelve development-exposed source localities. No independent confirmation, new forecaster or deployment.','',
        '## Prespecified Equal-Count Primary','',
        'MSE matched-count minus tail selected positive-harm ratio (percentage points): **'+
        ci(d['paired']['mse_matched_count']['positive_harm_reduction_pp'])+'**. Positive means the tail model reduces harm.',
        'Matched-count tail ADE gain (%): **'+ci(d['paired']['mse_matched_count']['ADE_gain_percent'])+'**.',
        'Counts match within the same locality/recording/frame, including unknown-label rows; no later-query allocation.','',
        '## All Actions','',
        '| Action | ADE gain over floor % | ADE gain over CV % | Hard gain over floor % | Easy gain over CV % | Endpoint FDE gain over floor % | Intervention % | Positive harm ratio % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for p,m in d['summary'].items():
        rows.append('| '+p+' | '+' | '.join(ci(m[k]) for k in ('all_gain_floor','all_gain_CV','hard_gain_floor','easy_gain_CV','FDE_gain_floor'))+' | '+ci(m['intervention_rate'],100)+' | '+ci(m['selected_positive_harm_ratio'],100)+' |')
    rows+=['','`mse_matched_count` is diagnostic ranking at tail-model counts, not a claim that the MSE predicted budget holds.',
        'Floor intervention is0 by definition of the incremental comparison. Undefined risk is not zero risk.','',
        '## Paired Loss Controls','',
        '| Tail versus control | ADE gain % | Positive harm reduction pp | Intervention difference pp |',
        '|---|---:|---:|---:|']
    for p,m in d['paired'].items():
        rows.append('| '+p+' | '+' | '.join(ci(m[k]) for k in ('ADE_gain_percent','positive_harm_reduction_pp','intervention_difference_pp'))+' |')
    rows+=['','## Worst Views and Label Sensitivity','',
        '| Action | Worst easy gain % | Risk-violating views | Undefined-risk views | Zero-CV harmed views | Complete-label ADE gain % | Partial-label ADE gain % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p,m in d['summary'].items():
        w=d['worst_views'][p]
        rows.append(f"| {p} | {w['worst_easy_gain_CV']:.4f} | {w['risk_violating_views']} | {w['undefined_risk_views']} | {w['zero_CV_harmed_views']} | {ci(m['complete_gain_floor'])} | {ci(m['partial_gain_floor'])} |")
    rows+=['','| Action | P95 error / floor P95 | Unknown-label interventions per dependent view |',
           '|---|---:|---:|']
    for p,m in d['summary'].items():
        rows.append('| '+p+' | '+ci(m['p95_ratio_to_floor'])+' | '+ci(m['unknown_interventions'])+' |')
    rows+=['','Each action has216 dependent held-locality views, not216 independent scenes.',
        'Positive harm is distinct from net ADE and easy degradation. All primary rosters remain fixed.','',
        '## Moment Diagnostics','',
        '| Loss | Predicted selected harm % | Actual selected harm % | Predicted/actual reference | Zero predicted harm fraction | Harm MSE | Easy harm MSE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p,m in d['quality'].items():
        rows.append('| '+p+' | '+ci(m['predicted_selected_harm_ratio'],100)+' | '+ci(m['actual_selected_harm_ratio'],100)+' | '+
            ' | '.join(ci(m[k]) for k in ('predicted_over_actual_reference','zero_predicted_harm_fraction','harm_MSE','easy_harm_MSE'))+' |')
    rows+=['','Moment MSE is in squared image-local ADE units. Weighted loss estimates tilted harm scores, not calibrated means.','',
        '## Three Forecaster Seeds','',
        '| Seed | MSE gain over floor % | Tail gain over floor % | MSE matched-count gain % | Tail harm ratio % |',
        '|---|---:|---:|---:|---:|']
    for s,m in d['by_seed'].items():
        rows.append('| '+s+' | '+' | '.join(ci(m[p]['all_gain_floor']) for p in ('mse','tail4','mse_matched_count'))+' | '+ci(m['tail4']['selected_positive_harm_ratio'],100)+' |')
    rows+=['','## Gates','',*[f'- {k}: {v}' for k,v in d['gates'].items()],'',
        'Source and seed breakdowns are retained for every metric in summary.json. Bootstrap unit:12 locality means after averaging dependent seed/producer/fit-half views. Overlapping windows are not independent.',
        'All intervals are unadjusted development uncertainty, not a safety certificate or final independent test.',
        'Image-local detector silver, obs8/pred12 rawstride12; not historicalt50, metric, seconds, human gold, physical safety, true3D or foundation.',
        'Stage5C and SMC remain disabled. The research goal is not complete.']
    return '\n'.join(rows)+'\n'


def training_report():
    freeze=json.loads((run.PUBLIC/'decision_freeze.json').read_text()); heads=[]
    for ref in freeze['heads']:
        assert run.artifact(ROOT/ref['path'])==ref
        r=json.loads((ROOT/ref['path']).read_text()); f=r['fit']
        heads.append(dict(arm=r['arm'],seconds=f['seconds'],updates=f['step'],draws=f['total_draws'],
            unknown_draws=f['unknown_rows_sampled'],first_MSE=f['trace'][0]['fixed_training_MSE'],
            final_MSE=f['trace'][-1]['fixed_training_MSE'],first_tail_loss=f['trace'][0]['fixed_training_tail_loss'],
            final_tail_loss=f['trace'][-1]['fixed_training_tail_loss']))
    import numpy as np
    summary={a:dict(heads=sum(h['arm']==a for h in heads),
        updates=sum(h['updates'] for h in heads if h['arm']==a),fit_seconds=sum(h['seconds'] for h in heads if h['arm']==a),
        unknown_draws=sum(h['unknown_draws'] for h in heads if h['arm']==a),
        MSE_decreasing_heads=sum(h['final_MSE']<h['first_MSE'] for h in heads if h['arm']==a),
        tail_loss_decreasing_heads=sum(h['final_tail_loss']<h['first_tail_loss'] for h in heads if h['arm']==a),
        mean_first_MSE=float(np.mean([h['first_MSE'] for h in heads if h['arm']==a])),
        mean_final_MSE=float(np.mean([h['final_MSE'] for h in heads if h['arm']==a]))) for a in ('mse','tail4')}
    run.immutable_json(run.PUBLIC/'training_summary.json',summary)
    lines=['# Training Completion','','## Material Passport','',
        'Fresh216 risk-head fits. Fixed training monitors are not validation loss or downstream evidence.','',
        '| Loss | Heads | Updates | Cumulative fit seconds | Unknown draws | Heads with lower fixed MSE | Heads with lower fixed tail loss | Mean initial/final MSE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for a,m in summary.items(): lines.append(f"| {a} | {m['heads']} | {m['updates']} | {m['fit_seconds']:.2f} | {m['unknown_draws']} | {m['MSE_decreasing_heads']} | {m['tail_loss_decreasing_heads']} | {m['mean_first_MSE']:.6f} / {m['mean_final_MSE']:.6f} |")
    lines+=['','Paired heads use identical initial models, fitting features, targets, normalizers, draws and RNG states.',
        'Only loss weights differ. The first100-update pilot is included in the2,000-update budget.',
        'Forecasters, floor producer chain and ridge utility remain frozen. No independent-source tuning.']
    (run.PUBLIC/'training_report.md').write_text('\n'.join(lines)+'\n')


def main():
    training_report(); d=json.loads((run.PUBLIC/'summary.json').read_text())
    (run.PUBLIC/'results.md').write_text(report(d))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    policies=['ridge','mse','tail4','mse_matched_count']; labels=['Frozen ridge','Moment MSE','Tail weighted','MSE: tail-matched count']
    for ax,key,mult,title in ((axes[0],'all_gain_floor',1,'Incremental ADE gain (%)'),(axes[1],'selected_positive_harm_ratio',100,'Selected positive harm (%)')):
        for i,p in enumerate(policies):
            m=d['summary'][p][key]
            if m['point'] is None:
                ax.text(.05,i,'undefined',transform=ax.get_yaxis_transform(),va='center')
                continue
            pt=m['point']*mult; lo,hi=[x*mult for x in m['ci95']]
            ax.errorbar(pt,i,xerr=[[pt-lo],[hi-pt]],fmt='o',capsize=3,color='#167d73' if p=='tail4' else '#555555')
        ax.set_yticks(range(4),labels); ax.invert_yaxis(); ax.set_title(title)
        ax.axvline(0 if key=='all_gain_floor' else 2,color='#ba4340',lw=1,linestyle='--')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Fixed-floor loss contrast: development evidence, not certified safety')
    fig.savefig(run.PUBLIC/'risk_gain_comparison.png',dpi=150,metadata={'Software':'M3W matched risk loss report'})
    plt.close(fig)
    print(json.dumps(dict(primary=d['paired']['mse_matched_count'],gates=d['gates'])))


if __name__=='__main__': main()
