"""Reproducible public aggregates for the fixed-producer incremental probe."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_fixed_floor_probe as run


def number(r):
    if r['point'] is None or r['ci95'] is None: return 'undefined'
    return f"{r['point']:.4f} [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]"


def report(d):
    rows=['# Fixed-Producer Incremental Probe Results','',
        'Fresh_run:108 joint ridge fits,216 five-output linear heads, held scoring and3,000 locality-bootstrap draws.',
        'Cached_verified: nine repaired forecasters and all upstream damping-floor producers.',
        'No new neural forecasters, independent confirmation or deployment. Twelve already-opened development localities.', '',
        '## Prespecified Primary Contrast','',
        'Floor-target safe probe versus matched CV-target safe probe ADE gain (%): **'+number(d['primary'])+'**.','',
        'This tests the target reference with the SAME frozen floor on probe fitting and readout sources.',
        'Four sources fit the forecaster, four fit the floor, two fit each probe and two evaluate it.', '',
        '## Every Prespecified Action','',
        '| Action | ADE gain over floor % | ADE gain over CV % | Easy gain over CV % | Hard gain over floor % | FDE gain over floor % | Intervention fraction |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p,m in d['summary'].items():
        rows.append('| '+p+' | '+' | '.join(number(m[k]) for k in ('all_gain_floor','all_gain_CV','easy_gain_CV','hard_gain_floor','FDE_gain_floor','intervention_rate'))+' |')
    rows+=['','Floor is the comparator, so its intervention fraction here is0. Probe actions switch from that floor',
        'to neural. Oracle uses future ADE only for diagnosis and carries the chosen trajectory into FDE.',
        'Intervention includes unknown-label rows; their outcomes remain undefined rather than zero.','',
        '## Safety and Label Sensitivity','',
        '| Action | Worst held-view easy gain over CV % | Zero-CV harmed views | Incremental risk-violating views | Complete-label gain over floor % | Partial-label gain over floor % |',
        '|---|---:|---:|---:|---:|---:|']
    for p,m in d['summary'].items():
        w=d['worst_views'][p]
        rows.append(f"| {p} | {w['worst_easy_gain_CV']:.4f} | {w['zero_reference_harmed_views']} | {w['risk_violating_views']} | {number(m['complete_gain_floor'])} | {number(m['partial_gain_floor'])} |")
    rows+=['','Risk is selected positive incremental harm divided by selected floor error, NOT net degradation.',
        'A null selected ratio means no defined selected reference mass, not certified safety.',
        'There are216 dependent held-locality views per action. Overlapping windows are not independent samples.','',
        '## Exact Default-Action Decomposition','',
        'All quantities below are percentage points of each locality floor error, then equally averaged.',
        'Original net gain = captured benefit - selected harm - CV fallback regression + CV fallback relief.',
        'Rebased net gain = captured benefit - selected harm. No new decision is fitted for these controls.','',
        '| Action | Captured benefit | Selected harm | Missed benefit | Oracle benefit | CV fallback regression | CV fallback relief |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p,m in d['decomposition'].items():
        rows.append('| '+p+' | '+' | '.join(number(m[k]) for k in ('captured_benefit','selected_harm','missed_benefit','oracle_benefit','fallback_regression','fallback_relief'))+' |')
    rows+=['','## Held Prediction Skill','',
        'MSE skill is relative to fitting-only source-balanced constant moments with the same causal envelope.',
        'Gain AUROC labels incremental advantage over the frozen floor. These statistics do not certify risk.','',
        '| Target reference | Benefit MSE skill % | Harm MSE skill % | Reference MSE skill % | Easy harm MSE skill % | Floor gain AUROC |',
        '|---|---:|---:|---:|---:|---:|']
    for p,m in d['learnability'].items():
        rows.append('| '+p+' | '+' | '.join(number(m[k]) for k in ('benefit_MSE_skill_percent','harm_MSE_skill_percent','reference_MSE_skill_percent','easy_harm_MSE_skill_percent','floor_gain_AUROC'))+' |')
    rows+=['','## Three Forecaster Seeds','',
        '| Seed | Parent rebase gain over floor % | Floor positive gain % | Floor safe gain % | CV safe gain % |',
        '|---|---:|---:|---:|---:|']
    for s,policies in d['by_seed'].items():
        rows.append('| '+s+' | '+' | '.join(number(policies[p]['all_gain_floor']) for p in ('parent_rebased','floor_positive','floor_safe','cv_safe'))+' |')
    rows+=['','## Gates','',*[f'- {k}: {v}' for k,v in d['gates'].items()],'',
        'Intervals average dependent seed/producer/fitting-half views within each of12 localities before resampling.',
        'These are unadjusted development intervals, not independent confirmation or simultaneous claims.',
        'All parent damping calibrated-supported actions match this fixed zero-cutoff floor exactly.',
        'A weak linear probe can test this registered repair but cannot prove that no nonlinear causal predictor exists.',
        'Silver image-local obs8/pred12 at raw-frame stride12. Not historical Stage37t50, metric, seconds,',
        'human gold, physical safety, true3D, foundation evidence or submission readiness. Stage5C/SMC remain off.']
    return '\n'.join(rows)+'\n'


def main():
    d=json.loads((run.PUBLIC/'summary.json').read_text())
    (run.PUBLIC/'results.md').write_text(report(d))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['parent_original','parent_rebased','parent_calibrated_rebased','cv_positive','floor_positive','cv_safe','floor_safe']
    fig,ax=plt.subplots(figsize=(10,5.6),layout='constrained')
    for i,p in enumerate(labels):
        m=d['summary'][p]['all_gain_floor']; pt=m['point']; lo,hi=m['ci95']
        ax.errorbar(pt,i,xerr=[[pt-lo],[hi-pt]],fmt='o',color='#167d73' if pt>0 else '#ba4340',capsize=3)
    names=['Parent: CV fallback','Parent: damping fallback','Calibrated parent: damping fallback',
           'CV targets: gain only','Floor targets: gain only','CV targets: screened','Floor targets: screened']
    ax.axvline(0,color='#666666',lw=1); ax.set_yticks(range(len(labels)),names); ax.invert_yaxis()
    ax.set_xlabel('ADE gain over protected damping (%)\n3,000 locality-bootstrap draws; development only\nAverage gains do not establish selected-harm safety')
    ax.set_title('Incremental value with the same frozen floor producer')
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(run.PUBLIC/'incremental_probe.png',dpi=150,metadata={'Software':'M3W registered aggregate report'})
    plt.close(fig)
    print(json.dumps(dict(primary=d['primary'],gates=d['gates'])))


if __name__=='__main__': main()
