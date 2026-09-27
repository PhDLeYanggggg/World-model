"""Deterministic tables and figures for the frozen centered-risk policy test."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_centered_risk_policy as run


def fmt(v,scale=1):
    if v['point'] is None:return 'undefined (fixed roster)'
    ci=v['ci95'];return f"{scale*v['point']:.6f} [{scale*ci[0]:.6f}, {scale*ci[1]:.6f}]"


def main():
    s=json.loads((run.PUBLIC/'summary.json').read_text())
    lines=['# Centered-Risk Policy: Same-Count Development Test','','## Evidence Role','',
        'fresh_run:108new causal decision groups and a held-development readout. cached_verified: neural forecasts, risk heads, protected floor, utilities and216fitting-only offsets. No new training or fitting in this policy trial.',
        'The twelve localities were already opened source-training development data. Producer/controller/head-fit/held roles remain4/4/2/2. No independent selection, calibration or confirmation data opened. Obs8/pred12,raw-frame stride12,image-local detector-silver; not metric, seconds, human gold or physical safety.',
        'All decisions were committed before reading the new held costs. Both parent families are reported, not selected after the readout. The2%selected-harm screen and undefined denominators are unchanged.','',
        '## All Registered Policies','','Values are equal-locality means with3000paired-locality bootstrap draws. Seeds17/29/43 and repeated source-role views are averaged within locality;216views are not216independent scenes. Nominal intervals are exploratory, not multiplicity-adjusted or independent-confirmation guarantees.','',
        '| Policy | ADE gain/floor % [CI] | FDE gain/floor % | Hard gain/floor % | Intervention % | Selected harm risk | Violations / undefined views |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p in run.POLICIES:
        m=s['summary'][p];w=s['worst_views'][p]
        lines.append(f"| {p} | {fmt(m['all_gain_floor'])} | {fmt(m['FDE_gain_floor'])} | {fmt(m['hard_gain_floor'])} | {fmt(m['intervention_rate'],100)} | {fmt(m['selected_positive_harm_ratio'])} | {w['risk_violating_views']} / {w['undefined_selected_risk_views']} |")
    lines+=['','## Registered Paired Contrasts','','Positive ADE gain means lower error. Positive harm reduction means less positive harm. The all-reference diagnostic uses the fixed total floor-error denominator; it is not the selected-risk primary and has no substituted2%threshold.','',
        '| New versus control | ADE gain % [CI] | Fixed-denominator harm reduction pp [CI] | Intervention difference pp |','|---|---:|---:|---:|']
    for p,m in s['paired'].items():lines.append(f"| {p} | {fmt(m['ADE_gain_percent'])} | {fmt(m['all_reference_harm_reduction_pp'])} | {fmt(m['intervention_difference_pp'])} |")
    lines+=['','## Added Harm and Lost Benefit','','All contributions use the same total floor-error denominator within each view. They are descriptive percentage points, not a new risk certificate.','',
        '| Contrast | Added benefit pp | Added harm pp | Removed benefit pp | Removed harm pp | Net gain change pp |','|---|---:|---:|---:|---:|---:|']
    for p,m in s['exchanges'].items():
        fields=['added_benefit_over_floor_pp','added_harm_over_floor_pp','removed_benefit_over_floor_pp','removed_harm_over_floor_pp','gain_change_over_floor_pp']
        lines.append('| '+p+' | '+' | '.join(fmt(m[k]) for k in fields)+' |')
    lines+=['','## Easy, Tail and Unknown Labels','','| Policy | Worst view easy gain/CV % | p95 error ratio to floor [CI] | Unknown interventions [mean, CI] | Entirely abstaining views |','|---|---:|---:|---:|---:|']
    for p in run.POLICIES:
        m=s['summary'][p];w=s['worst_views'][p]
        lines.append(f"| {p} | {w['worst_easy_gain_CV']:.6f} | {fmt(m['p95_ratio_to_floor'])} | {fmt(m['unknown_interventions'])} | {w['abstaining_views']} |")
    diag=run.PUBLIC/'admission_diagnosis.json'
    if diag.exists():
        d=json.loads(diag.read_text())
        lines+=['','## Admission Collapse: Posthoc Causal Accounting','',
            'Counts repeat rows across source-role/seed contexts; they are not independent observations. No outcome or threshold is used in this decomposition.','',
            '| Arm | Old independent admitted | Retained | Rejected all axis only | Rejected easy axis only | Rejected both |','|---|---:|---:|---:|---:|---:|']
        for arm,m in d['summary'].items():lines.append('| '+arm+' | '+' | '.join(str(m[k]) for k in ('raw_admitted','centered_retained','rejected_all_only','rejected_easy_only','rejected_both'))+' |')
    lines+=['','## Gate Results','','```json',json.dumps(s['gates'],indent=2),'```','',
        'Passing a numeric development contrast does not certify future selected-set risk. Undefined selected-risk views are not set to zero or dropped. No deployment change, formal-primary replacement, Stage5C execution or SMC activation.','',
        '## Compute','','| Phase | Seconds | Peak RSS bytes | PID |','|---|---:|---:|---:|']
    for phase in ('decision_runtime','evaluation_runtime','prediction_replay','evaluation_replay'):
        path=run.PUBLIC/(phase+'.json')
        if path.exists():
            r=json.loads(path.read_text());lines.append(f"| {phase} | {r['seconds']:.2f} | {r['peak_RSS_bytes']} | {r['pid']} |")
    lines+=['','No new checkpoint is necessary: parent checkpoints and fitted offsets are immutable references. Actions, group hashes and heartbeat are saved locally. Full legacy integration tests and cold raw reconstruction remain not_run; scoped verification is recorded separately.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    loc=['# Locality and Seed Breakdown','','No locality is silently removed when one of its repeated views has an undefined metric.','',
         '| Arm | Locality | Centered ADE/floor % | Same-count raw ADE/floor % | Centered hard/floor % | Matched ADE contrast % | Fixed-denominator harm reduction pp |','|---|---|---:|---:|---:|---:|---:|']
    for arm in run.ARMS:
        a=s['summary'][arm+'_centered_joint'];b=s['summary'][arm+'_matched_raw_joint'];m=s['paired'][arm+'_centered_joint_vs_'+arm+'_matched_raw_joint']
        for site in a['all_gain_floor']['by_site']:
            vals=[a['all_gain_floor']['by_site'][site],b['all_gain_floor']['by_site'][site],a['hard_gain_floor']['by_site'][site],m['ADE_gain_percent']['by_site'][site],m['all_reference_harm_reduction_pp']['by_site'][site]]
            loc.append('| '+arm+' | '+site+' | '+' | '.join('undefined' if v is None else f'{v:.8f}' for v in vals)+' |')
    loc+=['','| Seed | Arm | Centered ADE/floor % | Same-count raw ADE/floor % |','|---|---|---:|---:|']
    for seed,policies in s['by_seed'].items():
        for arm in run.ARMS:loc.append(f"| {seed} | {arm} | {fmt(policies[arm+'_centered_joint']['all_gain_floor'])} | {fmt(policies[arm+'_matched_raw_joint']['all_gain_floor'])} |")
    (run.PUBLIC/'locality_breakdown.md').write_text('\n'.join(loc)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,metric,title in zip(axes,['ADE_gain_percent','all_reference_harm_reduction_pp'],['ADE gain at identical counts (%)','Positive harm reduction (percentage points)']):
        for i,arm in enumerate(run.ARMS):
            v=s['paired'][arm+'_centered_joint_vs_'+arm+'_matched_raw_joint'][metric]
            if v['point'] is not None:
                point=v['point'];lo,hi=v['ci95'];ax.errorbar(point,i,xerr=[[point-lo],[hi-point]],fmt='o',capsize=5,color=['#167c80','#a14758'][i])
        ax.axvline(0,color='gray',linestyle='--',linewidth=1);ax.set_yticks(range(2),['Pointwise head','Subset-aggregate head']);ax.set_title(title,fontsize=10);ax.grid(axis='x',alpha=.2)
        ax.xaxis.set_major_locator(MaxNLocator(4));ax.tick_params(axis='x',labelsize=9);ax.set_ylim(-.5,1.5)
    fig.suptitle('Centered risk versus raw risk at the same per-query count\n12 development localities; nominal paired bootstrap intervals',fontsize=11)
    fig.savefig(run.PUBLIC/'matched_contrasts.png',dpi=150,metadata={'Software':'M3W deterministic report'});plt.close(fig)


if __name__=='__main__':main()
