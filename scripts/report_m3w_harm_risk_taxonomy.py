"""Separate abstention, missing support and observed risk without changing gates."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import run_m3w_moving_harm_refinement as run
from scripts.report_m3w_moving_harm_refinement import text_once

source=run.source
PUBLIC=run.PUBLIC
LABELS=('no_path_change','active_support_undefined','known_contract_violation',
    'unknown_completion_not_certified','supported')


def category(record):
    if record['selected']==0:return 'no_path_change'
    keys=('known_all_risk','known_easy_risk','known_easy_net_degradation')
    if any(record[k]is None for k in keys):return 'active_support_undefined'
    if any(record[k]>.02 for k in keys):return 'known_contract_violation'
    if any(record[k]is None or record[k]>.02 for k in ('completion_all_risk_upper','completion_easy_risk_upper')):
        return 'unknown_completion_not_certified'
    if not record['supported']:raise ValueError('Risk/support taxonomy missed a required condition')
    return 'supported'


def main():
    summary=source.read(PUBLIC/'summary.json');verified=source.read(PUBLIC/'verification.json')
    if not verified['all72cost_outputs_and_policy_scalars_exact']or verified['summary_sha256']!=source.sha(PUBLIC/'summary.json'):
        raise ValueError('Exact complete replay required')
    records=[row for ref in summary['group_receipts']for row in source.read(source.checked(ref))['rows']]
    use=source.read(PUBLIC/'policy_usage.json');held=source.read(PUBLIC/'held_cost_check/analysis.json')
    counts={arm:dict.fromkeys(LABELS,0)for arm in run.ARMS}
    for arm in run.ARMS:
        for group in records:
            for record in group['policies'][arm]['joint_free']['recordings'].values():counts[arm][category(record)]+=1
        if sum(counts[arm].values())!=348:raise ValueError('Retain all348repeated recording views')
        if counts[arm]['supported']!=summary['safety'][arm]['joint_free']['supported_record_views']:
            raise ValueError('Taxonomy cannot change the original support gate')
    source.once(PUBLIC/'risk_taxonomy.json',dict(counts=counts,priority=list(LABELS),
        old_gate_unchanged=True,no_path_change_is_analytic_identity_not_an_estimated_risk=True,
        overlapping_raw_violation_counts_remain_in_original_summary=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=PUBLIC/'figures';preview=run.PRIVATE/'figure_previews'
    plt.rcParams.update({'svg.fonttype':'none','font.size':9})
    fig,ax=plt.subplots(figsize=(7.4,3.2))
    for y,key,label in ((0,'known_pp','Known labels'),(1,'completion_lower_pp','Completion lower')):
        value=summary['paired']['adaptive_minus_fixed']['joint_free'][key]
        point=value['point'];lo,hi=value['CI95'];ax.hlines(y,lo,hi,color='#226b8e',lw=2)
        ax.plot(point,y,'o',color='#226b8e');ax.plot([lo,hi],[y,y],'|',color='#226b8e',markersize=8)
        ax.text(-.16,y+.23,f'{point:+.4f} [{lo:+.4f}, {hi:+.4f}] pp',fontsize=8,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=1))
    ax.axvline(0,color='.5',lw=1);ax.set_yticks([0,1],['Known labels','Completion lower'])
    ax.set_xlim(-.18,.18);ax.set_ylim(1.6,-.45)
    ax.set_xlabel('Adaptive minus fixed, percentage points; grouped 95% development CI')
    ax.set_title('Registered primary detail\n23 supported exposed source groups; not independent confirmation')
    fig.tight_layout();fig.savefig(figures/'primary_detail.svg');fig.savefig(preview/'primary_detail.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.4,4.5));x=np.arange(4);bottom=np.zeros(4)
    names=('No path change (not risk estimation)','Active support undefined','Known contract violation',
        'Unknown completion not certified','Original full support')
    colors=('#c6c6c6','#87b5d4','#c25857','#d6a153','#397f6b')
    for key,label,color in zip(LABELS,names,colors):
        values=np.array([counts[a][key]for a in run.ARMS]);ax.bar(x,values,bottom=bottom,color=color,label=label,width=.62)
        for i,value in enumerate(values):
            if value>=15:ax.text(i,bottom[i]+value/2,str(value),ha='center',va='center',fontsize=8)
        bottom+=values
    ax.set_xticks(x,run.ARMS);ax.set_ylabel('Repeated recording views (not independent videos)')
    ax.set_ylim(0,370);ax.set_title('Same 2% contract, all 348 recording views retained\nPriority partition; overlapping raw violations remain reported separately')
    handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=8,frameon=False)
    fig.subplots_adjust(bottom=.25,left=.1,right=.98,top=.84)
    fig.savefig(figures/'risk_footprint.svg');fig.savefig(preview/'risk_footprint.png',dpi=160);plt.close(fig)
    primary=summary['paired']['adaptive_minus_fixed']['joint_free'];risk=summary['safety']['adaptive_region']['joint_free']
    a=held['aggregate']['adaptive_region']['joint_free'];m=a['moments']
    text=f'''# Failure analysis: better cost magnitudes, unresolved deployment risk

## What Was Actually Tested

54new tree refiners /1,080retained primary trees on18frozen verified neural
heads. No new neural trajectory training, residual model or large backbone.
The registered primary is adaptive minus fixed region, not whichever arm looks
best on held data. All54fits finish before held scoring. All72held cost outputs
and original policy scalars replay exactly; cached solver bindings/feasibility
are verified, not bitwise MILP reoptimization. The separate288cell fitting
audit replays72fitting hashes and18old scalar blocks.

## What Improved

This is not merely selection confounding. On the exact original frozen fitting
choice region, actual/predicted H changes42.7012to0.7856, and EH202.2497to0.3729
for the adaptive refiner. It removes severe underestimation but overshoots;
these are pooled fitting masses, not a calibration theorem. On its own fitting
selection, H/EH ratios are1.0322/0.9167. References are pointwise unchanged.

Known adaptive-minus-fixed gain is {primary['known_pp']['point']:+.5f}pp,
CI[{primary['known_pp']['CI95'][0]:+.5f},{primary['known_pp']['CI95'][1]:+.5f}].
Worst signed easy-net degradation is{risk['worst_known_easy_net_degradation_percent']:.5f}%,
which passes the unchanged2%signed-net condition. Do not hide these positives.

## Why The Primary And Complete Risk Still Fail

The paired completion-lower effect is{primary['completion_lower_pp']['point']:+.5f}pp,
CI[{primary['completion_lower_pp']['CI95'][0]:+.5f},{primary['completion_lower_pp']['CI95'][1]:+.5f}].
The lower endpoint is not positive. That is failure of the registered robust
criterion, not proof that the unobserved true effect is negative.

Held own-selection actual/predicted ratios remain H{m['H']['ratio']:.4f},
EH{m['EH']['ratio']:.4f}, R{m['R']['ratio']:.4f}, ER{m['ER']['ratio']:.4f}.
Selected harm is still underestimated, and the easy reference is overestimated
in pooled mass. These are different selected fitting/held populations, so they
do not uniquely establish covariate shift or intrinsic feature insufficiency.
They do rule out treating near-correct fitting aggregates as held-record safety.

42known all-risk and72known easy-positive-harm views violate2%;47/348meet the
full original support/risk contract. Signed easy-net error can cancel gains
against harms and uses a different denominator; its1.37%pass cannot replace
positive-harm and support requirements.214selected-easy views are undefined.
The priority partition is161no-path-change views,53active-support-undefined,
72known-contract-violating,15unknown-completion-not-certified and47supported.
No-path-change is analytically zero trajectory change, not an empirical
selected-event risk estimate. This diagnostic does not change the old gate.
Some categories can also have other deficiencies; raw counts remain separate.

The source-macro switch rate falls70.9258%to5.3308%; adaptive known gain versus
floor falls11.7276%to0.3696%. The old frozen utility is unsafe, not a gold target,
but the drop shows that severe conservatism trades away most intervention
utility. The global tree control has0.3914%known gain versus floor but3.2540%
worst signed easy-net degradation; it is neither a selected winner nor safe.
48/138adaptive same-count group contrasts are empty and remain not estimable.

Source048has a negative known adaptive-minus-fixed effect (about-0.2053pp).
Source065has a small known effect but a wide missing-outcome contrast; it is
not dropped. All23source points remain in per_source.md. No camera/layout cause
is inferred from source IDs or unavailable images.

## Limits And Next Experiment

The original24group primary remains not estimable. These23groups are exposed
SOURCE development, not independent confirmation or verified IID scenes.
3000group bootstrap resamples follow producer/seed averaging. Available-label
ADE with a fixed observed-floor denominator is not full12step complete-data
ADE or a verified video-frame/seconds metric. Pixel-space only; no true3D,
foundation, metric or seconds claim. Main v9PDF is not regenerated by this report.

Next, test source-excluded, recording-level uncertainty and reference-support
estimation using fitting/model-selection roles only. Compare against the
already-tested global/fixed/adaptive controls, preserving the2%contract and
explicit abstention states. Do not set a multiplier from these held ratios,
rerun held threshold sweeps or relabel empty controls as safe. Independent new
source roles remain closed pending their explicit allocation decision.

Fitting success does not prove squared error is intrinsically wrong, that
features are unlearnable, or that another larger network will solve the issue.
No deployment upgrade or CCF-A submission-candidate claim. Stage5C and SMC
remain off. [Common-region evidence](fitting_check/readout.md),
[held post-decision evidence](held_cost_check/readout.md),
[original decision](decision.json), [all source points](per_source.md).
'''
    text_once(PUBLIC/'failure_analysis.md',text)
    print('Original gate unchanged; all348views partitioned, primary detail and risk plots generated')


if __name__=='__main__':main()
