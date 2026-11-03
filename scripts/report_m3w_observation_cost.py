"""Complete observation-support cost report from replayed public scalar evidence."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_observation_cost_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_observation_cost_v1'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    read=lambda p:json.loads(p.read_text())
    s=read(PUBLIC/'summary.json');v=read(PUBLIC/'verification.json');iv=read(PUBLIC/'inference_verification.json')
    train=read(PUBLIC/'training_freeze.json');pilot=read(PUBLIC/'pilot.json')
    if not v['all72_metric_groups_exact'] or not iv['all36_cost_support_amplitude_arrays_exact'] or v['summary_sha256']!=sha(PUBLIC/'summary.json'):
        raise ValueError('Complete separate prediction and metric replay required')
    fits=[]
    for ref in train['fits']:
        p=ROOT/ref['path']
        if sha(p)!=ref['sha256']: raise ValueError('Fitting receipt changed')
        fits.append(read(p))
    if len(fits)!=36: raise ValueError('All36fitsrequired')
    from scripts import run_m3w_observation_cost as runner
    cfg=read(ROOT/'configs/m3w_observation_cost_v1.json');registration=read(PUBLIC/'registration.json')
    paths=[PUBLIC/'groups'/(j['key']+'_'+site+'.json') for j in registration['jobs'] for site in j['evaluation_sites']]
    groups=[read(p) for p in paths]
    if runner.summarize(groups,cfg,s['source_localities'])!=s: raise ValueError('Current public groups must reproduce the summary')
    support_delta=runner.parent.locality_interval(groups,
        lambda r:r['support']['identified_support']['AUROC']-r['support']['marginal_support']['AUROC']
            if all(r['support'][a]['AUROC'] is not None for a in cfg['arms']) else None,
        s['source_localities'],3000,cfg['bootstrap_seed'])
    mean_amplitude={name:sum(g['amplitudes'][name]['mean'] for g in groups)/72 for name in
        ('marginal_support_bounded','identified_support_bounded')}
    old=PUBLIC.parent/'european_bound_aware_cost_v1'
    oldgroups=[read(old/'groups'/p.name) for p in paths]
    mean_amplitude['direct_bound_bounded']=sum(g['amplitudes']['bound_aware_bounded']['mean'] for g in oldgroups)/72
    mean=lambda values:sum(values)/len(values)
    fmt=lambda r:'not estimable' if r['point'] is None else f'{r["point"]:+.4f} [{r["CI95"][0]:+.4f}, {r["CI95"][1]:+.4f}]'
    lines=['# Observation-Support Cost Learning On SOURCE','', '## Material Passport','',
        'Fresh_run36full real Torch cost-head fits, actual recovery/full refits,',
        'held cost/support/amplitude inference, policy scoring and separate replay.',
        'Cached_verified EqMotion forecasts and frozen direct-bound comparator.',
        'This report uses public scalar receipts,not new training or inference.','',
        '## Registered Comparisons','',
        f'Primary identified-minus-direct paired completion lower: **{fmt(s["contrasts"]["primary"]["completion_lower_pp"])} pp**.',
        f'Paired known-label effect: {fmt(s["contrasts"]["primary"]["known_pp"])} pp.',
        f'Within-new-family identification lower: {fmt(s["contrasts"]["identified_minus_marginal"]["completion_lower_pp"])} pp.',
        f'Marginal-family minus direct lower: {fmt(s["contrasts"]["marginal_minus_direct"]["completion_lower_pp"])} pp.','',
        'The primary versus the old direct head is an architecture/objective family',
        'comparison,not isolated loss/capacity matching. Only the two new arms',
        'match architecture, initialization, normalization and complete query draws.',
        'A secondary comparison cannot replace the registered primary. A negative',
        'completion-lower effect is not automatically observed-ADE regression.',
        'Here the separate known-label interval is also below zero: the observed',
        'utility regression is supported under this exposed-SOURCE bootstrap.','',
        '|Policy|Known gain % [95% CI]|Individual completion-lower % [95% CI]|Supported/444|Worst signed easy-net %|',
        '|---|---:|---:|---:|---:|']
    names=('floor','full_neural','direct_bound_bounded','marginal_support_bounded','identified_support_bounded','marginal_support_binary','identified_support_binary')
    for name in names:
        a=s['accuracy'][name];r=s['safety'][name]
        lines.append(f'|{name}|{fmt(a["known"])}|{fmt(a["completion_lower"])}|{r["supported_record_views"]}/444|{r["worst_signed_easy_net_percent"]:.4f}|')
    lines+=['','Individual lower bounds must not be subtracted to replace a direct',
        'paired contrast. The zero-action floor adds no prediction error but has',
        'empty selected reference support;0/444does not make it operationally unsafe.','',
        '## Complete Risk And Subsets','',
        '|Bounded policy|Known all/easy violations|Completion all/easy violations|Undefined easy|Active %|Hard gain % [95% CI]|Easy gain % [95% CI]|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in ('direct_bound_bounded','marginal_support_bounded','identified_support_bounded'):
        a=s['accuracy'][name];r=s['safety'][name]
        lines.append(f'|{name}|{r["known_all_violations"]}/{r["known_easy_violations"]}|{r["completion_all_violations"]}/{r["completion_easy_violations"]}|{r["undefined_easy_views"]}|{100*a["switch_rate"]["point"]:.4f}|{fmt(a["hard"])}|{fmt(a["easy"])}|')
    lines+=['', 'All original2%all/easypositive-harm,signed easy-net and finite-support',
        'requirements remain. No changed threshold, epsilon intervention, selected',
        'seed or favorable locality rescues a failure. Full-policy pass is',
        f'**{str(s["original_risk_pass"]).lower()}**; registered primary support is',
        f'**{str(s["primary_supported"]).lower()}**.','',
        'Equal-locality mean amplitudes: '+', '.join(f'{k}={v:.6f}' for k,v in mean_amplitude.items())+'.',
        'Active counts alone therefore do not describe the intervention change.',
        'Composition and identification attenuate many surviving interventions.','',
        '## Fitting And Availability Diagnostics','',
        '|New arm|Full-bound fitting objective|Known-only fitting objective|Held unknown-support AUROC [95% CI]|AUPRC [95% CI]|Brier [95% CI]|ECE [95% CI]|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ('marginal_support','identified_support'):
        rows=[r for r in fits if r['arm']==arm];m=s['support'][arm]
        full=mean([r['fitting_objectives']['bound_aware']['total'] for r in rows]);known=mean([r['fitting_objectives']['known_only']['total'] for r in rows])
        lines.append(f'|{arm}|{full:.6f}|{known:.6f}|{fmt(m["AUROC"])}|{fmt(m["AUPRC"])}|{fmt(m["Brier"])}|{fmt(m["ECE"])}|')
    lines+=['', 'Fitting objectives are complete common decoded scores with actual query-',
        'visit weights,averaged across18jobs per arm. They are not the unlike',
        'multitask training totals. Availability diagnostics predict wholly missing',
        'released-label support,not baseline failure,physical visibility or complete',
        'twelve-step labels. BCE is unweighted by class; its better empirical score',
        'would still not establish calibrated selected-record risk. No comparison',
        'to the older forest AUROC is treated as a matched feature effect.','',
        f'Posthoc paired availability-AUROC diagnostic,identified-minus-marginal: {fmt(support_delta)}.',
        'This descriptive secondary target is not a trajectory/risk success and',
        'does not replace the registered primary. It predicts released labels only.','',
        '## Locality Effects','', '|Locality|Primary known pp|Primary paired lower pp|', '|---|---:|---:|']
    for site in s['source_localities']:
        k=s['contrasts']['primary']['known_pp']['by_locality'][site];l=s['contrasts']['primary']['completion_lower_pp']['by_locality'][site]
        lines.append(f'|{site}|{k:+.4f}|{l:+.4f}|')
    lines+=['','## Runtime And Roles','',
        f'Primary updates:{train["primary_optimizer_updates"]}; full-refit:{train["first_full_refit_updates"]}; pilot:{pilot["actual_optimizer_updates"]}.',
        f'Fitting PID{train["pid"]}:{train["seconds"]:.3f}s,peak{train["peak_RSS_bytes"]:,}bytes.',
        f'Separate verification PID{v["pid"]}:{v["seconds"]:.3f}s,including{iv["seconds"]:.3f}s complete cost/support/amplitude inference.',
        'All36prediction triplets and72metric groups replay exactly. Nativearm64',
        'Torch CPU4/inter-op1/workers0,atomic checkpoints,heartbeat and resume.',
        'The old direct policy reproduces exactly on every required view.','',
        '149510SOURCEwhitelist windows,74recordings,twelve exposed localities.',
        'Each evaluation locality is excluded from both forecaster and critic fits.',
        'Three actual seeds17/29/43;444recording views are repeated,not444independent',
        'scenes. CIs use3000locality resamples after excluded-view/seed averaging.',
        'These are development findings,not independent final confirmation.','',
        'No private calibration,confirmation or DroneCrowd arrays. Targets/masks',
        'are supervision/evaluation only; no future endpoint,central velocity,test',
        'endpoint goals,eventual track length or future masks in inference.',
        'Available-label ADE is not full12ADE; bounds do not fill partial futures.',
        'Pixels/rawstride12/released silver are not metric,seconds,human gold,',
        'physical safety,true3D or foundation. New models are cost heads,not new',
        'trajectory/JEPA/Transformer dynamics. No deployment,Stage5C orSMC.']
    (PUBLIC/'readout.md').write_text('\n'.join(lines)+'\n')
    (PUBLIC/'diagnostic_effects.json').write_text(json.dumps(dict(
        result_source='posthoc_public_scalar_diagnostics_not_new_primary_or_new_inference',
        support_AUROC_effect=support_delta,mean_amplitude=mean_amplitude,
        group_bindings={str(p.relative_to(ROOT)):sha(p) for p in paths},
        summary_sha256=sha(PUBLIC/'summary.json'),reporter_sha256=sha(Path(__file__)),
        deployment_changed=False,submission_ready=False),indent=2,sort_keys=True)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'m3w-observation-cost-v1','font.size':10})
    fig,ax=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    keys=('direct_bound_bounded','marginal_support_bounded','identified_support_bounded');labels=('Direct','Mixture','Identified');colors=('#6f7c86','#b46c25','#117f88')
    for i,key in enumerate(keys):
        value=s['accuracy'][key]['known'];p=value['point'];lo,hi=value['CI95']
        ax[0].errorbar(i,p,yerr=[[p-lo],[hi-p]],fmt='o',color=colors[i],capsize=4)
    ax[0].set(xticks=range(3),xticklabels=labels,ylabel='Observed ADE gain over fixed floor (%)',title='Known-label utility')
    value=s['contrasts']['primary']['completion_lower_pp'];p=value['point'];lo,hi=value['CI95']
    ax[1].errorbar(p,0,xerr=[[p-lo],[hi-p]],fmt='o',color='#b74950',capsize=5)
    ax[1].axvline(0,color='#777777',linestyle=':');ax[1].set(yticks=[],xlabel='Direct paired completion-lower effect (pp)',title='Registered primary, 95% locality CI')
    ax[2].bar(labels,[s['safety'][k]['supported_record_views'] for k in keys],color=colors)
    ax[2].axhline(444,color='#777777',linestyle='--');ax[2].set(ylabel='Supported recording/pipeline views',title='Original full-risk contract',ylim=(0,470))
    for a in ax:a.grid(axis='y',alpha=.2);a.spines[['top','right']].set_visible(False)
    fig.suptitle('Explicit observation support is not automatically safe forecast admission')
    home=PUBLIC/'figures';home.mkdir(exist_ok=True)
    fig.savefig(home/'support_cost_comparison.svg',metadata={'Date':None})
    fig.savefig(PRIVATE/'support_cost_comparison.png',dpi=140);plt.close(fig)
    (home/'sources.json').write_text(json.dumps(dict(result_source='verified_public_scalar_figure_not_new_inference',
        summary_sha256=sha(PUBLIC/'summary.json'),verification_sha256=sha(PUBLIC/'verification.json'),
        reporter_sha256=sha(Path(__file__)),svg_sha256=sha(home/'support_cost_comparison.svg'),
        deployment_changed=False,submission_ready=False),indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(readout=str(PUBLIC/'readout.md'),primary=s['contrasts']['primary'],risk=s['original_risk_pass'])))


if __name__=='__main__':main()
