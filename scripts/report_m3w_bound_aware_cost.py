"""Complete matched-cost readout from verified public scalars, not new inference."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_bound_aware_cost_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_bound_aware_cost_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    read=lambda p:json.loads(p.read_text())
    s=read(PUBLIC/'summary.json');v=read(PUBLIC/'verification.json');iv=read(PUBLIC/'inference_verification.json')
    train=read(PUBLIC/'training_freeze.json');pilot=read(PUBLIC/'pilot.json')
    if (not v['all72_metric_groups_exact'] or not iv['all36_cost_and_amplitude_arrays_exact']
            or v['summary_sha256']!=sha(PUBLIC/'summary.json')):
        raise ValueError('Complete independent cost/amplitude and metric replay required')
    fits=[]
    for ref in train['fits']:
        path=ROOT/ref['path']
        if sha(path)!=ref['sha256']:raise ValueError('Fitting receipt changed')
        fits.append(read(path))
    groups=[read(p) for p in sorted((PUBLIC/'groups').glob('*.json'))]
    if len(fits)!=36 or len(groups)!=72:raise ValueError('Complete fixed fit/group roster required')
    mean=lambda x:sum(x)/len(x)
    loss={a:{target:mean([r['fitting_objectives'][target]['total'] for r in fits if r['arm']==a])
             for target in ('known_only','bound_aware')} for a in ('known_only','bound_aware')}
    fmt=lambda r:'not estimable' if r['point'] is None else f'{r["point"]:+.4f} [{r["CI95"][0]:+.4f}, {r["CI95"][1]:+.4f}]'
    lines=['# Matched Bound-Aware SOURCE Cost Learning', '', '## Material Passport', '',
        'Fresh_run36 full real Torch cost-head fits, both first-job full refits,',
        'three-group recovery pilot, complete held cost/amplitude inference and',
        'separate replay. Cached_verified EqMotion forecasts and SOURCE input roles.',
        'This report aggregates public scalars, not fresh training/inference.', '',
        '## Registered Result', '',
        f'Primary direct paired completion-lower effect: **{fmt(s["primary_effects"]["completion_lower_pp"])} pp**.',
        f'Known-label paired effect: {fmt(s["primary_effects"]["known_pp"])} pp.',
        'The primary interval crosses zero; the registered advancement criterion',
        'does not pass. Its negative point is not proof of a statistically significant',
        'observed-ADE regression. The known-label difference is also inconclusive.', '',
        '|Policy|Known gain vs fixed floor % [95% CI]|Individual completion-lower % [95% CI]|Supported/444|Worst signed easy-net degradation %|',
        '|---|---:|---:|---:|---:|']
    names=('floor','full_neural','known_only_bounded','bound_aware_bounded','known_only_binary','bound_aware_binary')
    for name in names:
        a=s['accuracy'][name];r=s['safety'][name]
        lines.append(f'|{name}|{fmt(a["known"])}|{fmt(a["completion_lower"])}|{r["supported_record_views"]}/444|{r["worst_signed_easy_net_percent"]:.4f}|')
    lines+=['', 'Do not subtract the two individual lower bounds to replace the direct paired',
        'primary. Unknown-row differences use disagreement between the two actual',
        'policy paths, not subtraction of their separate worst-case bounds.',
        'The zero-action floor adds no error but has no selected reference support;',
        'its0/444 is not evidence that retaining the baseline is operationally unsafe.', '',
        '## Risk And Intervention Trade-Off', '',
        '|Bounded policy|Known all/easy violations|Completion all/easy violations|Undefined easy support|Active rate %|Mean amplitude, equal locality|',
        '|---|---:|---:|---:|---:|---:|']
    for name in ('known_only_bounded','bound_aware_bounded'):
        r=s['safety'][name]
        bysite={site:mean([g['amplitudes'][name]['mean'] for g in groups if g['site']==site]) for site in s['source_localities']}
        lines.append(f'|{name}|{r["known_all_violations"]}/{r["known_easy_violations"]}|{r["completion_all_violations"]}/{r["completion_easy_violations"]}|{r["undefined_easy_views"]}|{100*s["accuracy"][name]["switch_rate"]["point"]:.4f}|{mean(list(bysite.values())):.6f}|')
    lines+=['', 'Full-risk support improves28->108views, but known-all violations increase',
        '79->143 while known-easy violations decrease330->235. This is a mixed trade-off,',
        'not uniform safety improvement. No candidate satisfies the original contract.',
        'There is no epsilon intervention, confidence search or selected favorable seed.', '',
        '## Same-Target Fitting Check', '',
        '|Fitted arm|Score against known-only target|Score against full bound target|',
        '|---|---:|---:|']
    for a in ('known_only','bound_aware'):
        lines.append(f'|{a}|{loss[a]["known_only"]:.6f}|{loss[a]["bound_aware"]:.6f}|')
    lines+=['', 'These are complete decoded fitting objectives using actual query-visit',
        'multiplicities, then averaging eighteen jobs. Compare models within each',
        'target column; unlike training objectives are not directly a performance',
        'ranking. Both arms share preprocessing, initial weights, architecture,',
        'query draws and full coverage budgets. Training/fitting errors are not',
        'held-domain or selected-policy calibration results.', '',
        '## Observed Failure Slices', '',
        '|Locality|Known paired effect pp|Direct paired lower pp|',
        '|---|---:|---:|']
    for site in s['source_localities']:
        lines.append(f'|{site}|{s["primary_effects"]["known_pp"]["by_locality"][site]:+.4f}|{s["primary_effects"]["completion_lower_pp"]["by_locality"][site]:+.4f}|')
    records=[]
    for g in groups:
        for key,r in g['policies']['bound_aware_bounded']['recordings'].items():
            if r['known_easy_net_degradation'] is not None:
                pred=g['amplitudes']['bound_aware_bounded']['predicted_record_costs'][key]
                records.append(dict(trial=g['trial'],site=g['site'],recording=key,actual=r,predicted=pred))
    worst=max(records,key=lambda r:r['actual']['known_easy_net_degradation']);p=worst['predicted'];a=worst['actual']
    ratio=100*p['easy_harm']/p['easy_reference'] if p['easy_reference']>0 else None
    lines+=['',f'Worst signed easy-net view: {worst["site"]}, recording{worst["recording"]}, {worst["trial"]}.',
        f'Predicted easy-risk ratio is {ratio:.6f}% but known easy-risk is {100*a["known_easy_risk"]:.4f}%',
        f'and completion-easy upper is {100*a["completion_easy_risk_upper"]:.4f}%.',
        'The predicted numerator and denominator are both estimates; this contrast',
        'does not identify which one alone causes the error. It directly shows that',
        'the nominal predicted guard is not calibrated to actual selected risk.', '',
        '## Data Roles And Compute', '',
        'For each forecaster groupF, critic fitting uses another groupG and evaluation',
        'uses the remaining groupH. Both G choices rotate. Evaluation sites are in',
        'neither forecaster nor critic fitting. All149,510SOURCE whitelist windows,',
        '74recordings,twelve localities and actual seeds17/29/43 remain. The444',
        'recording/pipeline views are repeats, not444independent scenes. SOURCE has',
        'already informed research choices; this is not independent final confirmation.',
        'Intervals use3000locality resamples after excluded-fold/seed averaging.',
        'Gains use fitting-selected floor denominators,not the earlier CV denominator.', '',
        f'Primary updates:{train["primary_optimizer_updates"]};full-refit updates:{train["first_full_refit_updates"]};pilot:{pilot["actual_optimizer_updates"]}.',
        f'Primary fitting PID{train["pid"]}:{train["seconds"]:.3f}s,peak{train["peak_RSS_bytes"]:,}bytes.',
        f'Separate complete verification PID{v["pid"]}:{v["seconds"]:.3f}s including{iv["seconds"]:.3f}s cost/amplitude inference.',
        'All36cost/amplitude arrays and72metric groups replay exactly. Nativearm64',
        'CPU4/inter-op1/workers0,atomic checkpoints,heartbeat and explicit resume.',
        'These are cost heads,not new trajectory/JEPA/Transformer training.', '',
        '## Verdict', '',
        'Explicit bound targets materially improve their fitting objective and some',
        'risk views,but do not repair complete policy safety or support the registered',
        'paired primary. Keep deployment unchanged. Do not choose a binary arm or',
        'favorable locality to rescue the failed primary. New reference-normalization,',
        'threshold,amplitude or cohort rules require their own prospective protocol.',
        'No calibration private arrays,final confirmation or DroneCrowd are accessed.',
        'Masked ADE is not full12ADE; bounds cover wholly unknown rows,not missing',
        'partial-future steps. Pixels/rawstride12/released silver only,not metric,',
        'seconds,human gold,physical safety,true3D or foundation. Stage5C/SMC off.', '']
    (PUBLIC/'readout.md').write_text('\n'.join(lines))
    (PUBLIC/'selected_risk_failure.json').write_text(json.dumps(worst,indent=2,sort_keys=True)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','svg.fonttype':'none',
        'svg.hashsalt':'m3w_bound_aware_cost','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for i,(name,color) in enumerate((('known_only_bounded','#707b85'),('bound_aware_bounded','#187e8c'))):
        r=s['accuracy'][name]['known'];p=r['point'];lo,hi=r['CI95']
        axes[0].errorbar(i,p,yerr=[[p-lo],[hi-p]],fmt='o',capsize=4,color=color)
        axes[2].bar(i,s['safety'][name]['supported_record_views'],color=color,width=.55)
    r=s['primary_effects']['completion_lower_pp'];p=r['point'];lo,hi=r['CI95']
    axes[1].errorbar([p],[0],xerr=[[p-lo],[hi-p]],fmt='o',capsize=4,color='#b24c54')
    axes[1].axvline(0,color='#777777',linestyle=':',linewidth=1)
    axes[1].set(yticks=[],xlabel='Direct paired completion-lower effect (pp)',title='Registered primary: unsupported',ylim=(-1,1))
    for ax in (axes[0],axes[2]):ax.set(xticks=[0,1],xticklabels=['Known only','Bound aware'])
    axes[0].set(ylabel='Known ADE gain over fixed floor (%)',title='Observed utility, 95% locality CI')
    axes[2].axhline(444,color='#777777',linestyle='--',linewidth=1)
    axes[2].set(ylabel='Views meeting full original contract',title='More supported views, no full pass',ylim=(0,470))
    for ax in axes:ax.grid(axis='y' if ax!=axes[1] else 'x',alpha=.2)
    fig.suptitle('Explicit unknown bounds help fitting; full selected-risk failure remains',fontsize=11)
    directory=PUBLIC/'figures';directory.mkdir(exist_ok=True)
    fig.savefig(directory/'cost_target_comparison.svg',metadata={'Date':None})
    fig.savefig(PRIVATE/'cost_target_comparison.png',dpi=150);plt.close(fig)
    receipt=dict(result_source='verified_public_scalars_fresh_report_and_figure_not_new_inference',
        summary_sha256=sha(PUBLIC/'summary.json'),verification_sha256=sha(PUBLIC/'verification.json'),
        reporter_sha256=sha(Path(__file__)),svg_sha256=sha(directory/'cost_target_comparison.svg'),
        private_preview_sha256=sha(PRIVATE/'cost_target_comparison.png'),preview_visually_checked=False,
        deployment_changed=False,submission_ready=False)
    (directory/'sources.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
