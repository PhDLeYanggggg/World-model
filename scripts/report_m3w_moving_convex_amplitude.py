"""Read the complete convex decoder comparison without selecting a held winner."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.report_m3w_moving_amplitude_cost import interval,secondary_intervals

PUBLIC=ROOT/'outputs/publication_readiness_2026_09/moving_convex_amplitude_v1'
PARENT=ROOT/'outputs/publication_readiness_2026_09/moving_amplitude_cost_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/moving_convex_amplitude_v1'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def usage(records):
    sites=sorted({r['site']for r in records});rows=[]
    for arm in ('unconstrained','convex'):
        switches=[];amplitudes=[];empty=0;counts=np.zeros(8,np.int64)
        for site in sites:
            view=[r for r in records if r['site']==site]
            switches.append(np.mean([r['policies'][arm]['joint_free']['switch_rate']for r in view]))
            amplitudes.append(np.mean([r['amplitudes'][arm]['joint_free']['mean_amplitude_all_rows']for r in view]))
        for r in records:
            empty+=r['matched_utility'][arm]['contrast']['known_pp']is None
            counts+=r['amplitudes'][arm]['joint_free']['action_counts']
        rows.append(dict(arm=arm,source_macro_switch_percent=float(100*np.mean(switches)),
            source_macro_amplitude_all_rows=float(np.mean(amplitudes)),repeated_action_counts=counts.tolist(),
            empty_matched_group_views=empty,required_matched_group_views=len(records),
            full_scope_same_count_utility_estimable=empty==0))
    return rows


def solvers(jobs):
    if len(set(jobs))!=18:raise ValueError('Every held job required')
    rows=[]
    for job in sorted(set(jobs)):
        for arm,base,tag in (('unconstrained',ROOT/'data/stage_cvpr2027_experiments/moving_amplitude_cost_v1','amplitude_bank'),
                ('convex',PRIVATE,'convex')):
            p=base/'allocation'/(job+'_'+tag+'.npz')
            with np.load(p,allow_pickle=False)as raw:meta=json.loads(str(raw['metadata_json']))
            if (meta['bindings']['seconds']!=.25 or meta['completed_queries']<=0
                    or any(sum(meta['reason_counts'][policy].values())!=meta['completed_queries']
                        for policy in ('joint_free','joint_matched'))):
                raise ValueError('Original complete solver-budget receipts required')
            rows.append(dict(arm=arm,job=job,queries=meta['completed_queries'],reasons=meta['reason_counts'],
                nonoptimal=meta['nonoptimal_queries'],changed=meta['changed_queries'],sha256=sha(p),
                result_source='cached_verified'if arm=='unconstrained'else'fresh_run_saved_feasibility_replayed'))
    return rows


def figures(summary,fit,controls,use):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.world_model.m3w_causal_descriptor_head import read_checkpoint
    out=PUBLIC/'figures';out.mkdir(exist_ok=True)
    preview=PRIVATE/'figure_previews';preview.mkdir(exist_ok=True)
    plt.rcParams.update({'svg.fonttype':'none','font.size':9})
    fig,ax=plt.subplots(figsize=(7.5,3))
    for i,key in enumerate(('completion_lower_pp','known_pp')):
        r=summary['primary_convex_minus_unconstrained']['joint_free'][key]
        if r['point']is None:ax.text(0,i,'not estimable')
        else:ax.errorbar(r['point'],i,xerr=[[r['point']-r['CI95'][0]],[r['CI95'][1]-r['point']]],fmt='o',capsize=4)
    ax.set_yticks([0,1],['Completion lower','Known labels']);ax.invert_yaxis();ax.axvline(0,color='.5',lw=1)
    ax.set_title('Convex decoder minus matched unconstrained amplitude head\n23 supported source groups, exposed development secondary')
    ax.set_xlabel('Paired effect (pp), grouped 95% development CI')
    fig.tight_layout();fig.savefig(out/'paired_effect.svg');fig.savefig(preview/'paired_effect.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,3.5))
    for arm,items,color in (('Unconstrained, cached control',controls,'#176e8b'),('Convex, fresh fit',fit,'#b45151')):
        traces=[read_checkpoint(ROOT/r['checkpoint']['path'])['trace']for r in items]
        matrix=np.array([[r['total']for r in trace]for trace in traces]);steps=[r['step']for r in traces[0]]
        ax.plot(steps,np.median(matrix,0),label=arm,color=color)
        ax.fill_between(steps,matrix.min(0),matrix.max(0),alpha=.1,color=color)
    ax.set_yscale('log');ax.set_xlabel('Optimizer updates per head');ax.set_ylabel('Same active-action fitting objective')
    ax.set_title('Median and min-max across 18 fitting runs per arm\nNot validation intervals or a safety result');ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(out/'fitting_losses.svg');fig.savefig(preview/'fitting_losses.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,3))
    for i,row in enumerate(use):
        counts=np.asarray(row['repeated_action_counts']);values=np.r_[counts[0],counts[2:]]/counts.sum()*100
        ax.bar(np.arange(7)+(i-.5)*.36,values,width=.36,label=row['arm'])
    ax.set_xticks(np.arange(7),['Floor','1/32','1/16','1/8','1/4','1/2','1'])
    ax.set_xlabel('Chosen intervention amplitude');ax.set_ylabel('Repeated row views (%)')
    ax.set_title('Joint-free action usage, row-weighted rather than independent sources');ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(out/'amplitudes.svg');fig.savefig(preview/'amplitudes.png',dpi=170);plt.close(fig)


def main():
    summary=read(PUBLIC/'summary.json');verify=read(PUBLIC/'verification.json');freeze=read(PUBLIC/'training_freeze.json')
    if (not verify['all36cost_outputs_and_policy_scalars_exact']or not verify['cached_control_policies_exact_to_parent']
            or verify['summary_sha256']!=sha(PUBLIC/'summary.json')or freeze['complete_new_heads']!=18):
        raise ValueError('Full fits, cached-control equality and separate exact replay required')
    refits=read(PUBLIC/'first_full_refits.json')
    if not refits['convex_exact']or not refits['unconstrained_exact_to_original']:
        raise ValueError('Both complete first-job refits required')
    fits=[];controls=[];records=[]
    for ref in freeze['fits']:
        p=ROOT/ref['path']
        if sha(p)!=ref['sha256']:raise ValueError('Changed new fit receipt')
        r=read(p)
        if sha(ROOT/r['checkpoint']['path'])!=r['checkpoint']['sha256']:raise ValueError('Changed new checkpoint')
        fits.append(r)
    for ref in read(PARENT/'training_freeze.json')['fits']:
        p=ROOT/ref['path']
        if sha(p)!=ref['sha256']:raise ValueError('Changed original control receipt')
        r=read(p)
        if r['arm']=='amplitude_bank':
            if sha(ROOT/r['checkpoint']['path'])!=r['checkpoint']['sha256']:raise ValueError('Changed control checkpoint')
            controls.append(r)
    for ref in summary['group_receipts']:
        p=ROOT/ref['path']
        if sha(p)!=ref['sha256']:raise ValueError('Changed complete metric receipt')
        records.extend(read(p)['rows'])
    if len(fits)!=18 or len(controls)!=18 or len(records)!=138:raise ValueError('Complete comparison required')
    use=usage(records);solver=solvers([r['job']['key']for r in fits])
    secondary=secondary_intervals(records,read(ROOT/'configs/m3w_moving_amplitude_cost_v1.json'))
    figures(summary,fits,controls,use)
    primary=summary['primary_convex_minus_unconstrained']['joint_free']
    pred=sum(np.asarray(r['own_selected_costs']['predicted_on_selected_known_moments'])for r in fits)
    actual=sum(np.asarray(r['own_selected_costs']['realized_selected_known_moments'])for r in fits)
    fraction=lambda i:float(actual[i]/pred[i])if pred[i]>0 else None
    fitting=dict(known_H_over_predicted_H=fraction(1),known_EH_over_predicted_EH=fraction(4),
        all18predicted_H_EH_shape_counts_zero=all(not any(v for comp in r['fitting_shape']['predicted'].values()for v in comp.values())for r in fits))
    lines=['# Matched Convex Amplitude Decoder','','## Material Passport','',
        'Fresh_run: 18 real Torch convex heads, 36,000 new primary updates, complete',
        'first-convex refit and first-control refit, new held convex decisions.',
        'Cached_verified: all 18 original unconstrained control heads and their',
        'saved decisions, with fresh exact fitting/held predictions and policy',
        'scalar replay. This is not 36 fresh fits. Separate full replay verifies',
        'all new outputs/scalars and saved solver feasibility, not bitwise MILP',
        'reoptimization. Original 24-group primary remains not estimable.','',
        '## Fixed Supported-Group Development Contrast','',
        'Convex minus unconstrained joint-free known: '+interval(primary['known_pp'])+' pp.',
        'Paired completion lower: '+interval(primary['completion_lower_pp'])+' pp.','',
        'All intervals use 3,000 source-group resamples after producer/actual-seed',
        'averaging. They are exposed-development intervals, not independent final',
        'confirmation or corrections for the research search. No held threshold,',
        'seed, checkpoint or scientific endpoint was selected.','',
        '| Arm | Policy | Known gain over floor (%) [95% CI] | Completion lower (%) [95% CI] | Original support | Known all/easy violations | Undefined easy | Worst known easy-net (%) |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for arm,policies in summary['safety'].items():
        for policy,r in policies.items():
            worst='not estimable'if r['worst_known_easy_net_degradation_percent']is None else f"{r['worst_known_easy_net_degradation_percent']:.4f}"
            lines.append(f"| {arm} | {policy} | {interval(summary['known_accuracy'][arm][policy])} | {interval(summary['completion_lower'][arm][policy])} | "
                f"{r['supported_record_views']}/{r['required_record_views']} | {r['known_all_risk_violations']}/{r['known_easy_risk_violations']} | {r['undefined_easy_views']} | {worst} |")
    lines+=['','The floor makes no path change; undefined selected-risk support is not',
        'evidence that it causes harm, nor an empirical finite-support certificate.',
        'Every unsupported/empty-event view is retained. All known easy/hard,',
        'available-label ADE, mean group-p95 and available-endpoint FDE diagnostics',
        'are in `secondary_metrics.json`. Missing required subsets stay not',
        'estimable; endpoints/partial labels are not a full-roster 12-step claim.','',
        '## Intervention Size And Fitting Diagnostics','',
        '| Arm | Source-macro switch (%) | Source-macro mean amplitude, all rows | Empty same-count group views |',
        '|---|---:|---:|---:|']
    for r in use:lines.append(f"| {r['arm']} | {r['source_macro_switch_percent']:.4f} | {r['source_macro_amplitude_all_rows']:.6f} | {r['empty_matched_group_views']}/{r['required_matched_group_views']} |")
    lines+=['','Same-count comparisons are utility only. Empty is not estimable, not a',
        'safe zero; equal counts need not mean equal path-change amplitude.',
        'Histograms count repeated row views, unlike source-macro switch means.',
        f"Own fitting-region known H / predicted H: {fitting['known_H_over_predicted_H']}; EH / predicted EH: {fitting['known_EH_over_predicted_EH']}.",
        f"All 18 constrained H/EH fitting-shape checks zero: {fitting['all18predicted_H_EH_shape_counts_zero']}.",
        'These are pooled resubstitution cost masses, not calibration or a causal',
        'decomposition. Correct shape need not mean correct cost magnitude.','',
        '## Solver And Execution','',
        '| Arm | Repeated query views | Nonoptimal matched | Nonoptimal free |',
        '|---|---:|---:|---:|']
    for arm in ('unconstrained','convex'):
        rows=[r for r in solver if r['arm']==arm]
        lines.append(f"| {arm} | {sum(r['queries']for r in rows)} | {sum(r['nonoptimal'].get('joint_matched',0)for r in rows)} | {sum(r['nonoptimal'].get('joint_free',0)for r in rows)} |")
    for phase in ('pilot','train','evaluate','verify'):
        r=read(PUBLIC/(phase+'_execution.json'))
        lines.append(f"{phase}: {r['seconds']:.3f} s; peak RSS {r['peak_RSS_bytes']:,} bytes.")
    lines+=['','Fixed 0.25s/query; timeout/post-check failure retains the original',
        'predicted-feasible independent parent, not a true-risk pass or exact',
        'optimum. Joint allocation pools predicted budgets, not collision costs.',
        'Code/config/normalization/input/coverage/budgets are frozen. The decoder',
        'uses nonnegative convex hinges with stable linear arithmetic, not a',
        'precision ablation or an invented general convex-network architecture.',
        'FP32 fitting loss and FP64 hinge inference are disclosed in the protocol.',
        'No independent calibration or confirmation was accessed. This is not',
        'new trajectory dynamics, JEPA, latent generation or deployment. Pixels',
        'and native annotation steps are not verified metres/seconds or SDD t50.',
        'Partial-label error and whole-missing bounds leave partial steps unknown.',
        'No true3D, foundation, physical-safety or submission-ready claim.',
        'Stage5C execution and SMC remain disabled.']
    decision=dict(primary_lower_positive=primary['completion_lower_pp']['CI95']is not None and primary['completion_lower_pp']['CI95'][0]>0,
        complete_original_risk_pass=summary['safety']['convex']['joint_free']['full_original_risk_pass'],
        original24group_primary='not_estimable_not_passed',independent_confirmation=False,
        new_deployment=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)
    for name,value in (('policy_usage',dict(rows=use,action_count_order=['fallback','unavailable_floor_alias','1/32','1/16','1/8','1/4','1/2','1'])),
            ('solver_summary',dict(rows=solver)),('fitting_summary',fitting),('secondary_metrics',secondary),('decision',decision)):
        (PUBLIC/(name+'.json')).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    (PUBLIC/'readout.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
