"""Read the complete matched amplitude study without promoting incomplete risk."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np

PUBLIC=ROOT/'outputs/publication_readiness_2026_09/moving_amplitude_cost_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/moving_amplitude_cost_v1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(row):
    if row['point']is None or row['CI95']is None:
        return 'not estimable'
    return f"{row['point']:+.4f} [{row['CI95'][0]:+.4f},{row['CI95'][1]:+.4f}]"


def usage(records):
    sites=sorted({r['site']for r in records})
    if not sites:
        raise ValueError('Nonempty complete source records required')
    result=[]
    for arm in ('binary_only','amplitude_bank','amplitude_bank_binary_deploy'):
        rates=[];amplitudes=[];empty=0;counts=np.zeros(8,np.int64)
        for site in sites:
            rows=[r for r in records if r['site']==site]
            rates.append(np.mean([r['policies'][arm]['joint_free']['switch_rate']for r in rows]))
            amplitudes.append(np.mean([r['amplitudes'][arm]['joint_free']['mean_amplitude_all_rows']for r in rows]))
        for row in records:
            empty+=row['matched_utility'][arm]['contrast']['known_pp']is None
            counts+=row['amplitudes'][arm]['joint_free']['action_counts']
        result.append(dict(arm=arm,mean_source_switch_percent=float(100*np.mean(rates)),
            mean_source_amplitude_all_rows=float(np.mean(amplitudes)),
            repeated_action_counts=counts.tolist(),empty_matched_group_views=empty,
            required_group_views=len(records),full_scope_same_count_utility_estimable=empty==0))
    return result


def fitting_summary():
    raw=read(PUBLIC/'fitting_check/analysis.json')
    if not read(PUBLIC/'fitting_check/verify_execution.json')['all_scalars_exact']:
        raise ValueError('Complete fitting-region replay first')
    rows=[]
    for arm in ('binary_only','amplitude_bank'):
        for region in ('all_legal_full_amplitude','binary_only_independent','amplitude_bank_independent','amplitude_bank_binary_independent'):
            values=[r['scores']for r in raw['rows']if r['arm']==arm and r['region']==region]
            if len(values)!=18:
                raise ValueError('All18fitting views required in each fixed action region')
            actual=np.sum([r['realized_selected_known_moments']for r in values],0)
            pred=np.sum([r['predicted_on_selected_known_moments']for r in values],0)
            ratio=lambda i:float(actual[i]/pred[i])if pred[i]>0 else None
            rows.append(dict(arm=arm,region=region,H_over_predicted_H=ratio(1),EH_over_predicted_EH=ratio(4),
                known_selected_rows=sum(r['selected_known']for r in values)if 'selected_known'in values[0]else None))
    return rows


def figures(summary,fits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.world_model.m3w_causal_descriptor_head import read_checkpoint
    public,private=PUBLIC/'figures',PRIVATE/'figure_previews'
    public.mkdir(exist_ok=True);private.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(7.5,3.2))
    for i,key in enumerate(('completion_lower_pp','known_pp')):
        row=summary['primary_amplitude_minus_binary']['joint_free'][key]
        if row['point']is None:
            ax.text(0,i,'not estimable')
        else:
            ax.errorbar(row['point'],i,xerr=[[row['point']-row['CI95'][0]],[row['CI95'][1]-row['point']]],fmt='o',capsize=4)
    ax.set_yticks([0,1],['Completion lower','Known labels']);ax.invert_yaxis();ax.axvline(0,color='0.5',lw=1)
    ax.set_title('Amplitude-bank minus matched binary-only head\nSupported 23-group development secondary, not independent test')
    ax.set_xlabel('Paired effect (pp), grouped 95% CI')
    fig.tight_layout();fig.savefig(public/'paired_effect.svg');fig.savefig(private/'paired_effect.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3.8))
    for ax,arm in zip(axes,('binary_only','amplitude_bank')):
        traces=[read_checkpoint(ROOT/r['checkpoint']['path'])['trace']for r in fits if r['arm']==arm]
        matrix=np.array([[r['total']for r in trace]for trace in traces])
        steps=[r['step']for r in traces[0]]
        ax.plot(steps,np.median(matrix,0),color='#176e8b');ax.fill_between(steps,matrix.min(0),matrix.max(0),alpha=.15)
        ax.set_title(arm.replace('_',' '));ax.set_xlabel('Fresh optimizer updates')
        ax.set_ylabel('Own active-action fitting objective');ax.set_yscale('log')
    fig.suptitle('Different active supervision sets: loss levels are not directly comparable')
    fig.tight_layout();fig.savefig(public/'fitting_losses.svg');fig.savefig(private/'fitting_losses.png',dpi=170);plt.close(fig)


def main():
    summary,freeze,verify=(read(PUBLIC/p)for p in ('summary.json','training_freeze.json','verification.json'))
    if (freeze['complete_heads']!=36 or not freeze['all18pairs_exact']
            or not verify['all36new_cost_outputs_and_policy_scalars_exact']
            or verify['summary_sha256']!=sha(PUBLIC/'summary.json')):
        raise ValueError('Complete real training and exact inference/scalar replay required')
    # Cached legacy comparisons retain their own original scalar provenance.
    parent=ROOT/'outputs/publication_readiness_2026_09/moving_shared_motion_cost_v1'
    old_verify=read(parent/'verification.json')
    if old_verify['summary_sha256']!=sha(parent/'summary.json'):
        raise ValueError('Changed verified legacy summary')
    for ref in read(parent/'summary.json')['group_receipts']:
        if sha(ROOT/ref['path'])!=ref['sha256']:
            raise ValueError('Changed legacy cached group')
    fits=[];records=[]
    for ref in freeze['fits']:
        if sha(ROOT/ref['path'])!=ref['sha256']:
            raise ValueError('Changed complete fit receipt')
        row=read(ROOT/ref['path'])
        if sha(ROOT/row['checkpoint']['path'])!=row['checkpoint']['sha256']:
            raise ValueError('Changed complete head checkpoint')
        fits.append(row)
    for ref in summary['group_receipts']:
        if sha(ROOT/ref['path'])!=ref['sha256']:
            raise ValueError('Changed held metric receipt')
        records.extend(read(ROOT/ref['path'])['rows'])
    if len(records)!=138:
        raise ValueError('Complete supported group matrix required')
    fit=fitting_summary();use=usage(records);figures(summary,fits)
    primary=summary['primary_amplitude_minus_binary']['joint_free']
    same=summary['shared_weight_deployment_contrast']['joint_free']
    lines=['# Matched Amplitude-Conditioned Cost Learning','','## Material Passport','',
        'Fresh_run: 36 real Torch cost heads, 72,000 new primary updates, two',
        'complete first-job refits, fitting-region checks and full policy inference.',
        'Cached_verified: original forecasts/data and legacy binary policy metrics,',
        'plus a separate exact replay of all new cost outputs and policy scalars.',
        'The old forecasts are not retrained. These are exposed-source development',
        'secondary results, not independent final confirmation. Original 24-group',
        'primary remains not estimable; all 23 supported groups remain.','',
        '## Registered Contrasts','',
        f"Joint-free amplitude-bank minus binary-only known: {interval(primary['known_pp'])} pp.",
        f"Fixed paired completion lower: {interval(primary['completion_lower_pp'])} pp.",
        f"Shared-weight full bank minus binary deployment known: {interval(same['known_pp'])} pp.",
        f"Shared-weight paired completion lower: {interval(same['completion_lower_pp'])} pp.",'',
        'Intervals use 3,000 source-group resamples after excluded-producer and',
        'actual-seed averaging. They are nominal development intervals, not',
        'independent confirmation or adjustments for the wider research search.','',
        '| Model | Policy | Known gain over floor (%) [95% CI] | Completion lower (%) [95% CI] | Original risk support | Known all/easy violations | Worst known easy-net degradation (%) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for arm,policies in summary['safety'].items():
        for policy,risk in policies.items():
            worst=risk['worst_known_easy_net_degradation_percent']
            value='not estimable'if worst is None else f'{worst:.4f}'
            lines.append(f"| {arm} | {policy} | {interval(summary['known_accuracy'][arm][policy])} | "
                f"{interval(summary['completion_lower'][arm][policy])} | {risk['supported_record_views']}/{risk['required_record_views']} | "
                f"{risk['known_all_risk_violations']}/{risk['known_easy_risk_violations']} | {value} |")
    lines+=['','The legacy arm is cached_verified, not freshly inferred here or input-',
        'matched to the new seven-amplitude inventory. The two new heads share',
        'the same inputs/capacity; their treatment changes active supervision and',
        'action availability. The shared-weight binary deployment isolates only',
        'the secondary inference effect. All use the same frozen physical floor.',
        'Original CV defines easy labels; reference moments are not alpha-scaled.',
        'Actual candidate-to-floor envelopes are shared by both new heads.','',
        '## Intervention Size And Matched Utility','',
        '| Model | Source-macro switch (%) | Source-macro mean amplitude, all rows | Empty matched group views |',
        '|---|---:|---:|---:|']
    for r in use:
        lines.append(f"| {r['arm']} | {r['mean_source_switch_percent']:.4f} | {r['mean_source_amplitude_all_rows']:.6f} | "
            f"{r['empty_matched_group_views']}/{r['required_group_views']} |")
    lines+=['','Means average within source group before averaging groups. Raw amplitude',
        'histograms retain all repeated row views in the machine-readable receipts.',
        'Equal intervention counts do not mean equal amplitude/path-change mass.',
        'Same-count controls measure utility only; empty comparisons are not',
        'estimable, not safe zeros. Full-policy risk uses every required recording.','',
        '## Fitting-Side Reliability','',
        'These are pooled known-cost mass ratios over all 18 fitting jobs, not',
        'independent or official macro scores. Fixed regions include candidate',
        'identity, not just selected agents. All are evaluated before new held',
        'readout; no region or checkpoint is selected from those outcomes.','',
        '| Scoring head | Fixed causal action region | H / predicted H | EH / predicted EH |',
        '|---|---|---:|---:|']
    for r in fit:
        values=['not estimable'if r[k]is None else f'{r[k]:.4f}'for k in ('H_over_predicted_H','EH_over_predicted_EH')]
        lines.append(f"| {r['arm']} | {r['region']} | {values[0]} | {values[1]} |")
    train,evaluate,replay=(read(PUBLIC/(p+'_execution.json'))for p in ('train','evaluate','verify'))
    lines+=['','## Execution And Limits','',
        f"Training/refits: {train['seconds']:.3f} s, peak RSS {train['peak_RSS_bytes']:,} bytes.",
        f"Held evaluation: {evaluate['seconds']:.3f} s; separate replay: {replay['seconds']:.3f} s.",
        'Native arm64 CPU4/interop1/workers0, atomic checkpoints and exact resume.',
        'Fitting-loss panels use different active-action targets; smaller values',
        'in one panel do not by themselves show better learning than the other.',
        'Saved solver bindings/feasibility replay is not bitwise MILP reoptimization.',
        'Joint allocation pools predicted budgets, not geometric collision costs.',
        'Required views repeat producer/seed contexts, not independent recordings.',
        'No unsupported or empty-event view is dropped to manufacture a risk pass.',
        'Available-label ADE and bounds for wholly unknown futures do not identify',
        'unobserved steps within partial trajectories. Pixels/native 8/12 steps',
        'are not verified metres/seconds or SDD raw t50. No new trajectory backbone,',
        'JEPA lift, independent confirmation, deployment, physical-safety, true3D,',
        'foundation or submission-ready claim. Stage5C execution and SMC stay off.']
    (PUBLIC/'readout.md').write_text('\n'.join(lines)+'\n')
    (PUBLIC/'fitting_summary.json').write_text(json.dumps(dict(rows=fit),indent=2,sort_keys=True,allow_nan=False)+'\n')
    (PUBLIC/'policy_usage.json').write_text(json.dumps(dict(rows=use),indent=2,sort_keys=True,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
