"""Read the verified continuation contrast without promoting a failed full gate."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.evaluation.m3w_recording_lineage import sha256

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_decoder_link_repair_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_decoder_link_repair_v1'


def read(path):
    return json.loads(path.read_text())


def number(x):
    return 'not estimable' if x is None else f'{x:+.4f}'


def interval(row):
    if row['point'] is None or row['CI95'] is None:
        return 'not estimable'
    return f"{number(row['point'])} [{number(row['CI95'][0])}, {number(row['CI95'][1])}]"


def fitting_summary():
    result = read(PUBLIC/'fitting_check/analysis.json')
    if not read(PUBLIC/'fitting_check/verify_execution.json')['exact_scalar_replay']:
        raise ValueError('Complete fitting-region diagnostic and replay required')
    rows = []
    for arm in ('frozen_parent', 'continue_original', 'decoder_auxiliary'):
        for region in ('all_neural', 'frozen_parent_independent', 'continue_original_independent', 'decoder_auxiliary_independent'):
            values = [r['scores'] for r in result['rows'] if r['arm'] == arm and r['region'] == region]
            if len(values) != 18:
                raise ValueError('All18fitting job views required')
            actual = np.sum([r['realized_selected_known_moments'] for r in values], 0)
            predicted = np.sum([r['predicted_on_selected_known_moments'] for r in values], 0)
            ratio = lambda a,b:float(a/b) if b > 0 else None
            rows.append(dict(arm=arm, region=region,
                H_over_predicted_H=ratio(actual[1],predicted[1]),
                EH_over_predicted_EH=ratio(actual[4],predicted[4]),
                known_all_risk_percent=ratio(100*actual[1],actual[2]),
                known_easy_risk_percent=ratio(100*actual[4],actual[3]),
                selected_row_views=sum(r['selected'] for r in values)))
    return rows


def policy_usage(records):
    sites = sorted({r['site'] for r in records})
    if not sites:
        raise ValueError('Nonempty source-group records required')
    result = []
    for arm in ('frozen_parent', 'continue_original', 'decoder_auxiliary'):
        rates = [np.mean([r['policies'][arm]['joint_free']['switch_rate']
            for r in records if r['site'] == site]) for site in sites]
        utility = [r['matched_utility'][arm] for r in records]
        empty = sum(r['status'] == 'not_estimable_empty_matched_control' for r in utility)
        result.append(dict(arm=arm, mean_source_group_switch_percent=float(100*np.mean(rates)),
            repeated_group_views=len(records), empty_matched_views=empty,
            empty_matched_queries=sum(r['empty_queries'] for r in utility),
            full_scope_matched_utility_estimable=empty == 0))
    return result


def figures(summary, fits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    public, private = PUBLIC/'figures', PRIVATE/'figure_previews'
    public.mkdir(exist_ok=True)
    private.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9, 'svg.fonttype':'none'})
    values = summary['paired_auxiliary_minus_continuation']['joint_free']
    fig, ax = plt.subplots(figsize=(7, 3))
    for i,key in enumerate(('completion_lower_pp','known_pp')):
        row = values[key]
        ax.errorbar(row['point'], i, xerr=[[row['point']-row['CI95'][0]], [row['CI95'][1]-row['point']]],
            fmt='o', capsize=4, color=('#1766a0','#b15c25')[i])
    ax.set_yticks([0,1], ['Completion lower','Known labels'])
    ax.invert_yaxis()
    ax.axvline(0, color='0.4', lw=1)
    ax.set_xlabel('Auxiliary minus equal-budget continuation (pp), grouped 95% CI')
    ax.set_title('Supported 23-group development secondary\nOriginal 24-group primary not estimable')
    fig.tight_layout()
    fig.savefig(public/'paired_continuation.svg')
    fig.savefig(private/'paired_continuation.png', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for arm,color in (('continue_original','#1766a0'),('decoder_auxiliary','#b15c25')):
        runs = []
        from src.world_model.m3w_selection_controller import read_checkpoint
        for receipt in fits:
            if receipt['arm'] == arm:
                runs.append(read_checkpoint(ROOT/receipt['checkpoint']['path'])['trace'])
        steps = [r['step'] for r in runs[0]]
        matrix = np.asarray([[r['original_total'] for r in trace] for trace in runs])
        ax.plot(steps,np.median(matrix,0),color=color,label=arm.replace('_',' '))
        ax.fill_between(steps,matrix.min(0),matrix.max(0),color=color,alpha=.12)
    ax.set_yscale('log')
    ax.set_xlabel('Additional updates after frozen 2k parent')
    ax.set_ylabel('Common original fitting objective')
    ax.set_title('Median of 18 fits; band is run range, not validation confidence')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(public/'common_fitting_loss.svg')
    fig.savefig(private/'common_fitting_loss.png',dpi=170)
    plt.close(fig)


def main():
    summary, freeze, verified = (read(PUBLIC/p) for p in ('summary.json','training_freeze.json','verification.json'))
    if (freeze['complete_continuations'] != 36 or not freeze['all18pairs_exact']
            or not verified['all36fresh_cost_predictions_and_policy_scalars_exact']
            or verified['summary_sha256'] != sha256(PUBLIC/'summary.json')):
        raise ValueError('Complete real continuations and original cost/policy replay required')
    fits = []
    for ref in freeze['fits']:
        if sha256(ROOT/ref['path']) != ref['sha256']:
            raise ValueError('Changed complete fit receipt')
        receipt = read(ROOT/ref['path'])
        if sha256(ROOT/receipt['checkpoint']['path']) != receipt['checkpoint']['sha256']:
            raise ValueError('Changed complete continuation checkpoint')
        fits.append(receipt)
    group_rows = []
    for ref in summary['group_receipts']:
        if sha256(ROOT/ref['path']) != ref['sha256']:
            raise ValueError('Changed complete held group receipt')
        group_rows.extend(read(ROOT/ref['path'])['rows'])
    usage = policy_usage(group_rows)
    fit = fitting_summary()
    figures(summary, fits)
    contrast = summary['paired_auxiliary_minus_continuation']['joint_free']
    lines = ['# Decoder-Link Matched Continuation', '',
        '## Material Passport', '',
        'Fresh_run: 36 real Torch continuations from 18 fixed 2k parents, 72,000 new',
        'primary updates, two complete first-job refits, full fitting-region check',
        'and held-policy inference. Cached_verified: parent forecasts/weights and',
        'separate new cost/policy scalar replay. Warm updates are not recounted.',
        'This is exposed-source development, not independent final confirmation.',
        'Original 24-group primary remains not estimable; 23 groups are secondary.', '',
        '## Fixed Equal-Budget Contrast', '',
        'Both arms continue identical weights, Adam/sampler/RNG, inputs and original',
        'normalizers for 2,000 updates. The only factor is a fixed unit auxiliary',
        'on coherent action cost fractions and harm-weighted EH/H gate logits.',
        'It changes loss geometry and example weighting; it is not solely a',
        'floating-point repair or a novel scoring rule. Earlier log-loss and',
        'hurdle negative results are not reclassified.', '',
        f"Joint-free known auxiliary-minus-continuation: {interval(contrast['known_pp'])} pp.",
        f"Fixed paired completion-lower: {interval(contrast['completion_lower_pp'])} pp.", '',
        '| Model | Policy | Known gain [95%CI] | Completion lower [95%CI] | Original risk support | Known all/easy violations | Worst known easy degradation (%) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for arm, policies in summary['safety'].items():
        for policy,risk in policies.items():
            lines.append(f"| {arm} | {policy} | {interval(summary['known_accuracy'][arm][policy])} | "
                f"{interval(summary['completion_lower'][arm][policy])} | {risk['supported_record_views']}/{risk['required_record_views']} | "
                f"{risk['known_all_risk_violations']}/{risk['known_easy_risk_violations']} | {number(risk['worst_known_easy_net_degradation_percent'])} |")
    lines += ['', 'Required views repeat producers/seeds, not independent places. Unknown or',
        'empty reference support is not measured safe zero. Same-count controls',
        'remain utility-only and cannot replace the full original 2% risk contract.', '',
        '## Intervention And Empty Controls', '',
        'Switch rates average within source group over excluded producers and',
        'actual seeds, then average source groups. They are not IID row estimates.', '',
        '| Model | Joint-free switch (%) | Empty same-count group views | Empty same-count queries |',
        '|---|---:|---:|---:|']
    for row in usage:
        lines.append(f"| {row['arm']} | {row['mean_source_group_switch_percent']:.4f} | "
            f"{row['empty_matched_views']}/{row['repeated_group_views']} | {row['empty_matched_queries']} |")
    lines += ['', 'None of these arms has a same-count utility estimate for the complete',
        'required group scope. Empty comparisons stay not estimable; they are not',
        'zero utility or safety evidence. Shrinking interventions alone does not',
        'establish a better learned admission rule.', '',
        '## Fitting Cost Reliability', '',
        'These are pooled 18-job known-mass ratios on critic-fitting sources only,',
        'not official macro results, independent tests or new policy selection.',
        'All causal regions are fixed before labels are used. Compare each model',
        'on all four regions, not just the region that looks best after selection.', '',
        '| Scoring model | Fixed selection region | Known H / predicted H | Known EH / predicted EH | Known all/easy risk (%) |',
        '|---|---|---:|---:|---:|']
    for row in fit:
        lines.append(f"| {row['arm']} | {row['region']} | {number(row['H_over_predicted_H'])} | "
            f"{number(row['EH_over_predicted_EH'])} | {number(row['known_all_risk_percent'])}/{number(row['known_easy_risk_percent'])} |")
    train, evaluate, replay = (read(PUBLIC/(p+'_execution.json')) for p in ('train','evaluate','verify'))
    lines += ['', '## Actual Cost And Scope', '',
        f"Training/refit invocation: {train['seconds']:.3f} s, peak RSS {train['peak_RSS_bytes']:,} bytes.",
        f"Held evaluation: {evaluate['seconds']:.3f} s; separate replay: {replay['seconds']:.3f} s.",
        'Native arm64 CPU4/interop1/workers0. Checkpoints/RNG/sampler support',
        'identical resume. Model/cache arrays stay private. The loss figure uses',
        'the common original fitting objective,not incomparable treatment totals.',
        'No new trajectory/JEPA/Transformer fitting, latent rollout or visual-scene',
        'contribution is established by a downstream cost-head continuation.',
        'Pixels/native 8/12 steps are not metres/seconds or SDD raw t50.',
        'No independent confirmation, deployment, submission-ready, true 3D or',
        'foundation claim. Stage5C execution and SMC are off.']
    (PUBLIC/'readout.md').write_text('\n'.join(lines)+'\n')
    (PUBLIC/'policy_usage.json').write_text(json.dumps(dict(rows=usage,
        result_source='derived_verified_group_receipts_not_new_training'),indent=2,sort_keys=True,allow_nan=False)+'\n')
    (PUBLIC/'fitting_summary.json').write_text(json.dumps(dict(rows=fit,
        result_source='derived_verified_fitting_mass_ratios_not_new_training'),indent=2,sort_keys=True,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
