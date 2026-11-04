"""Report the complete C-only veto test without promoting fitting to validation."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_moving_cross_critic_veto as run
from scripts.report_m3w_moving_harm_refinement import text_once
from scripts.report_m3w_moving_amplitude_cost import interval
from src.evaluation.m3w_amplitude_opportunity import FRACTIONS

source, PUBLIC, PRIVATE = run.source, run.PUBLIC, run.PRIVATE


def benefit_loss(row):
    utility = row['secondary_full_strategy_paired']['known_pp']
    harm_reduction = row['matched_known_harm_reduction_pp']
    return None if utility is None or harm_reduction is None else harm_reduction - utility


def action_amounts(receipts):
    totals = {a: np.zeros(8, np.int64) for a in run.ARMS}
    for receipt in receipts:
        path = source.checked(receipt['decision_cache'])
        with np.load(path, allow_pickle=False) as cache:
            for arm in run.ARMS:
                count = np.bincount(cache[arm]+1, minlength=8)
                if count.shape != (8,) or count[1] != 0:
                    raise ValueError('No floor-index intervention or new action class')
                totals[arm] += count
    if totals['cross_veto'][1:].sum() != totals['self_count_matched'][1:].sum():
        raise ValueError('Complete same-count match required')
    result = {}
    for arm, counts in totals.items():
        n, selected = int(counts.sum()), int(counts[1:].sum())
        mass = float(np.dot(counts[1:], [0., *FRACTIONS]))
        result[arm] = dict(repeated_rows=n, selected_rows=selected, action_counts=counts.tolist(),
            mean_alpha_all_rows=mass/n, mean_alpha_selected=mass/selected if selected else None,
            role='pooled_repeated_rows_not_source_macro_or_CI')
    return result


def figures(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    public, private = PUBLIC / 'figures', PRIVATE / 'figure_previews'
    public.mkdir(exist_ok=True)
    private.mkdir(exist_ok=True)
    plt.rcParams.update({'svg.fonttype': 'none', 'font.size': 9})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for i, metric in enumerate(('known_pp', 'completion_lower_pp')):
        row = summary['primary_nonempty_matched'][metric]
        if row['CI95'] is None:
            axes[0].text(.02, .7-i*.3, 'Not estimable: required empty view', transform=axes[0].transAxes, fontsize=8)
        else:
            axes[0].errorbar(row['point'], i, xerr=[[row['point']-row['CI95'][0]], [row['CI95'][1]-row['point']]], fmt='o', capsize=4)
    axes[0].set_yticks([0, 1], ['Known utility', 'Direct lower'])
    axes[0].set_ylim(1.5, -.5)
    axes[0].axvline(0, color='.5', lw=1)
    axes[0].set_xlabel('Cross-veto minus same-count self (pp)')
    values = [summary['switch_percent'][a]['point'] for a in run.ARMS]
    axes[1].bar(range(4), values, color=['#35719a', '#8f657b', '#4f8c70', '#b78643'])
    axes[1].set_xticks(range(4), ['Adaptive', 'Frozen', 'Cross veto', 'Self, same\ncount'], fontsize=8)
    axes[1].set_ylabel('Source-macro switch %')
    fig.suptitle('Supervised C only: fresh prototype, not independent validation', fontsize=10)
    fig.tight_layout()
    fig.savefig(public / 'matched_diagnostic.svg')
    fig.savefig(private / 'matched_diagnostic.png', dpi=160)
    plt.close(fig)


def main():
    s = source.read(PUBLIC / 'analysis.json')
    v = source.read(PUBLIC / 'verification.json')
    if (not v['all36C_cost_arrays_and_all144cost_cells_exact']
            or not v['all_four_policy_and_count_matched_blocks_exact']
            or v['analysis_sha256'] != source.sha(PUBLIC / 'analysis.json')):
        raise ValueError('Complete real C run and replay before conclusions')
    receipts = [source.read(source.checked(r)) for r in s['group_receipts']]
    if len(receipts) != 18:
        raise ValueError('All models and seeds required')
    amounts = action_amounts(receipts)
    source.once(PUBLIC / 'action_amounts.json', source.stable(amounts))
    d = source.read(PUBLIC / 'decision.json')
    if d != source.stable(run.diagnostic_decision(s)):
        raise ValueError('Do not change the fixed fitting-diagnostic decision')
    rows = [row for receipt in receipts for row in receipt['rows']]
    sites = sorted({row['site'] for row in rows})
    benefit = source.group_interval(rows, benefit_loss, sites,
        source.read(ROOT / 'configs/m3w_moving_amplitude_cost_v1.json'))
    source.once(PUBLIC / 'benefit_harm_tradeoff.json', source.stable(dict(
        input=source.artifact(PUBLIC / 'analysis.json'),
        full_strategy_known_benefit_lost_pp=benefit,
        identity='utility_cross_minus_self = benefit_cross_minus_self + harm_self_minus_cross',
        scope='posthoc_algebraic_decomposition_full_strategy_known_labels_not_nonempty_primary_or_new_policy',
        new_training=False, new_H_prediction=False, deployment_changed=False)))
    text_once(PUBLIC / 'interpretation.md', '# Why The Shared-C Second Opinion Was Not Enough\n\n'
        'This is a post-readout algebraic explanation of the complete fixed Cprototype, '
        'not a newly selected strategy or independent validation.\n\n'
        'The full-strategy known utility is benefit minus positive harm. Using the same '
        'full known floor denominator in each source view, the cross-veto loses benefit '
        'relative to count-matched self selection by ' + interval(benefit) + 'pp. '
        'It reduces known harm by ' + interval(s['matched_known_harm_reduction_pp']) + 'pp. '
        'Both contrasts use paired source-group resampling; subtracting separate CI endpoints '
        'would not give this interval. The complete nonempty-query primary is different '
        'and remains negative as registered. Do not add its differently normalized contrast '
        'to this full-strategy harm measure.\n\n'
        'The veto therefore rejects useful predictions as well as harmful ones. Reduced '
        'harm alone did not buy better utility at a fixed query-level intervention count, '
        'and the full recording-risk contract also failed. The two models share Ctraining; '
        'this does not test a genuinely independent critic. Different retained amplitude '
        'mixtures are another limitation, not evidence of a new risk theorem.\n\n'
        'This negative result rules out promoting this particular shared-C veto or scaling '
        'it solely because its harm numerator fell. It does not prove that independently '
        'fitted risk estimation, new causal information or all future dynamics methods fail. '
        'Those require their own source-excluded fitting/validation design. No Hprototype '
        'score or threshold search was run. The original primary/support failures and '
        'reserved-source boundaries remain unchanged. No deployment, submission readiness, '
        'metric/seconds claim, Stage5C execution or SMC.\n')
    effect = ['|Cross-veto minus same-count-self|Known pp [95% CI]|Direct completion-lower pp [95% CI]|', '|---|---:|---:|']
    for label, row in (('Primary: causally nonempty queries', s['primary_nonempty_matched']),
                       ('Secondary: complete strategy population', s['secondary_full_strategy_paired'])):
        effect.append(f'|{label}|{interval(row["known_pp"])}|{interval(row["completion_lower_pp"])}|')
    table = ['|Strategy|C risk support /348|Known all/easy violations|Undefined easy|Worst defined easy-net %|Source-macro switch %|Mean alpha, switched rows|',
             '|---|---:|---:|---:|---:|---:|---:|']
    for arm in run.ARMS:
        r, amount = s['safety'][arm], amounts[arm]['mean_alpha_selected']
        amount = 'not_estimable' if amount is None else f'{amount:.6f}'
        easy = r['worst_defined_signed_easy_net_percent']
        easy = 'not_estimable' if easy is None else f'{easy:.6f}'
        table.append(f'|{arm}|{r["supported_record_views"]}|{r["known_all_risk_violations"]}/{r["known_easy_risk_violations"]}'
            f'|{r["undefined_easy_views"]}|{easy}|{s["switch_percent"][arm]["point"]:.6f}|{amount}|')
    text_once(PUBLIC / 'readout.md', '# Cross-Critic Veto: Complete C Mechanism Check\n\n'
        '## Material Passport\n\n'
        'fresh_run: all 18 original C contexts, 36 fresh frozen-neural cost arrays, fixed causal '
        'prototype and complete exact replay. cached_verified: frozen weights, original forecasts '
        'and C identity/normalizer bindings. not_run: new model training, any new H prototype '
        'evaluation, independent calibration or confirmation. Both critics share supervised C data.\n\n'
        + '\n'.join(effect) + '\n\n' + '\n'.join(table) + '\n\n'
        + 'Matched known-harm reduction, relative to full known floor cost (pp): '
        + interval(s['matched_known_harm_reduction_pp']) + '.\n\n'
        + f'Nonempty/empty repeated query views: {s["nonempty_query_views"]}/{s["empty_query_views"]}; '
        f'required source views with unavailable primary: {s["empty_source_views"]}/138. '
        f'Worth a source-excluded follow-up under the fixed diagnostic rule: {d["worth_source_excluded_followup"]}.\n\n'
        'The primary excludes queries solely by causal zero-count status, never future labels; '
        'all 138 required source views remain. A required empty/unsupported utility cannot be '
        'replaced by secondary full-strategy accuracy or called safe. All 348 recording-risk '
        'views remain for each full strategy. No deployment or validation pass follows from C.\n\n'
        'Counts match in every query, and selected action fractions are unchanged per retained '
        'agent. Fractions can nevertheless differ between the two subsets. The alpha table '
        'is descriptive pooled repeated-row composition, not source-macro or CI. Without '
        'amplitude matching this test does not isolate independent critic information from '
        'intervention severity. Nor is a second model trained on the same Cdata independent.\n\n'
        'Only the veto changes decisions; it never reselects another action. All moments and '
        'rankings are past-only. Both original C diagonal cost cells per job match their saved '
        'training diagnostics. No future label, endpoint, central velocity, test goal or '
        'future-validity mask enters the prototype. Future labels only score fixed choices.\n\n'
        'Three actual model seeds and 3000 conservative source-group bootstrap draws do not '
        'create independent physical scenes or a final test. Original 24-group primary remains '
        'not_estimable; 23 supported groups are exposed development. Reserved sources stay closed. '
        'No hypothesis is confirmed by code checks or a smaller harm numerator alone.\n\n'
        'This is not Double-Qlearning, a new safety theorem, metric/seconds prediction, true3D, '
        'foundation success or a new trajectory model. Native8/12annotation order is not verified '
        'sensor-as-of time. Stage5C execution and SMC remain disabled.\n')
    text_once(PUBLIC / 'gates.md', '# Fitting Diagnostic, Not Deployment Gates\n\n'
        + '\n'.join(f'- {k}: {value}' for k, value in d.items()) + '\n')
    lines = ['# Every Supported C Source Group', '',
        '|Source group|Primary known pp|Primary lower pp|Known harm reduction pp|Known benefit lost pp|',
        '|---|---:|---:|---:|---:|']
    for site in sites:
        metrics = (s['primary_nonempty_matched']['known_pp'], s['primary_nonempty_matched']['completion_lower_pp'],
                   s['matched_known_harm_reduction_pp'], benefit)
        values = ['not_estimable' if 'by_source_group' not in m else f'{m["by_source_group"][site]:+.6f}' for m in metrics]
        lines.append('|' + site + '|' + '|'.join(values) + '|')
    text_once(PUBLIC / 'per_source.md', '\n'.join(lines) + '\n\n'
        'Primary uses causally nonempty queries. Harm and benefit decomposition use the complete '
        'strategy population; do not combine their differently normalized columns as one identity. '
        'Source components are not verified independent physical scenes. No source is selected or dropped.\n')
    source.once(PUBLIC / 'runtime_summary.json', dict(phases={p: source.read(PUBLIC / (p+'_execution.json'))
        for p in ('run', 'verify')}, new_optimizer_updates=0, new_model_fits=0,
        phase_seconds_measurement='time.monotonic_elapsed_not_CPU_time_or_guaranteed_wall_clock_interval'))
    figures(s)
    print(source.json.dumps(dict(primary=s['primary_nonempty_matched'], decision=d), sort_keys=True))


if __name__ == '__main__':
    main()
