"""Read the fixed current-selection comparison; never select a model from H."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import evaluate_m3w_moving_adaptive_record_cost as run
from scripts.report_m3w_moving_harm_refinement import text_once
from scripts.report_m3w_moving_amplitude_cost import interval, secondary_intervals

source, PUBLIC, PRIVATE = run.source, run.PUBLIC, run.PRIVATE


def positive(effect):
    ci = effect['CI95']
    return ci is not None and len(ci) == 2 and np.isfinite(ci).all() and ci[0] > 0


def scientific_decision(summary):
    risk = summary['safety']['adaptive']['joint_free']
    support = (risk['required_record_views'] == 348 and risk['supported_record_views'] == 348
        and risk['full_original_risk_pass'] and risk['undefined_easy_views'] == 0
        and risk['known_all_risk_violations'] == 0 and risk['known_easy_risk_violations'] == 0
        and risk['worst_known_easy_net_degradation_percent'] is not None
        and np.isfinite(risk['worst_known_easy_net_degradation_percent'])
        and risk['worst_known_easy_net_degradation_percent'] <= 2)
    contrasts = {arm: positive(summary['paired'][arm]['joint_free']['completion_lower_pp'])
        for arm in ('point', 'shared_tree', 'frozen_teacher')}
    own = positive(summary['completion_lower']['adaptive']['joint_free'])
    return dict(primary_vs_frozen_teacher_positive=contrasts['frozen_teacher'],
        strong_shared_tree_contrast_positive=contrasts['shared_tree'],
        original_point_contrast_positive=contrasts['point'], own_floor_lower_positive=own,
        full_original_risk_pass=bool(support),
        supported_source_mechanism_gate=bool(contrasts['frozen_teacher'] and contrasts['shared_tree'] and own and support),
        original24group_primary='not_estimable_not_passed', independent_confirmation=False,
        new_neural_cost_training=True, new_trajectory_training=False, new_deployment=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False)


def cost_diagnostics(receipts):
    rows = []
    for model in ('frozen_teacher', 'adaptive'):
        for region in ('frozen_teacher', 'adaptive', 'shared_tree'):
            cells = [cell['scores'] for receipt in receipts for cell in receipt['fixed_region_cost_diagnostics']
                if cell['model'] == model and cell['region'] == region]
            if len(cells) != 18:
                raise ValueError('Every fixed model/region combination needs all 18 held jobs')
            actual = np.sum([c['realized_selected_known_moments'] for c in cells], axis=0)
            estimate = np.sum([c['predicted_on_selected_known_moments'] for c in cells], axis=0)
            ratios = {name: float(a/b) if b > 0 else None
                for name, a, b in zip(('B', 'H', 'R', 'ER', 'EH'), actual, estimate)}
            rows.append(dict(model=model, region=region, known_actual_over_predicted=ratios,
                known_selected_rows=sum(c['selected_known'] for c in cells),
                unknown_selected_rows=sum(c['selected_unknown'] for c in cells),
                role='post_choice_pooled_repeated_row_mass_diagnostic_not_CI_or_calibration'))
    return rows


def validate_solver_metadata(meta):
    n = meta['completed_queries']
    if n <= 0 or meta['bindings']['seconds'] != .25:
        raise ValueError('Complete original-budget solver metadata required')
    for policy in ('joint_matched', 'joint_free'):
        counts = meta['reason_counts'][policy]
        if (any(v < 0 for v in counts.values()) or sum(counts.values()) != n
                or not 0 <= meta['nonoptimal_queries'][policy] <= n
                or not 0 <= meta['changed_queries'][policy] <= n):
            raise ValueError('Every query must retain its actual solver outcome')


def solver_diagnostics(receipts):
    rows = []
    for receipt in receipts:
        job = receipt['job']['key']
        for bank, base, suffix in (('frozen_teacher', run.run.teacher.PRIVATE, 'record_excess'),
                                   ('adaptive', PRIVATE, 'adaptive_selection')):
            path = base / 'allocation' / (job + '_' + suffix + '.npz')
            with np.load(path, allow_pickle=False) as packet:
                meta = source.json.loads(str(packet['metadata_json']))
            validate_solver_metadata(meta)
            rows.append(dict(job=job, bank=bank, queries=meta['completed_queries'],
                reason_counts=meta['reason_counts'], nonoptimal_queries=meta['nonoptimal_queries'],
                changed_queries=meta['changed_queries'], allocation=source.artifact(path)))
    if len(rows) != 36:
        raise ValueError('Both complete18-bank solver reports required')
    return dict(rows=rows, actual_repeated_query_workload_not_IID_data=True,
        zero_count_or_analytic_optimum_separate_from_MILP=True,
        original_point_and_shared_controls_not_reoptimized=True,
        bitwise_MILP_reoptimization_claimed=False)


def figures(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    public, private = PUBLIC / 'figures', PRIVATE / 'figure_previews'
    public.mkdir(exist_ok=True)
    private.mkdir(exist_ok=True)
    plt.rcParams.update({'svg.fonttype': 'none', 'font.size': 9})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, metric, title in zip(axes, ('known_pp', 'completion_lower_pp'), ('Known labels', 'Direct completion lower')):
        for i, control in enumerate(('point', 'shared_tree', 'frozen_teacher')):
            r = summary['paired'][control]['joint_free'][metric]
            if r['CI95'] is not None:
                ax.errorbar(r['point'], i, xerr=[[r['point']-r['CI95'][0]], [r['CI95'][1]-r['point']]], fmt='o', capsize=4)
            else:
                ax.text(0, i, 'not estimable')
        ax.set_yticks(range(3), ['Vs point neural', 'Vs shared tree', 'Vs frozen teacher'])
        ax.axvline(0, color='.5', lw=1)
        ax.set_ylim(2.5, -.5)
        ax.set_title(title)
        ax.set_xlabel('Adaptive joint utility contrast (pp)')
    fig.suptitle('23 exposed source groups; three seeds; grouped 95% CI, not confirmation', fontsize=10)
    fig.tight_layout()
    fig.savefig(public / 'paired_effects.svg')
    fig.savefig(private / 'paired_effects.png', dpi=160)
    plt.close(fig)


def main():
    summary = source.read(PUBLIC / 'summary.json')
    verify = source.read(PUBLIC / 'verification.json')
    freeze = source.read(PUBLIC / 'training_freeze.json')
    refit = source.read(PUBLIC / 'first_full_refits.json')
    if (not verify['all54neural_cost_arrays_and_policy_blocks_exact']
            or not verify['all_three_cached_parent_policy_blocks_exact']
            or verify['summary_sha256'] != source.sha(PUBLIC / 'summary.json')
            or freeze['complete_new_adaptive_cost_heads'] != 18
            or not refit['frozen_teacher_matches_original']
            or not refit['adaptive_refit_and_selection_stream_exact']):
        raise ValueError('Complete all fits, full controls and Hreplay before conclusions')
    receipts = [source.read(source.checked(r)) for r in summary['group_receipts']]
    records = [r for receipt in receipts for r in receipt['rows']]
    if len(records) != 138 or len({r['site'] for r in records}) != 23:
        raise ValueError('Retain every supported source view')
    decision = scientific_decision(summary)
    diagnostics = cost_diagnostics(receipts)
    solver = solver_diagnostics(receipts)
    source.once(PUBLIC / 'decision.json', source.stable(decision))
    source.once(PUBLIC / 'cost_diagnostics.json', source.stable(diagnostics))
    source.once(PUBLIC / 'solver_diagnostics.json', source.stable(solver))
    source.once(PUBLIC / 'secondary_metrics.json', source.stable(secondary_intervals(records, source.read(run.run.teacher.PARENT.CONFIG))))
    zero = {arm: dict(known_zero_CV_rows=sum(r['zero_CV_diagnostic'][arm]['known_zero_CV_rows'] for r in records),
        positive_error_on_zero_CV_rows=sum(r['zero_CV_diagnostic'][arm]['positive_error_on_zero_CV_rows'] for r in records))
        for arm in run.ARMS}
    source.once(PUBLIC / 'zero_CV_diagnostic.json', source.stable(dict(repeated_views_not_unique_rows=zero,
        role='post_choice_supplementary_not_new_safety_gate')))
    source.once(PUBLIC / 'runtime_summary.json', dict(phases={phase: source.read(PUBLIC / (phase + '_execution.json'))
        for phase in ('pilot', 'train', 'evaluate', 'verify')}, primary_new_neural_cost_heads=18,
        primary_optimizer_updates=36000, nominal_pilot_optimizer_updates=1500,
        full_refit_optimizer_updates=4000, new_trajectory_training=False,
        phase_seconds_measurement='time.monotonic_elapsed_not_CPU_time_or_guaranteed_wall_clock_interval',
        cached_frozen_teacher_controls=18, cached_original_point_controls=18))
    table = ['|Adaptive joint-free versus|Known utility pp [95%CI]|Direct completion-lower pp [95%CI]|', '|---|---:|---:|']
    for arm in ('point', 'shared_tree', 'frozen_teacher'):
        row = summary['paired'][arm]['joint_free']
        table.append(f'|{arm}|{interval(row["known_pp"])}|{interval(row["completion_lower_pp"])}|')
    risk_table = ['|Joint-free arm|Support /348|All violations|Easy violations|Undefined easy views|Worst signed easy-net %|', '|---|---:|---:|---:|---:|---:|']
    for arm in run.ARMS:
        row = summary['safety'][arm]['joint_free']
        easy = row['worst_known_easy_net_degradation_percent']
        easy = 'not_estimable' if easy is None else f'{easy:.6f}'
        risk_table.append(f'|{arm}|{row["supported_record_views"]}|{row["known_all_risk_violations"]}|{row["known_easy_risk_violations"]}|{row["undefined_easy_views"]}|{easy}|')
    solver_table = ['|New bank|Policy|Repeated queries|Nonoptimal/fallback queries|', '|---|---|---:|---:|']
    for bank in ('frozen_teacher', 'adaptive'):
        cells = [r for r in solver['rows'] if r['bank'] == bank]
        for policy in ('joint_matched', 'joint_free'):
            total = sum(r['queries'] for r in cells)
            nonoptimal = sum(r['nonoptimal_queries'][policy] for r in cells)
            solver_table.append(f'|{bank}|{policy}|{total}|{nonoptimal} ({100*nonoptimal/total:.2f}%)|')
    text_once(PUBLIC / 'readout.md', '# Current-Selection Neural Cost Training\n\n'
        '18 fresh neural cost heads, 36,000 primary updates, three actual seeds; same models, '
        'inputs, initializations, sampled queries and budget as frozen-teacher controls. '
        'The only loss change is detached current independent selection instead of fixed teacher selection. '
        'Both complete model banks froze before any new H readout. All 54 neural prediction arrays and '
        'all policy blocks replay; all three cached control decisions match the verified parent. '
        'Tree costs were not freshly inferred; cached allocation replay is not bitwise MILP reoptimization.\n\n'
        + '\n'.join(table) + '\n\n' + '\n'.join(risk_table) + '\n\n' + '\n'.join(solver_table) + '\n\n'
        + f'Supported-source mechanism gate:{decision["supported_source_mechanism_gate"]}. '
        'New deployment: False. Original 24-group primary: not_estimable, not passed.\n\n'
        'Current selection is causal but the record loss uses C supervision. Its selection is independent-agent, '
        'not the final joint choice; sampled record multisets are not unbiased full-video loss estimates. '
        'No gradient through action choice; this is not SPO+ or a new safety theorem. '
        'A no-switch batch has no extra record term, not an observed safety pass.\n\n'
        'solver_diagnostics.json retains actual per-query reason counts for both new banks, '
        'including time/status limits and postcheck fallbacks. Feasibility replay does not '
        'mean every decision was an optimal MILP solution. No solver budget was increased.\n\n'
        'All 348 repeated recording views, four policies, original 2% positive-harm/support and signed-net '
        'condition remains. Empty same-count utility stays NA. Known-label and bounded-completion '
        'estimands are separate. Cost ratios/zero-CVcounts are supplementary diagnostics, not model selection. '
        'All 23 supported groups are exposed development, not independent tests or verified IID scenes. '
        'Reserved 20 Moving groups, DUT and DroneCrowd confirmation remain closed.\n\n'
        'No new trajectory model, metric/seconds/physical-safety/true3D/foundation/human-gold claim. '
        'Native 8/12 supplied annotation order, not verified sensor-as-of time. Stage5C/SMC off.\n')
    text_once(PUBLIC / 'gates.md', '# Fixed Scientific Gates\n\n' + '\n'.join(f'- {k}: {v}' for k, v in decision.items())
        + '\n\nExecution, development efficacy, complete risk support and independent confirmation are distinct.\n')
    lines = ['# Every Supported Source', '', '|Source|Adaptive minus point lower pp|Minus shared lower pp|Minus frozen lower pp|', '|---|---:|---:|---:|']
    for site in sorted({r['site'] for r in records}):
        values = []
        for arm in ('point', 'shared_tree', 'frozen_teacher'):
            row = summary['paired'][arm]['joint_free']['completion_lower_pp']
            values.append('not_estimable' if 'by_source_group' not in row else f'{row["by_source_group"][site]:+.6f}')
        lines.append('|' + site + '|' + '|'.join(values) + '|')
    text_once(PUBLIC / 'per_source.md', '\n'.join(lines) + '\n')
    text_once(PUBLIC / 'model_data_card.md', '# Model And Data Card\n\n'
        'CostHead width64,28229parameters,428causal input fields (380local+48zeros). '
        'Original Cfit-only normalizers, seeds17/29/43,2000updates per head, CPU4/interop1/workers0. '
        'Record IDs and current detached actions are training-loss metadata, not network inputs. '
        'Future endpoint/mask/easy label never enters inference. Frozen9EqMotion trajectory proposals '
        'and physical floor are unchanged. No scene-image or JEPA training.\n\n'
        '195286past-eligible SOURCE rows,64admitted recordings;23supported source groups. '
        'Original P/C/Hsource exclusions remain. Repeated/overlapping windows are not IID samples. '
        'Partial labels retain conservative bounds; unavailable full-roster FDE stays NA. '
        'No independent confirmation, public cold-start reproduction proof, deployment, Stage5C or SMC.\n')
    text_once(PUBLIC / 'failure_analysis.md', '# What The Current-Selection Repair Establishes\n\n'
        f'Primary adaptive-minus-frozen positive:{decision["primary_vs_frozen_teacher_positive"]}. '
        f'Strong shared control positive:{decision["strong_shared_tree_contrast_positive"]}. '
        f'Complete original risk:{decision["full_original_risk_pass"]}.\n\n'
        'The C-only trigger showed a frozen-teacher head fitting the teacher region while underpredicting '
        'harm on its own region. It did not prove Hgeneralization. Read cost_diagnostics.json for '
        'both models on all three fixed Hregions; ratios are pooled repeated-row mass, not source means or CIs. '
        'A closer cost ratio without complete utility/risk support is not a successful replacement. '
        'The detached independent training mask and joint deployment mask still differ. Finite reference '
        'events, partial labels, mini-batch aggregation and exposed sources limit inference. '
        'Do not change thresholds or select a head on these Hresults. The original24primary and independent '
        'confirmation remain missing regardless of an observed development contrast. Main v9 PDF not recompiled.\n')
    figures(summary)
    print(source.json.dumps(dict(decision=decision, primary=summary['paired']['frozen_teacher']['joint_free']), sort_keys=True))


if __name__ == '__main__':
    main()
