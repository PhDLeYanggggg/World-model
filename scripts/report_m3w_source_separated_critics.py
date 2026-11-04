"""Report all completed source-separated neural fits and exact nested-C replay."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_source_separated_critics as run
from scripts.report_m3w_moving_harm_refinement import text_once
from scripts.report_m3w_moving_amplitude_cost import interval

source, PUBLIC = run.source, run.PUBLIC


def per_source_value(rows, site, kind, metric):
    values = [r['nonempty_matched'][kind]['contrast'][metric] for r in rows if r['site'] == site]
    if not values or any(v is None for v in values):
        return 'NA'
    return str(float(source.np.mean(values)))


def pooled_cost_diagnostics(groups):
    totals = {}
    for group in groups:
        for cell in group['cost_cells']:
            key = cell['model']+'__'+cell['region']
            values = totals.setdefault(key, dict(actual=source.np.zeros(5), predicted=source.np.zeros(5), selected_known=0))
            score = cell['scores']
            values['actual'] += score['realized_selected_known_moments']
            values['predicted'] += score['predicted_on_selected_known_moments']
            values['selected_known'] += score['selected_known']
    return {k: dict(known_actual_moments=v['actual'].tolist(),
        predicted_on_exact_same_known_rows=v['predicted'].tolist(), selected_known_repeated_rows=v['selected_known'],
        actual_over_predicted_H=float(v['actual'][1]/v['predicted'][1]) if v['predicted'][1] > 0 else None,
        actual_over_predicted_EH=float(v['actual'][4]/v['predicted'][4]) if v['predicted'][4] > 0 else None)
        for k, v in totals.items()}


def figures(fits, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    np = source.np
    traces = [r['trace'] for r in fits]
    steps = np.array([r['step'] for r in traces[0]])
    if any([r['step'] for r in trace] != steps.tolist() for trace in traces):
        raise ValueError('Plot every complete registered training trace on its actual update grid')
    losses = np.array([[r['total'] for r in trace] for trace in traces])
    plt.rcParams.update({'svg.fonttype': 'none', 'font.size': 9})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    lo, mid, hi = np.quantile(losses, [.1, .5, .9], axis=0)
    axes[0].fill_between(steps, lo, hi, alpha=.2, color='#35719a', label='10th-90th percentile of fits')
    axes[0].plot(steps, mid, color='#35719a', label='Median (108 fits)')
    axes[0].set_xlabel('Optimizer updates per model')
    axes[0].set_ylabel('Own-part normalized training objective')
    axes[0].legend(fontsize=7)
    axes[0].set_title('Fitting loss is not validation accuracy', fontsize=9)
    for y, (kind, metric, label) in enumerate((
            ('same_data', 'known_pp', 'Same data: known'),
            ('same_data', 'completion_lower_pp', 'Same data: direct lower'),
            ('disjoint_data', 'known_pp', 'Disjoint data: known'),
            ('disjoint_data', 'completion_lower_pp', 'Disjoint data: direct lower'))):
        r = summary['nonempty_matched'][kind][metric]
        if r['CI95'] is None:
            axes[1].text(.03, y, 'NA: a required view is unavailable', fontsize=7,
                transform=axes[1].get_yaxis_transform())
        else:
            axes[1].plot(r['CI95'], [y, y], color='#4f8c70', linewidth=1.5)
            axes[1].plot(r['point'], y, 'o', color='#4f8c70', markersize=4)
    axes[1].set_yticks(range(4), ['Same data: known', 'Same data: direct lower', 'Disjoint data: known', 'Disjoint data: direct lower'], fontsize=8)
    axes[1].set_ylim(3.5, -.5)
    estimable = any(v['CI95'] is not None for c in summary['nonempty_matched'].values() for v in c.values())
    if estimable:
        axes[1].axvline(0, color='.5', lw=1)
        axes[1].set_xlabel('Veto minus count/alpha-matched control (pp)')
        axes[1].set_title('Nested C development, not independent test', fontsize=9)
    else:
        axes[1].set_xlim(0, 1); axes[1].set_xticks([])
        axes[1].set_xlabel('Primary not estimable; no replacement CI', fontsize=8)
        axes[1].set_title('Required empty views remain in the primary', fontsize=9)
    fig.tight_layout()
    public, private = PUBLIC / 'figures', run.PRIVATE / 'figure_previews'
    public.mkdir(exist_ok=True); private.mkdir(exist_ok=True)
    fig.savefig(public / 'training_and_validation.svg', metadata={'Date': None})
    fig.savefig(private / 'training_and_validation.png', dpi=160)
    plt.close(fig)


def main():
    summary = source.read(PUBLIC / 'analysis.json')
    verified = source.read(PUBLIC / 'verification.json')
    if (not verified['all216_unique_nested_C_predictions_and_all108policy_blocks_exact']
            or not verified['all1620_cost_cells_exact']
            or verified['analysis_sha256'] != source.sha(PUBLIC / 'analysis.json')):
        raise ValueError('Real completed training and separate exact inference replay required')
    freeze = source.read(PUBLIC / 'training_freeze.json')
    fits = [source.read(source.checked(r)) for r in freeze['fits']]
    groups = [source.read(source.checked(r)) for r in summary['group_receipts']]
    if len(fits) != 108 or len(groups) != 108 or sum(r['steps'] for r in fits) != 216000:
        raise ValueError('Complete prescribed population and model budget required')
    receipts = {phase: source.read(PUBLIC / (phase+'_execution.json')) for phase in ('pilot', 'train', 'evaluate', 'verify')}
    interrupted = source.read(PUBLIC / 'interruption_20261104.json') if (PUBLIC / 'interruption_20261104.json').exists() else None
    primary_fitter_seconds = sum(r['seconds'] for r in fits)
    cfg = source.read(run.previous.teacher.PARENT.CONFIG)
    rows = [r for group in groups for r in group['rows']]
    sites = sorted({r['site'] for r in rows})
    decomposition = {}
    for kind in ('same_data', 'disjoint_data'):
        def benefit_lost(r):
            h, u = r['matched_known_harm_reduction_pp'][kind], r['secondary_full_strategy'][kind]['known_pp']
            return None if h is None or u is None else h-u
        decomposition[kind] = source.group_interval(rows, benefit_lost, sites, cfg)
    source.once(PUBLIC / 'benefit_harm_decomposition.json', source.stable(dict(
        full_strategy_known_benefit_lost_pp=decomposition,
        identity='utility = benefit_change + harm_reduction; identical full-strategy denominators only',
        role='posthoc_algebra_not_a_new_policy_or_replacement_primary')))
    source.once(PUBLIC / 'pooled_cost_diagnostics.json', source.stable(dict(cells=pooled_cost_diagnostics(groups),
        role='post_readout_repeated_rows_descriptive_not_source_macro_CI_or_calibration',
        no_new_policy=True, known_rows_identical_between_predicted_and_actual_mass=True)))
    table = ['|Contrast, veto minus its query count/alpha matched control|Known pp [95% CI]|Direct lower pp [95% CI]|Known harm reduction pp, full denominator|',
        '|---|---:|---:|---:|']
    for kind in ('same_data', 'disjoint_data'):
        c = summary['nonempty_matched'][kind]
        table.append(f"|{kind}|{interval(c['known_pp'])}|{interval(c['completion_lower_pp'])}|{interval(summary['matched_known_harm_reduction_pp'][kind])}|")
    risk = ['|Arm|Supported /696|Known all risk violations|Known easy risk violations|Easy risk unavailable|Worst defined signed easy-net %|Source-macro switch %|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in run.ARMS:
        r = summary['safety'][arm]
        worst = r['worst_defined_signed_easy_net_percent']
        risk.append(f"|{arm}|{r['supported_record_views']}|{r['known_all_risk_violations']}|{r['known_easy_risk_violations']}|{r['undefined_easy_views']}|{worst if worst is not None else 'NA'}|{interval(summary['switch_percent'][arm])}|")
    text_once(PUBLIC / 'readout.md', '# Source-Separated Neural Critics: Complete Nested C Readout\n\n'
        '**fresh_run:** 108 real Torch cost heads, 216,000 saved primary optimizer updates; '
        'not a new trajectory model, JEPA or generative run. All full-budget weights froze '
        'before new nested C validation scores. Both orientations and all three original '
        'producer seeds are retained. **cached_verified:** source admission, original '
        'forecast checkpoints, outer roles and existing inner-source mapping. '
        '**not_run:** new H readout, independent confirmation, deployment and paper submission.\n\n'
        'The user explicitly kept the original rules. There is no analytic identity '
        'certification. The original 2% risk limit, easy-net and support requirements, unavailable '
        'views and old failures remain. The original 24-source primary is still '
        'not_estimable, not passed.\n\n'
        '## Controlled Model Change\n\n'
        'Each single C part supplies two 28,229-parameter cost heads. Means, scales, '
        'target RMS and bias use only that own fitting part. For a held C part, '
        'the other two provide proposer/risk roles in both orientations. Same-data '
        'critics use the other cost seed on the proposer fitting part. Disjoint-data '
        'critics use different source groups, but are not statistically independent: '
        'forecaster, architecture and dataset family are shared. Less fitting data '
        'is a cost of this design. Vetoes retain an original action or reject it; '
        'no amplitude reselection or new threshold search. Matching counts AND '
        'amplitude histograms removes the prior count-only severity-composition issue, '
        'not displacement magnitude differences.\n\n'
        '## All Primary And Secondary Contrasts\n\n'+'\n'.join(table)+'\n\n'
        'Primary is the disjoint-data direct completion-lower contrast on causal '
        'nonempty matched queries. Known utility, same-data comparison, complete '
        'strategy and harm are secondary; none can replace a missing/negative primary. '
        'Required empty source views remain NA. Where estimable, 3,000 bootstrap '
        'resamples use 23 source units after producer/seed/orientation averaging; '
        '276 repeated source views and 696 record views are not independent sample sizes. These are exposed '
        'nested development sources, not a recovered final test.\n\n'
        'The outcome is the registered available-label ADE. Partially labelled '
        'rows are not silently treated as complete 12-step trajectories; endpoint '
        'errors remain available-endpoint diagnostics. The completion contrast '
        'does not establish metric, seconds-level or physical-safety guarantees.\n\n'
        '## Unchanged Full-Policy Risk\n\n'+'\n'.join(risk)+'\n\n'
        'Whole-record risk uses every required view, not just nonempty utility '
        'queries. A no-intervention/zero-reference view does not automatically '
        'pass. A positive model contrast alone does not authorize deployment.\n\n'
        '## Runtime And Reproducibility\n\n'
        + '\n'.join(f"- {phase}: PID {r['pid']}, {r['seconds']:.2f} s, peak RSS {r['peak_RSS_bytes']/1e9:.3f} GB." for phase, r in receipts.items())
        + '\n\nAll six pilot resumed/direct scientific states and the first full-budget '
        'refit agree. Separate replay reproduces 216 unique C prediction arrays, '
        '108 orientation policy blocks, 1,620 cost cells and every within-query '
        'count/amplitude histogram. Exact replay verifies engineering, not '
        'independent scientific confirmation. No CUDA/MPS claim: native arm64, four CPU '
        'threads, interop one, workers zero. Native 8/12 annotation order is not verified '
        'time, metric, true 3D, foundation or human-gold. Stage5C/SMC remain off.\n\n'
        '[Interrupted-run time accounting](runtime_recovery.md). '
        'The training receipt is the terminal resumed attempt, not the full wall span.\n')
    if interrupted:
        text_once(PUBLIC / 'runtime_recovery.md', '# Interrupted Training: Time Accounting\n\n'
            f"The original PID{interrupted['previous_pid']} and session{interrupted['previous_session']} "
            f"were confirmed absent at{interrupted['observation_UTC']}. No exit code or cause "
            'was observed.71complete models and the step100partial state were retained. '
            'Same-binding resume continued the saved optimizer/sampler/RNG; no model, '
            'normalizer, risk rule, threshold or update budget changed.\n\n'
            f"The train-execution receipt describes only terminal resumed PID{receipts['train']['pid']} "
            f"and its{receipts['train']['seconds']:.2f}s elapsed time. It is NOT the full "
            f"interrupted training wall span. Sum of saved primary fitter times:{primary_fitter_seconds:.2f}s. "
            'The original last-heartbeat elapsed time is6440.0074s, not an exact terminal '
            'duration. Do not sum overlapping timing definitions or count the unobserved '
            'wall gap as compute. Pilot1200and first full-refit2000updates are extra. '
            'Completed primary checkpoints contain exactly216000prescribed updates; '
            'any unsaved work before termination is unknown, not asserted as zero.\n')
    text_once(PUBLIC / 'per_source.md', '# Every Required Source: Nested Development Only\n\n'
        '|Source|Same-data known pp|Same-data lower pp|Disjoint-data known pp|Disjoint-data lower pp|\n'
        '|---|---:|---:|---:|---:|\n'
        + '\n'.join('|'+site+'|'+'|'.join(per_source_value(rows, site, k, m)
            for k in ('same_data', 'disjoint_data') for m in ('known_pp', 'completion_lower_pp'))+'|' for site in sites)
        + '\n\nNA means a required view was unavailable; do not drop it or report a complete CI.\n')
    gate = source.read(PUBLIC / 'decision.json')
    text_once(PUBLIC / 'gates.md', '# Original Rules, No Promotion\n\n'
        f"Fixed combined mechanism-and-full-risk pass: **{gate['source_excluded_mechanism_and_original_risk_pass']}**.\n\n"
        'A nested model can improve decision diagnostics without meeting the original '
        'full support/risk contract. No analytic identity rule was added. No old '
        'failure became a pass. No deployment, Stage5C, SMC, independent test or '
        'submission readiness is claimed.\n')
    source.once(PUBLIC / 'runtime_summary.json', source.stable(dict(phases=receipts,
        fresh_primary_fits=108, primary_optimizer_updates=216000, pilot_updates=1200,
        first_full_refit_updates=2000, unique_validation_arrays_each_inference=216,
        required_source_views=276, required_record_views=696, bootstrap_source_units=23,
        original_risk_rules_unchanged=True, primary_saved_fitter_seconds=primary_fitter_seconds,
        train_execution_is_terminal_resumed_attempt_only=interrupted is not None,
        previous_training_exit_code_observed=False if interrupted else None,
        source=source.artifact(PUBLIC / 'analysis.json'))))
    figures(fits, summary)
    print(source.json.dumps(dict(report=str((PUBLIC / 'readout.md').relative_to(ROOT)), decision=gate), indent=2))


if __name__ == '__main__': main()
