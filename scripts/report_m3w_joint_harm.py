"""Report frozen joint-event models without changing their registered decision."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_joint_harm as run
from scripts.report_m3w_moving_harm_refinement import text_once
from scripts.report_m3w_moving_amplitude_cost import interval
from scripts.report_m3w_source_separated_critics import per_source_value, pooled_cost_diagnostics

source, PUBLIC = run.source, run.PUBLIC


def require_complete(freeze, audit, verification, analysis_hash, fits, groups):
    if (freeze['complete_heads'] != 108 or not freeze['all_before_validation']
            or not audit['full108_complete'] or audit['complete_matched_pairs'] != 54
            or not verification['all108_policy_blocks_and_prediction_hashes_exact']
            or verification['analysis_sha256'] != analysis_hash
            or len(fits) != 108 or len(groups) != 108
            or sum(f['steps'] for f in fits) != 216000):
        raise ValueError('Complete frozen training, own-source audit and exact separate replay required')
    if any(f['steps'] != 2000 or f['parameters'] != 28489 for f in fits):
        raise ValueError('All registered models must receive their full matched budget')
    if sum(len(g['rows']) for g in groups) != 276:
        raise ValueError('Do not drop unavailable source views')


def outcome(analysis, decision):
    primary = analysis['nonempty_matched']['event_supervised']['completion_lower_pp']
    risk = analysis['safety']['event_supervised_veto']['full_original_risk_pass']
    positive = primary['CI95'] is not None and primary['CI95'][0] > 0
    if bool(positive and risk) != decision['original_rules_pass']:
        raise ValueError('Report cannot override the frozen scientific decision')
    if primary['CI95'] is None:
        primary_text = 'The complete primary is not estimable, not passed.'
    elif positive:
        primary_text = 'The registered primary lower confidence limit is positive.'
    else:
        primary_text = 'The registered primary does not have a positive lower confidence limit.'
    return primary_text + (' All original full-policy risk/support views pass.' if risk else
        ' Original full-policy risk/support requirements fail.')


def loss_arrays(fits, arm):
    traces = [f['trace'] for f in fits if f['arm'] == arm]
    if len(traces) != 54:
        raise ValueError('Plot all 54 fits per arm, not a selected best seed')
    steps = [r['step'] for r in traces[0]]
    if any([r['step'] for r in t] != steps for t in traces):
        raise ValueError('Training traces require an identical actual update grid')
    # The total losses differ by definition; compare the shared term separately.
    common = source.np.array([[r['regression']+.5*r['absolute_gain'] for r in t] for t in traces])
    occurrence = source.np.array([[r['occurrence'] for r in t] for t in traces])
    return source.np.array(steps), common, occurrence


def risk_diagnostics(groups):
    rows = [r for g in groups for r in g['rows']]
    result = {}
    for arm in run.POLICIES:
        views = [v for r in rows for v in r['policies'][arm]['recordings'].values()]
        defined = [100*v['known_easy_net_degradation'] for v in views
            if v['known_easy_net_degradation'] is not None]
        result[arm] = dict(required_record_views=len(views),
            worst_defined_signed_easy_net_percent=max(defined) if defined else None,
            undefined_full_known_easy_net_views=sum(v['known_easy_net_degradation'] is None for v in views),
            no_intervention_views=sum(v['selected'] == 0 for v in views))
    return result


def figures(fits, analysis):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype': 'none', 'svg.hashsalt': 'm3w-joint-harm-v1', 'font.size': 9})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for arm, color in zip(run.api.ARMS, ('#216a91', '#a24d4b')):
        steps, common, _ = loss_arrays(fits, arm)
        lo, mid, hi = source.np.quantile(common, [.1, .5, .9], axis=0)
        axes[0].fill_between(steps, lo, hi, alpha=.13, color=color)
        axes[0].plot(steps, mid, color=color, label=arm.replace('_', ' '))
    axes[0].set(xlabel='Optimizer updates per model', ylabel='Shared fitting objective',
        title='54 fits per arm: median and 10th-90th percentiles')
    axes[0].legend(fontsize=7)
    for y, arm in enumerate(run.api.ARMS):
        result = analysis['nonempty_matched'][arm]['completion_lower_pp']
        if result['CI95'] is None:
            axes[1].text(.02, y, 'NA: required empty views retained',
                transform=axes[1].get_yaxis_transform(), fontsize=8)
        else:
            axes[1].plot(result['CI95'], [y, y], color='#216a91')
            axes[1].plot(result['point'], y, 'o', color='#216a91')
    axes[1].set_yticks([0, 1], ['Product moment', 'Event supervised'])
    axes[1].set_ylim(1.5, -.5)
    if all(analysis['nonempty_matched'][a]['completion_lower_pp']['CI95'] is None for a in run.api.ARMS):
        axes[1].set_xlim(0, 1); axes[1].set_xticks([])
        axes[1].set_xlabel('No secondary result substitutes for the primary')
    else:
        axes[1].axvline(0, color='.5', lw=1)
        axes[1].set_xlabel('Veto minus matched control, direct lower (pp)')
    axes[1].set_title('Nested C development, not independent confirmation')
    fig.tight_layout()
    public, private = PUBLIC / 'figures', run.PRIVATE / 'figure_previews'
    public.mkdir(exist_ok=True); private.mkdir(exist_ok=True)
    fig.savefig(public / 'training_and_primary.svg', metadata={'Date': None})
    fig.savefig(private / 'training_and_primary.png', dpi=160)
    plt.close(fig)


def main():
    analysis = source.read(PUBLIC / 'analysis.json')
    freeze = source.read(PUBLIC / 'training_freeze.json')
    audit = source.read(PUBLIC / 'fit_audit.json')
    verification = source.read(PUBLIC / 'verification.json')
    decision = source.read(PUBLIC / 'decision.json')
    fits = [source.read(source.checked(r)) for r in freeze['fits']]
    groups = [source.read(source.checked(r)) for r in analysis['group_receipts']]
    require_complete(freeze, audit, verification, source.sha(PUBLIC / 'analysis.json'), fits, groups)
    result = outcome(analysis, decision)
    rows = [r for g in groups for r in g['rows']]
    phases = {p: source.read(PUBLIC / (p+'_execution.json')) for p in ('pilot', 'train', 'evaluate', 'verify')}
    costs = pooled_cost_diagnostics(groups)
    source.once(PUBLIC / 'pooled_cost_diagnostics.json', source.stable(dict(cells=costs,
        role='post_readout_repeated_rows_descriptive_not_source_macro_CI_or_calibration',
        no_new_policy=True, known_rows_identical_between_predicted_and_actual_mass=True)))
    source.once(PUBLIC / 'risk_diagnostics.json', dict(
        values=risk_diagnostics(groups), counts_overlap_not_additive=True,
        role='posthoc_description_original_policy_not_new_gate'))
    primary = ['|Arm vs own query-count/amplitude-matched control|Known pp [95% CI]|Direct lower pp [95% CI]|Unavailable /276|',
        '|---|---:|---:|---:|']
    secondary = ['|Full-strategy secondary contrast|Known pp [95% CI]|Direct lower pp [95% CI]|', '|---|---:|---:|']
    risk = ['|Policy|Supported /696|Known all-risk violations|Known easy-risk violations|Undefined selected easy risk|Switch % [95% CI]|',
        '|---|---:|---:|---:|---:|---:|']
    for arm in run.api.ARMS:
        m = analysis['nonempty_matched'][arm]
        primary.append(f"|{arm}|{interval(m['known_pp'])}|{interval(m['completion_lower_pp'])}|{analysis['unavailable_primary_views'][arm]}|")
        m = analysis['secondary_full_strategy'][arm]
        secondary.append(f"|{arm} minus own matched control|{interval(m['known_pp'])}|{interval(m['completion_lower_pp'])}|")
    m = analysis['event_minus_product_full']
    secondary.append(f"|Event veto minus product veto|{interval(m['known_pp'])}|{interval(m['completion_lower_pp'])}|")
    for arm in run.POLICIES:
        r = analysis['safety'][arm]
        risk.append(f"|{arm}|{r['supported']}|{r['known_all_violations']}|{r['known_easy_violations']}|{r['undefined_selected_easy']}|{interval(analysis['switch_percent'][arm])}|")
    diagnostic = ['|Same-row diagnostic (positive means error reduction)|All legal known actions [95% CI]|Fixed proposer choices [95% CI]|',
        '|---|---:|---:|']
    for name in analysis['cost_diagnostic']['all_legal']:
        diagnostic.append(f"|{name}|{interval(analysis['cost_diagnostic']['all_legal'][name])}|{interval(analysis['cost_diagnostic']['fixed_proposal'][name])}|")
    text_once(PUBLIC / 'readout.md', '# Matched Neural Joint-Harm Study\n\n'
        '**fresh_run:** 108 real Torch heads, 216,000 saved primary updates, complete nested-C evaluation and separate exact inference replay. '
        '**cached_verified:** parent forecasters, original roles, admission map, own-source preprocessing and frozen proposers. '
        '**not_run:** new H readout, independent confirmation, deployment and submission.\n\n'
        + result + '\n\n## Model Change\n\n'
        'Both arms have 28,489 parameters, identical initial weights, own-part preprocessing, query draws, '
        'optimizer settings and update budgets. The event arm adds exclusive-event cross entropy plus '
        'positive-event conditional-magnitude loss to the same five-moment and net-gain objective. '
        'This identifies the combined auxiliary treatment, not each component separately. '
        'The product-moment control has the same factorized architecture; its unsupervised probability '
        'factors are not a calibrated probability baseline. Fitting-only event priors are also reported. '
        'Neither arm is a new trajectory forecaster or a generative world model.\n\n'
        'Each critic and all learned preprocessing exclude its held validation sources. A separately '
        'fitted frozen proposer determines the action; the critic may retain or reject it, not choose '
        'another amplitude. All 108 new heads froze before the new validation readout.\n\n'
        '## Registered Primary\n\n'+'\n'.join(primary)+'\n\n'
        'The event-arm direct completion-lower contrast is primary. Matching fixes each query\'s '
        'intervention count and amplitude histogram, not displacement magnitudes. Empty required '
        'source views remain NA; no switch is forced. Original 24-group primary remains '
        'not_estimable_not_passed. The 23 available source units are exposed development sources, '
        'not recovered independent test data.\n\n'
        '## Secondary Results\n\n'+'\n'.join(secondary)+'\n\n'
        'Event-versus-product full policies have different intervention rates and are not a '
        'matched-ranking contrast. A positive secondary does not replace a missing primary. '
        'Intervals use 3,000 source-group resamples after context/seed/orientation averaging. '
        'The 276 repeated source views and 696 record views are not independent sample sizes.\n\n'
        '## Original Full-Policy Safety\n\n'+'\n'.join(risk)+'\n\n'
        'Original 2% risk, easy-net and support requirements are unchanged. Identity certification '
        'was not added. Counts overlap; zero-reference and no-intervention views do not automatically pass.\n\n'
        '## Same-Row Mechanism Checks\n\n'+'\n'.join(diagnostic)+'\n\n'
        'MSE reductions are product minus event; Brier gain over fit prior uses the event model and '
        'its own fitting-source prior. These are query-balanced available-label diagnostics, '
        'not calibration guarantees or proof of decision utility. Unknown futures are never '
        'negative occurrence labels; their original conservative loss remains. Partial labels '
        'keep the original available-label ADE meaning, not imputed full trajectories.\n\n'
        '## Runtime And Reproduction\n\n'
        + '\n'.join(f"- {p}: PID{r['pid']}, {r['seconds']:.2f}s, peak RSS {r['peak_RSS_bytes']/1e9:.3f}GB." for p, r in phases.items())
        + '\n\nSix real resumed/direct pilot states and the first full-budget refit agree exactly. '
        'All 54 paired initializations, sampling streams and preprocessing states match. The initial '
        'training attempt exited after a completed model/refit because an aggregate prior ndarray '
        'was not JSON serializable. A narrow wrapper fixed receipt serialization and resumed '
        'unchanged checkpoints. The terminal train receipt is the resumed attempt, not total '
        'wall span; see runtime_fix.md.\n\n'
        'Use `.venv-pytorch/bin/python scripts/run_m3w_moving_joint_harm_runtime.py PHASE` '
        'for pilot/train/evaluate/verify. Completed receipts are immutable; do not rerun completed '
        'phases in place. Use `--resume` only for an interrupted unchanged phase. Native arm64 '
        'CPU4/interop1/workers0; no MPS claim.\n\n'
        'Native observation8/prediction12 annotation-order image pixels; not raw-frame t50, '
        'verified seconds, metric, physical-safety, true3D, foundation or human-gold evidence. '
        'Stage5C and SMC remain off.\n')
    text_once(PUBLIC / 'per_source.md', '# All Required Exposed Sources\n\n'
        '|Source|Product known pp|Product direct lower pp|Event known pp|Event direct lower pp|\n'
        '|---|---:|---:|---:|---:|\n'
        + '\n'.join('|'+site+'|'+'|'.join(per_source_value(rows, site, arm, metric)
            for arm in run.api.ARMS for metric in ('known_pp', 'completion_lower_pp'))+'|'
            for site in sorted({r['site'] for r in rows}))
        + '\n\nNA retains an unavailable required view; no source is removed or converted to zero.\n')
    text_once(PUBLIC / 'gates.md', '# Unchanged Scientific Decision\n\n'
        f"Combined registered primary and full-policy risk pass: **{decision['original_rules_pass']}**.\n\n"
        + result + '\n\nEngineering completion and replay are separate from scientific success. '
        'No deployment, independent confirmation or submission readiness is claimed. '
        'No identity rule, new holdout score, threshold tuning, Stage5C or SMC was added.\n')
    source.once(PUBLIC / 'runtime_summary.json', source.stable(dict(phases=phases,
        fresh_primary_fits=108, primary_optimizer_updates=216000, pilot_updates=1200,
        first_full_refit_updates=2000, primary_saved_fitter_seconds=sum(f['seconds'] for f in fits),
        train_execution_is_terminal_resumed_attempt_only=True, complete_matched_pairs=54,
        original_risk_rules_unchanged=True, source=source.artifact(PUBLIC / 'analysis.json'))))
    figures(fits, analysis)
    print(source.json.dumps(dict(report=str(PUBLIC / 'readout.md'), result=result, decision=decision), indent=2))


if __name__ == '__main__':
    main()
