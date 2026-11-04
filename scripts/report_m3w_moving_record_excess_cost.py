"""Complete objective-only neural study; fitting success is not deployment."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_moving_record_excess_cost as run
from scripts.report_m3w_moving_harm_refinement import text_once
from scripts.report_m3w_moving_amplitude_cost import interval, secondary_intervals

source, PUBLIC = run.source, run.PUBLIC


def positive(row):
    return row['CI95'] is not None and row['CI95'][0] > 0


def scientific_decision(summary):
    risk = summary['safety']['record_excess']['joint_free']
    supported = (risk['required_record_views'] == 348 and risk['supported_record_views'] == 348
        and risk['full_original_risk_pass'] and risk['undefined_easy_views'] == 0
        and risk['known_all_risk_violations'] == 0 and risk['known_easy_risk_violations'] == 0
        and risk['worst_known_easy_net_degradation_percent'] is not None
        and risk['worst_known_easy_net_degradation_percent'] <= 2)
    primary = positive(summary['paired']['shared_tree']['joint_free']['completion_lower_pp'])
    mechanism = positive(summary['paired']['point']['joint_free']['completion_lower_pp'])
    own = positive(summary['completion_lower']['record_excess']['joint_free'])
    return dict(primary_vs_shared_tree_positive=primary, matched_neural_mechanism_positive=mechanism,
        own_floor_completion_lower_positive=own, full_original_risk_pass=bool(supported),
        supported_source_mechanism_gate=bool(primary and mechanism and own and supported),
        original24group_primary='not_estimable_not_passed', independent_confirmation=False,
        new_neural_cost_training=True, new_trajectory_training=False, new_deployment=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False)


def figures(summary, fits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    public, private = PUBLIC / 'figures', run.PRIVATE / 'figure_previews'
    public.mkdir(exist_ok=True)
    private.mkdir(exist_ok=True)
    plt.rcParams.update({'svg.fonttype': 'none', 'font.size': 9})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, metric, title in zip(axes, ('known_pp', 'completion_lower_pp'), ('Known labels', 'Direct completion lower')):
        for i, control in enumerate(('point', 'shared_tree')):
            r = summary['paired'][control]['joint_free'][metric]
            if r['CI95'] is not None:
                ax.errorbar(r['point'], i, xerr=[[r['point']-r['CI95'][0]], [r['CI95'][1]-r['point']]], fmt='o', capsize=4)
            else:
                ax.text(0, i, 'not estimable')
        ax.set_yticks([0, 1], ['Vs matched neural', 'Vs shared tree'])
        ax.axvline(0, color='.5', lw=1)
        ax.set_ylim(1.5, -.5)
        ax.set_title(title)
        ax.set_xlabel('Paired utility gain (pp)')
    fig.suptitle('23 exposed source groups, three actual seeds, grouped 95% CI; not confirmation', fontsize=10)
    fig.tight_layout()
    fig.savefig(public / 'paired_effects.svg')
    fig.savefig(private / 'paired_effects.png', dpi=160)
    plt.close(fig)
    new = [run.api.base.base.read_checkpoint(source.checked(r['checkpoint']))['trace'] for r in fits]
    old = [run.api.base.base.read_checkpoint(source.checked(r['cached_matched_control']))['trace'] for r in fits]
    steps = [r['step'] for r in new[0]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for traces, key, label in ((old, 'total', 'Point control'), (new, 'base_total', 'Aggregate-trained base loss')):
        values = np.array([[r[key] for r in trace] for trace in traces])
        axes[0].plot(steps, np.median(values, 0), label=label)
        axes[0].fill_between(steps, values.min(0), values.max(0), alpha=.1)
    values = np.array([[r['record_excess'] for r in trace] for trace in new])
    axes[1].plot(steps, np.median(values, 0), color='#ad593e')
    axes[1].fill_between(steps, values.min(0), values.max(0), color='#ad593e', alpha=.1)
    axes[0].set_title('Same original fitting objective')
    axes[1].set_title('New loss-only record-excess term')
    axes[0].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xlabel('Optimizer updates')
        ax.set_ylabel('Own fitting loss, median/min-max')
        ax.set_yscale('symlog', linthresh=1e-8)
    fig.suptitle('18 matched contexts per arm; training curves are not held-policy performance', fontsize=10)
    fig.tight_layout()
    fig.savefig(public / 'training_losses.svg')
    fig.savefig(private / 'training_losses.png', dpi=160)
    plt.close(fig)


def main():
    s = source.read(PUBLIC / 'summary.json')
    v = source.read(PUBLIC / 'verification.json')
    freeze = source.read(PUBLIC / 'training_freeze.json')
    refits = source.read(PUBLIC / 'first_full_refits.json')
    if (not v['all36neural_cost_outputs_and_all_policy_scalar_blocks_exact']
            or v['summary_sha256'] != source.sha(PUBLIC / 'summary.json')
            or freeze['complete_new_neural_cost_heads'] != 18
            or not refits['point_matches_original'] or not refits['record_excess_refit_exact']):
        raise ValueError('Complete all fits, full refits and held replay before conclusions')
    fits = [source.read(source.checked(r)) for r in freeze['fits']]
    for fit in fits:
        source.checked(fit['checkpoint'])
        source.checked(fit['cached_matched_control'])
    records = [r for ref in s['group_receipts'] for r in source.read(source.checked(ref))['rows']]
    if len(records) != 138 or len({r['site'] for r in records}) != 23:
        raise ValueError('All supported source groups required')
    d = scientific_decision(s)
    source.once(PUBLIC / 'decision.json', source.stable(d))
    source.once(PUBLIC / 'secondary_metrics.json', source.stable(secondary_intervals(records, source.read(run.PARENT.CONFIG))))
    runtime = dict(phases={p: source.read(PUBLIC / (p + '_execution.json')) for p in ('pilot', 'train', 'evaluate', 'verify')},
        primary_new_neural_cost_heads=18, primary_optimizer_updates=36000, cached_matched_neural_controls=18,
        completed_pilot_optimizer_updates=1500, completed_full_refit_optimizer_updates=4000,
        known_failed_pilot_unsaved_optimizer_updates=1, engineering_tiny_fit_updates_not_experiment_training=True,
        new_trajectory_training=False, fresh_shared_tree_cost_inference=False)
    source.once(PUBLIC / 'runtime_summary.json', source.stable(runtime))
    table = ['|New joint-free versus control|Known utility pp [95%CI]|Direct completion-lower pp [95%CI]|', '|---|---:|---:|']
    for arm in ('point', 'shared_tree'):
        r = s['paired'][arm]['joint_free']
        table.append(f'|{arm}|{interval(r["known_pp"])}|{interval(r["completion_lower_pp"])}|')
    risk_lines = ['|Joint-free arm|Full support /348|Known all violations|Known easy violations|Undefined easy views|Worst signed easy-net %|', '|---|---:|---:|---:|---:|---:|']
    for arm in ('point', 'shared_tree', 'record_excess'):
        r = s['safety'][arm]['joint_free']
        easy = 'not_estimable' if r['worst_known_easy_net_degradation_percent'] is None else f'{r["worst_known_easy_net_degradation_percent"]:.6f}'
        risk_lines.append(f'|{arm}|{r["supported_record_views"]}|{r["known_all_risk_violations"]}|{r["known_easy_risk_violations"]}|{r["undefined_easy_views"]}|{easy}|')
    text_once(PUBLIC / 'readout.md', '# Selected-Record Neural Cost Training\n\n'
        '18fresh28229parameter neural cost heads,36000primary optimizer updates,three actual seeds. '
        '18matched point controls are cached models with freshly replayed inference; shared-tree costs '
        'are not freshly inferred. Shared decisions are cached_verified with complete fresh scalar replay. '
        'All36neural cost arrays and all policy blocks replay exactly. No new trajectory producer.\n\n'
        + '\n'.join(table) + '\n\n' + '\n'.join(risk_lines) + '\n\n'
        + f'Supported-source mechanism gate:{d["supported_source_mechanism_gate"]}; '
        f'full original risk:{d["full_original_risk_pass"]}; new deployment:False.\n\n'
        'The only changed training factor is a weight1selected-record aggregate excess objective. '
        'The actual18initializations, source rows, normalizers, sampling draws and budgets match. '
        'All six real pilot recoveries, three original-loop comparisons and two complete first-job '
        'refits pass. One initial logging exception and its discarded unsaved step remain recorded.\n\n'
        'The original2%positive-harm/support and signed-net conditions all remain necessary. '
        'Signed-net preservation or fewer violations alone is not a pass. Empty same-count controls '
        'remain not_estimable; all policies, sources and recording views remain in summary/held. '
        'Unknown-label bounds are not observed future events. Original24primary stays not_estimable.\n\n'
        'Cteacher masks are supervised fitting masks, not OOF validation or independent calibration. '
        'Record aggregation is loss-only over sampled queries, not inference across future frames '
        'or an unbiased full-video loss estimator. Changing the loss may alter all five moments; '
        'this is not a reference-only or architecture ablation, SPO+implementation or safety theorem.\n\n'
        'All23supported groups are exposed development. No independent confirmation, reserved-role '
        'opening or Hcheckpoint/threshold selection. No metric/seconds/physical-safety/true3D/foundation/'
        'human-gold claim. Native8/12annotation order, not verified sensor-as-of time. Stage5C/SMC off.\n')
    lines = ['# Every Supported Source', '', '|Source|New-minus-point known pp|New-minus-point lower pp|New-minus-shared known pp|New-minus-shared lower pp|', '|---|---:|---:|---:|---:|']
    for site in sorted({r['site'] for r in records}):
        values = []
        for control in ('point', 'shared_tree'):
            for metric in ('known_pp', 'completion_lower_pp'):
                row = s['paired'][control]['joint_free'][metric]
                values.append('not_estimable' if 'by_source_group' not in row else f'{row["by_source_group"][site]:+.6f}')
        lines.append('|' + site + '|' + '|'.join(values) + '|')
    text_once(PUBLIC / 'per_source.md', '\n'.join(lines) + '\n')
    text_once(PUBLIC / 'gates.md', '# Fixed Scientific Gates\n\n'
        + '\n'.join(f'- {k}: {value}' for k, value in d.items()) + '\n\n'
        'No total combines execution checks with independent efficacy. Original24source primary '
        'and reserved confirmation remain incomplete even if a supported-source mechanism improves.\n')
    text_once(PUBLIC / 'model_data_card.md', '''# Model And Data Card

An existing causal cost network, not a new trajectory or latent-generative model.
Width64,28229parameters, original local380descriptor fields plus48zero slots,
same candidate codes and learned within-current-query context. Native CPU4,
interop1,workers0, three seeds17/29/43,2000updates per model. All preprocessing
is original Cfit-only; no future endpoint/mask/easy label/test goal enters the
network. Recording IDs and teacher actions are training-loss metadata only.

195286past-eligible SOURCE windows from64admitted recordings, with23supported
groups and original24group primary not_estimable. Overlapping windows and
repeated views are not independent samples. Supplied partial future labels
remain partial; full-roster FDE is unavailable where endpoints are missing.
All labels are released/derived, not human-gold promotion. No new scene images,
latent cache or simulation is admitted. Not SDDt50, metric or seconds-level.

Teacher selections are same-C supervised fitting, not inner-source OOF or
independent calibration. Sampled-record squared aggregate error is not an
unbiased full-video squared-loss estimator. No statistical safety theorem is
claimed. All original risk/support and signed-net conditions are unchanged.
Reserved20Moving groups, DUT and DroneCrowd confirmation remain closed.
Checkpoint/RNG/optimizer recovery, actual loop matching and full refits are
engineering evidence; held utility and full risk are separate scientific tests.

No new deployment or submission readiness. No true3D, foundation, physical-
safety, Stage5C execution or SMC. Private checkpoints and row data are excluded.
''')
    text_once(PUBLIC / 'failure_analysis.md', f'''# Aggregation Is A Test, Not A Safety Guarantee

Primary robust comparison positive:{d['primary_vs_shared_tree_positive']}.
Matched neural contrast positive:{d['matched_neural_mechanism_positive']}.
Own floor lower positive:{d['own_floor_completion_lower_positive']}.
Every original recording risk/support condition passed:{d['full_original_risk_pass']}.

Read all per-source and four-policy results, including negative/NA cases.
If utility improves without complete risk support, it is not deployable. If
fitting loss improves without held utility, this objective has not repaired
the decision problem. Fewer interventions can trade utility for preservation;
signed easy-net alone cannot hide positive harm or absent selected reference.

The controlled variable is the extra loss, not architecture, representation or
trajectory producer. Actual control-loop and initializer/draw equality rule
out an accidental training-loop substitution; they do not identify which cost
moment mediates a utility change. Cteacher-to-new-policy selection shift,
minibatch record aggregation, finite reference events, unknown future labels,
small source count and exposed development remain limitations. Optimizer's
curse is a plausible mechanism, not established causation or a new discovery.

Do not tune this weight or thresholds on H. A further repair needs its own
fitting-only choice rule and unchanged risk contract. Independent confirmation
and composition evidence remain required for paper claims. No metric/seconds,
physical-safety, true3D, foundation, Stage5C or SMC. Main v9 PDF is not recompiled.
''')
    figures(s, fits)
    print(source.json.dumps(dict(decision=d, primary=s['paired']['shared_tree']['joint_free'],
                                risk=s['safety']['record_excess']['joint_free']), sort_keys=True))


if __name__ == '__main__':
    main()
