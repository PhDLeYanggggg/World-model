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
    axes[0].plot(steps, mid, color='#35719a', label='Median across all108fits')
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
    axes[1].axvline(0, color='.5', lw=1)
    axes[1].set_xlabel('Veto minus own count/alpha-matched control (pp)')
    axes[1].set_title('Nested Cdevelopment, not independent test', fontsize=9)
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
        '**fresh_run:**108real Torch cost heads,216000primary optimizer updates; '
        'not a new trajectory model, JEPA or generative run. All full-budget weights froze '
        'before new nested Cvalidation scores. Both orientations and all three original '
        'producer seeds are retained. **cached_verified:** source admission, original '
        'forecast checkpoints, outer roles and existing inner-source mapping. '
        '**not_run:** new Hreadout, independent confirmation, deployment and paper submission.\n\n'
        'The user explicitly kept the original rules. There is no analytic identity '
        'certification. Original2%risk, easy-net, support requirements, all unavailable '
        'views and old failures remain. The original24source primary is still '
        'not_estimable, not passed.\n\n'
        '## Controlled Model Change\n\n'
        'Each single Cpart supplies two28229parameter cost heads. Means, scales, '
        'target RMS and bias use only that own fitting part. For a held Cpart, '
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
        'Required empty source views remain NA. The3000bootstrap uses23source '
        'units after producer/seed/orientation averaging;276repeated source views '
        'and696record views are not independent sample sizes. These are exposed '
        'nested development sources, not a recovered final test.\n\n'
        '## Unchanged Full-Policy Risk\n\n'+'\n'.join(risk)+'\n\n'
        'Whole-record risk uses every required view, not just nonempty utility '
        'queries. A no-intervention/zero-reference view does not automatically '
        'pass. A positive model contrast alone does not authorize deployment.\n\n'
        '## Runtime And Reproducibility\n\n'
        + '\n'.join(f"- {phase}: PID{r['pid']}, {r['seconds']:.2f}s, peakRSS{r['peak_RSS_bytes']/1e9:.3f}GB." for phase, r in receipts.items())
        + '\n\nAll6pilot resumed/direct scientific states and the first full-budget '
        'refit agree. Separate replay reproduces216unique Cprediction arrays, '
        '108orientation policy blocks,1620cost cells and every within-query '
        'count/amplitude histogram. Exact replay verifies engineering, not '
        'independent scientific confirmation. No CUDA/MPS claim; native arm64CPU4, '
        'interop1, workers0. Local native8/12annotation order, not verified time, '
        'metric, true3D, foundation or human-gold. Stage5C/SMC remain off.\n')
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
        original_risk_rules_unchanged=True, source=source.artifact(PUBLIC / 'analysis.json'))))
    figures(fits, summary)
    print(source.json.dumps(dict(report=str((PUBLIC / 'readout.md').relative_to(ROOT)), decision=gate), indent=2))


if __name__ == '__main__': main()
