"""Post-readout benefit/harm accounting; never select or change a policy."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_easy_hurdle as run
from scripts.report_m3w_easy_hurdle import put, value


def exchange(new, old):
    denominator = old['floor_error_sum']
    if denominator <= 0 or denominator != new['floor_error_sum']:
        raise ValueError('Common positive full-floor denominator required')
    for row in (new, old):
        np.testing.assert_allclose(row['floor_error_sum']-row['error_sum'],
                                   row['benefit_sum']-row['positive_harm_sum'],
                                   rtol=1e-7, atol=1e-7)
    loss = 100*(old['benefit_sum']-new['benefit_sum'])/denominator
    reduction = 100*(old['positive_harm_sum']-new['positive_harm_sum'])/denominator
    net = 100*(old['error_sum']-new['error_sum'])/denominator
    np.testing.assert_allclose(net, reduction-loss, rtol=1e-7, atol=1e-8)
    return dict(lost_benefit_pp=loss, harm_reduction_pp=reduction,
                net_gain_full_floor_pp=net)


def main():
    cfg, identity = run.identity()
    summary_path = run.PUBLIC/'summary.json'
    summary = json.loads(summary_path.read_text())
    assert summary['identity'] == identity
    runtime = json.loads((run.PUBLIC/'evaluation_runtime.json').read_text())
    assert runtime['summary'] == run.base.artifact(summary_path)
    details_path = run.ROOT/runtime['details']['path']
    assert run.base.artifact(details_path) == runtime['details']
    details = json.loads(details_path.read_text())
    index = {(r['group'], r['site'], r['policy']): r['metric'] for r in details['rows']}
    sites = sorted({r['site'] for r in details['rows']})
    assert len(sites) == 12
    comparisons = {}
    for new, old in run.CONTRASTS:
        rows = [dict(group=r['group'], site=r['site'],
                     metric=exchange(r['metric'], index[(r['group'], r['site'], old)]))
                for r in details['rows'] if r['policy'] == new]
        comparisons[new+'_vs_'+old] = {
            k: run.base.inter.paired_localities(rows, sites, k,
                cfg['bootstrap_resamples'], cfg['bootstrap_seed']) for k in rows[0]['metric']}
    monitors = {}
    for arm in run.api.ARMS:
        rows = [r for r in details['losses'] if r['arm'] == arm]
        assert len(rows) == 108 and all(r['step'] == 2000 for r in rows)
        monitors[arm] = {}
        for key in rows[0]['first']['monitor']:
            first = np.array([r['first']['monitor'][key] for r in rows])
            last = np.array([r['last']['monitor'][key] for r in rows])
            monitors[arm][key] = dict(initial_median=float(np.median(first)),
                final_median=float(np.median(last)), decreased_heads=int((last < first).sum()))
    out = dict(result_source='fresh_run_posthoc_accounting_of_frozen_development_readout',
        summary=run.base.artifact(summary_path), details=runtime['details'],
        comparisons=comparisons, fitting_monitor_components=monitors,
        independent_localities=12, repeated_role_views_not_independent=True,
        primary_changed=False, thresholds_changed=False, new_training=False,
        fixed_denominator_is_diagnostic=True)
    run.base.immutable_json(run.PUBLIC/'posthoc_accounting.json', out)
    lines = ['# Why Better Probability Fit Did Not Improve Selection', '',
        'Posthoc arithmetic on the frozen development readout. This does not modify any model, threshold, action, primary comparison or gate. Sources: summary.json and its hash-bound private evaluation details.', '',
        '## Benefit and Harm', '',
        'For each view, error = floor error - selected benefit + selected positive harm. Therefore net gain = harm reduction - lost benefit. Each term below uses the same full-floor denominator. This is not a substitute for the registered selected-risk denominator or the paired ADE comparison denominator.', '',
        '| New versus control | Lost benefit (pp) | Harm reduction (pp) | Net gain/full floor (pp) |',
        '|---|---:|---:|---:|']
    for name, row in comparisons.items():
        lines.append('| '+name+' | '+' | '.join(value(row[k]) for k in
            ('lost_benefit_pp', 'harm_reduction_pp', 'net_gain_full_floor_pp'))+' |')
    lines += ['', 'Intervals are 3,000 nominal paired-locality bootstrap draws across 12 opened development localities. They are not independent confirmation or simultaneous bounds. The three training seeds and repeated role views are not independent samples.', '',
        '## Actual Fitting Losses', '',
        'Each entry is the median across 108 paired fits of the same frozen fitting-monitor component at update 0 or 2,000. Different objectives must not be ranked by their total loss; occurrence and conditional components are unsupervised in the marginal arm. These are fitting diagnostics, not held accuracy or proof of convergence.', '',
        '| Arm | Component | Initial median | Final median | Heads decreased / 108 |',
        '|---|---|---:|---:|---:|']
    for arm, components in monitors.items():
        for key, row in components.items():
            lines.append(f"| {arm} | {key} | {row['initial_median']:.8g} | {row['final_median']:.8g} | {row['decreased_heads']} |")
    lines += ['', '## Supported Findings and Open Hypotheses', '',
        '- The occurrence and conditional-cost fits improve, but the composed signed-risk error does not improve in this readout. Proper occurrence scoring alone is not the downstream objective.',
        '- Matching every query count excludes intervention volume as the sole explanation for the primary negative result. The decomposition quantifies benefit lost versus harm avoided; it does not establish why the shared representation changed.',
        '- A positive mean signed-risk bias can suppress useful admissions. The all-risk score and utility remain frozen, so their residual errors can still produce realized selected-risk violations even when predicted constraints are feasible.',
        '- The sigmoid occurrence probability does not change an individual conditional-risk sign. Its possible value is in relative query weighting, not a separate per-agent safety veto.',
        '- Loss-scale competition, representation sharing and conditional-moment product bias remain hypotheses. This experiment jointly added two auxiliaries, so it cannot identify their separate causal effects.',
        '- Next work should measure component gradient scales and conflict on fitting sources, then preregister a single controlled remedy. Do not tune auxiliary weights, thresholds or eligibility on these held-development outcomes.', '',
        'No deployment change, new independent data, metric/seconds claim or risk certificate. Stage5C and SMC remain disabled.']
    put('failure_analysis.md', '\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
