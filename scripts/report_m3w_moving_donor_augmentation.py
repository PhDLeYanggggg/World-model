"""Keep risk information gains distinct from a full-policy safety result."""
import csv
import io
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_donor_augmentation as run


def conclusion(analysis):
    if any(analysis[k] for k in ('new_outer_C_H_readout', 'identity_certification', 'deployment_changed',
                                  'submission_ready', 'stage5c_executed', 'smc_enabled')):
        raise ValueError('Do not promote an own-B information screen into deployment')
    if len(analysis['folds']) != 24 or len(analysis['contexts']) != 3:
        raise ValueError('Complete24fit development readout required')
    return ('donor_augmentation_information_supported_not_deployable'
            if analysis['passing_contexts'] >= 2 else 'donor_augmentation_information_not_supported')


def main():
    p = run.PUBLIC
    a, reg, donors, freeze = [run.read(p/name) for name in
                             ('analysis.json', 'registration.json', 'donors.json', 'training_freeze.json')]
    status = conclusion(a)
    if not run.read(p/'verification.json')['all24_full_refits_exact'] or freeze['models'] != 24:
        raise ValueError('Complete refitting required')
    fits = [run.read(run.checked(ref)) for ref in freeze['fits']]
    for f in fits:
        run.checked(f['checkpoint'])
    lines = ['# Does Source-Held Donor Augmentation Improve Risk Learning?', '',
        'I compared the same risk architecture, original120 causal inputs and fixed',
        '4,096-update budget with own-B donors, producer-A crossfit donors, and a50/50',
        'mixture. The eight query-source folds and original decision threshold stay fixed.', '',
        f'**Information screen: {a["passing_contexts"]}/3 contexts.**',
        f'Verdict: `{status}`. No deployment change or full-policy safety claim.', '',
        '## Matched Final Readout', '',
        '| Context | Arm | Risk MSE | Decision cost | False-safe cost | Cost reduction vs own-only % |',
        '|---|---|---:|---:|---:|---:|']
    for c in a['contexts']:
        base = c['source_macro']['own_only']['total_joint_cost']
        for arm, m in c['source_macro'].items():
            gain = 100*(1-m['total_joint_cost']/base) if base else None
            lines.append(f'| {c["job"].split("_")[-1]} | {arm} | {m["normalized_MSE"]:.6f} | '
                f'{m["total_joint_cost"]:.6f} | {m["false_safe_cost"]:.3e} | '
                f'{gain:+.6f} |' if gain is not None else f'| {arm} | NA |')
    lines += ['', 'Positive reductions are better. Costs are source-macro known-proposal',
        'signed decision costs, not trajectory ADE/FDE or full-policy harm ratios.',
        'Every arm uses the same original own-B evaluation normalization per source.',
        'Cached forest/prior neural/continued neural controls are separate from the',
        'fresh matched own-only control; training samplers and budgets differ from',
        'the older models. No threshold is fitted to these outcomes.', '',
        '**The stricter forest comparison does not pass.** Augmented false-safe',
        'cost exceeds shared64 in all three contexts, even where total decision',
        'cost is lower. A better point-risk predictor is not a safe deployment rule.', '',
        '## Training And Source Coverage', '',
        'All24models complete4,096updates:98,304primaryupdates. All24are fully refit',
        'before final readout, another98,304updates. The recovery pilot contributes',
        '512additionalupdates and is not reused in the primary fits. One seed only.',
        'The network has9,890parameters. Sampling is source/query-balanced with',
        'replacement, not32epochs; actual distinct-row visits are retained.', '',
        '| Parent | Source components | Known legal candidates | Unknown legal candidates | Identical-to-floor candidates |',
        '|---|---:|---:|---:|---:|']
    for b in donors['banks']:
        lines.append(f'| A{b["producer_fold"]} | {len(b["sources"])} | {b["rows"]} | '
                     f'{b["unknown_legal_candidates"]} | {b["identical_to_floor_candidates"]} |')
    lines += ['', 'The full donor treatment includes six fixed candidate amplitudes, unlike',
        'the already-selected own-B candidates. Crossfit teachers also have smaller',
        'source training sets. This is a practical donor-bank treatment, not an',
        'isolated causal estimate of source count. No donor source is excluded',
        'after observing its forecasting error. Unknown futures are not called safe.', '',
        '## Evidence Boundaries', '',
        '- fresh_run: new A candidate features/labels,24riskfits,24full refits and final own-B readout.',
        '- cached_verified: six OOF forecast banks, own-B features/labels and old controls.',
        '- not_run: new outer C/H evaluation, independent confirmation or a deployable policy.',
        '- Original2% all/easy harm, signed easy-net and reference/support rules remain.',
        '- Empty references and empty utility controls remain NA. No identity certificate.',
        '- Original24-group main estimand remains NA; old failures remain failures.',
        '- Native8/12image-coordinate silver labels, not verified metric/seconds/true3D/foundation.',
        '- Annotation-generation temporal causality remains unverified. Stage5C/SMC off.', '',
        '[Per-source results](source_results.csv) | [Loss trace](loss_trace.csv)',
        '| [Failure analysis](failure_analysis.md) | [Reproduction](operation_zh.md)']
    (p/'README.md').write_text('\n'.join(lines)+'\n')
    trace = io.StringIO()
    w = csv.writer(trace, lineterminator='\n')
    w.writerow(['job', 'source', 'arm', 'step', 'batch_loss', 'gradient_norm', 'distinct_rows'])
    for f in fits:
        for row in f['trace']:
            w.writerow([f['job'], f['source'], f['arm'], row['step'], row['batch_loss'],
                        row['gradient_norm'], row['distinct_fitting_rows_drawn']])
    (p/'loss_trace.csv').write_text(trace.getvalue())
    out = io.StringIO()
    fields = ['job', 'source', 'arm', 'rows', 'normalized_MSE', 'total_joint_cost',
              'false_safe_cost', 'false_reject_cost', 'accepted_known_proposals', 'false_safe_count']
    w = csv.DictWriter(out, fieldnames=fields, lineterminator='\n')
    w.writeheader()
    for r in a['folds']:
        w.writerow({k: r[k] if k in r else r['metrics'][k] for k in fields})
    (p/'source_results.csv').write_text(out.getvalue())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for arm, offset, color in [('own_only', -.3, '#2563eb'), ('A_only', -.1, '#ca8a04'),
                               ('augmented', .1, '#15803d'), ('shared64', .3, '#6b7280')]:
        for ax, key in zip(axes, ['total_joint_cost', 'false_safe_cost']):
            ax.bar(run.np.arange(3)+offset, [c['source_macro'][arm][key] for c in a['contexts']],
                   width=.19, label=arm, color=color)
    for ax, title in zip(axes, ['Known-proposal decision cost', 'False-safe decision cost']):
        ax.set(xticks=range(3), xticklabels=['critic0', 'critic1', 'critic2'], title=title,
               xlabel='Exposed development context', ylabel='Source-macro cost')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.2)
    folder = run.PRIVATE/'figures'
    folder.mkdir(exist_ok=True)
    fig.savefig(folder/'donor_treatment.png', dpi=170)
    fig.savefig(folder/'donor_treatment.pdf')
    plt.close(fig)
    run.once(p/'result_summary.json', dict(status=status, passing_contexts=a['passing_contexts'],
        primary_updates=98304, full_repeat_updates=98304, pilot_updates=512,
        original_rules_unchanged=True, deployment_changed=False, submission_ready=False))
    print(status)


if __name__ == '__main__':
    main()
