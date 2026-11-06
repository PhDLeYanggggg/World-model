"""Report risk-seed stability separately from forecaster/generalization claims."""
import csv
import io
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_donor_seeds as run


def verdict(a):
    if (len(a['folds']) != 72 or len(a['contexts']) != 9
            or a['risk_head_seeds'] != [17, 29, 43] or a['forecaster_seeds'] != [17]
            or any(a[k] for k in ('new_outer_C_H_readout', 'identity_certification',
                'independent_confirmation', 'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Complete fixed-teacher component evidence, not main-model confirmation')
    return ('three_risk_seed_donor_information_replicated_not_deployable'
            if a['replication_screen_pass'] else 'risk_seed_replication_failed_not_deployable')


def main():
    p = run.PUBLIC
    a = run.read(p/'analysis.json')
    status = verdict(a)
    if not run.read(p/'verification.json')['all48_full_refits_exact']:
        raise ValueError('Full refits required')
    frozen = run.read(run.old.PUBLIC/'analysis.json')
    lines = ['# Does The Donor Effect Survive Risk-Head Seeds?', '',
        'I kept the teacher cache, data, model, update budget, source exclusions',
        'and decision rule fixed, and added risk-head seeds29/43 to the verified',
        'seed17 result. This is not a three-forecaster-seed main experiment.', '',
        f'Verdict: **{status}**.',
        f'Passing information contexts by seed: `{a["passing_contexts_by_seed"]}`.',
        'No deployment upgrade. The original full-policy safety rules still apply.', '',
        '## Seed Results', '',
        '| Risk seed | Context | Cost reduction vs own % | MSE reduction vs own % | Cost reduction vs forest % | False-safe cost nonworse than forest |',
        '|---|---|---:|---:|---:|---|']
    for c in a['contexts']:
        lines.append(f'| {c["seed"]} | {c["job"].split("_")[-1]} | '
            f'{100*c["cost_reduction_vs_own"]:+.6f} | {100*c["MSE_reduction_vs_own"]:+.6f} | '
            f'{100*c["cost_reduction_vs_forest"]:+.6f} | {c["false_safe_cost_nonworse_forest"]} |')
    lines += ['', 'Positive reductions are better. These are source-macro known-proposal',
        'risk/decision costs, not trajectory ADE/FDE or a full-policy safety ratio.',
        'The cohort remains48,061known proposals and1,274unknown proposals. Unknown',
        'outcomes are outside this known-only information estimate, not certified safe.',
        'Evaluation normalization and zero threshold are identical to the frozen',
        'comparison. No seed or checkpoint is chosen by these results.', '',
        '## Seed Variation', '',
        '| Context | Cost reduction vs own, mean +/- sample SD % | Min / max % |',
        '|---|---:|---:|']
    stats = []
    for old in frozen['contexts']:
        rows = [c for c in a['contexts'] if c['job']==old['job']]
        gain = run.np.asarray([100*c['cost_reduction_vs_own'] for c in rows])
        record = dict(job=old['job'], cost_gain_mean=float(gain.mean()), cost_gain_SD=float(gain.std(ddof=1)),
                      cost_gain_min=float(gain.min()), cost_gain_max=float(gain.max()))
        stats.append(record)
        lines.append(f'| {old["job"].split("_")[-1]} | {gain.mean():+.6f} +/- {gain.std(ddof=1):.6f} | '
                     f'{gain.min():+.6f} / {gain.max():+.6f} |')
    aug_mean = run.np.mean([c['source_macro']['augmented']['total_joint_cost'] for c in a['contexts']])
    own_mean = run.np.mean([c['source_macro']['own_only']['total_joint_cost'] for c in a['contexts']])
    forest_mean = run.np.mean([c['source_macro']['shared64']['total_joint_cost'] for c in frozen['contexts']])
    points = dict(own=100*(1-aug_mean/own_mean), forest=100*(1-aug_mean/forest_mean))
    lines += ['', '## Source-Cluster Uncertainty', '',
        '3000paired source-component replicates;2/3/3components per context.',
        'The same sampled components are used for all arms and risk seeds.',
        'Risk seeds are averaged, not resampled; windows are never sampling units.', '',
        '| Reference | Seed/context-macro point reduction % | Exploratory95% interval % |',
        '|---|---:|---:|']
    for ref in ('own', 'forest'):
        ci = a['source_bootstrap_vs_'+ref]['interval95']
        lines.append(f'| {ref} | {points[ref]:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] |')
    lines += ['', 'Here seed averaging means averaging individual-head costs, not risk scores.',
        'These intervals do not apply to the mean/unanimity ensemble diagnostics.',
        'Intervals are conditional on these eight exposed source components and',
        'fixed seed17 teachers. Small-cluster bootstrap and three head seeds do not',
        'establish new-domain coverage, independent confirmation or complete pipeline',
        'seed stability. Negative forest comparisons and source-level tradeoffs remain.', '',
        '## What Actually Ran', '',
        '- fresh_run:48new risk fits,48full exact refits and new-seed development readout.',
        '- cached_verified:24seed17heads/results, frozen donor/own-B arrays and teacher lineage.',
        '- Primary new updates196,608; repeats another196,608; separate pilot4,608.',
        '- The pilot matches the complete frozen seed17engine and seed29resume exactly.',
        '- not_run:new outer C/H, independent confirmation, new teacher seeds or deployment.',
        '- All three risk seeds are retained; no best-seed selection or threshold tuning.',
        '- Original2% harm, easy-net and support rules stay fixed; empty controls stay NA.',
        '- Original24-group main NA and old failures remain. No identity certificate.',
        '- Native8/12image-coordinate silver annotations; no verified meters or seconds.',
        '- No true3D/foundation, Stage5C execution or SMC. Not submission-ready.', '',
        '[Per-source seed results](source_seed_results.csv) | [Loss trace](loss_trace.csv)',
        '| [Failure analysis](failure_analysis.md) | [Reproduction](operation_zh.md)']
    (p/'README.md').write_text('\n'.join(lines)+'\n')
    out = io.StringIO()
    fields = ['seed', 'job', 'source', 'arm', 'rows', 'normalized_MSE', 'total_joint_cost',
              'false_safe_cost', 'false_reject_cost', 'accepted_known_proposals', 'false_safe_count', 'result_source']
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for r in a['folds']:
        writer.writerow({k:r[k] if k in r else r['metrics'][k] for k in fields})
    (p/'source_seed_results.csv').write_text(out.getvalue())
    consensus = run.read(p/'consensus/analysis.json')
    detail = ['# Fixed Mean And Unanimity Diagnostic', '',
        'Specified before seed29/43readout. All three augmented risk heads are used;',
        'no threshold, donor, seed or model is selected from these outcomes.', '',
        '| Context | Arm | Decision cost | Change vs forest % | False-safe cost | Accepted | Harmful |',
        '|---|---|---:|---:|---:|---:|---:|']
    for c in consensus['contexts']:
        for arm in ('mean','unanimity'):
            m, count = c['means'][arm], c['counts'][arm]
            effect = 100*(1-m['total_joint_cost']/c['forest']['total_joint_cost'])
            detail.append(f'| {c["job"].split("_")[-1]} | {arm} | {m["total_joint_cost"]:.7f} | '
                f'{effect:+.6f} | {m["false_safe_cost"]:.3e} | {count["accepted_known_proposals"]} | {count["false_safe_count"]} |')
    detail += ['', 'The maximum score implements a strict intersection of admissions. It',
        'reduces harmful counts versus the single seed17head46/229/602 to7/26/26,',
        'but also reduces accepted cases. The forest harmful counts are0/1/0.',
        'Both fixed ensemble rules lower total known-proposal cost in all three',
        'contexts relative to the forest; false-safe cost is still higher.', '',
        'These point comparisons do not establish a full-policy2% risk pass,',
        'coverage of uncertainty estimates, physical safety or independent transfer.',
        'The denominator, unknown-label contribution and reference/support conditions',
        'of the original full-population policy still need evaluation. An accepted-set',
        'intersection alone does not prove that a risk ratio falls. Empty support is NA.',
        'No deployment, identity certificate, new outer C/H, Stage5C or SMC.']
    (p/'consensus/README.md').write_text('\n'.join(detail)+'\n')
    trace = io.StringIO()
    writer = csv.writer(trace, lineterminator='\n')
    writer.writerow(['seed', 'job', 'source', 'arm', 'step', 'batch_loss', 'gradient_norm', 'distinct_rows'])
    for ref in run.read(p/'training_freeze.json')['fits']:
        f = run.read(run.checked(ref))
        run.checked(f['checkpoint'])
        for r in f['trace']:
            writer.writerow([f['seed'], f['job'], f['source'], f['arm'], r['step'], r['batch_loss'],
                             r['gradient_norm'], r['distinct_fitting_rows_drawn']])
    (p/'loss_trace.csv').write_text(trace.getvalue())
    run.once(p/'result_summary.json', run.stable(dict(status=status, statistics=stats, points=points,
        source_bootstrap_vs_own=a['source_bootstrap_vs_own'], source_bootstrap_vs_forest=a['source_bootstrap_vs_forest'],
        replication_screen_pass=a['replication_screen_pass'], passing_contexts_by_seed=a['passing_contexts_by_seed'],
        teacher_seeds=[17], risk_head_seeds=[17,29,43], deployment_changed=False, submission_ready=False)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    jobs = [c['job'] for c in frozen['contexts']]
    for seed, shift, color in ((17, -.23, '#2563eb'), (29, 0, '#15803d'), (43, .23, '#ca8a04')):
        rows = [next(c for c in a['contexts'] if c['seed']==seed and c['job']==j) for j in jobs]
        for ax, key in zip(axes, ['cost_reduction_vs_own', 'cost_reduction_vs_forest']):
            ax.bar(run.np.arange(3)+shift, [100*c[key] for c in rows], width=.23, label=f'Risk seed{seed}', color=color)
    for ax, title in zip(axes, ['Versus matched own-only', 'Versus conservative forest']):
        ax.axhline(0, linewidth=1, color='#dc2626')
        ax.set(xticks=range(3), xticklabels=['critic0', 'critic1', 'critic2'],
               xlabel='Exposed development context', ylabel='Decision-cost reduction (%)', title=title)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Fixed teacher17; risk-head seeds only', fontsize=11)
    directory = run.PRIVATE/'figures'
    directory.mkdir(exist_ok=True)
    fig.savefig(directory/'risk_seed_effects.png', dpi=170)
    fig.savefig(directory/'risk_seed_effects.pdf')
    plt.close(fig)
    print(status)


if __name__ == '__main__':
    main()
