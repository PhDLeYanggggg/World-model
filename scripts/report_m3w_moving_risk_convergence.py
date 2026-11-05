"""Summarize fixed-budget continuation without selecting a favorable checkpoint."""
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_risk_convergence as run


def reduction(value, reference):
    return 100*(1-value/reference) if reference>0 else None


def summarize(analysis):
    contexts, folds = [], []
    for c in analysis['contexts']:
        means = c['summary']['source_macro']
        arms = {}
        for name in run.ARMS:
            r = means[name]
            arms[name] = dict(r,
                cost_reduction_percent=reduction(r['total_joint_cost'], means['shared64']['total_joint_cost']),
                MSE_reduction_percent=reduction(r['normalized_MSE'], means['shared64']['normalized_MSE']),
                admitted=sum(f['metrics'][name]['accepted_known_proposals'] for f in c['folds']),
                harmful=sum(f['metrics'][name]['false_safe_count'] for f in c['folds']))
        contexts.append(dict(job=c['job'], arms=arms, passes=c['summary']['passes_information_screen'],
            cost_reduction_vs_short_percent=reduction(means['long_transport']['total_joint_cost'],
                                                     means['short_transport']['total_joint_cost'])))
        for f in c['folds']:
            d = f['fitting_diagnosis']
            folds.append(dict(job=c['job'], source=f['source'], fit_rows=f['fit_rows'],
                extra_updates=f['extra_updates'], epochs=f['epochs'], **d,
                fit_loss_reduction_percent=reduction(d['end_fit_loss'], d['start_fit_loss']),
                short_held_MSE=f['metrics']['short_plain']['normalized_MSE'],
                long_held_MSE=f['metrics']['long_plain']['normalized_MSE']))
    return dict(contexts=contexts, folds=folds, passing_contexts=analysis['passing_contexts'],
        additional_updates=sum(f['extra_updates'] for f in folds),
        fitting_loss_improved_folds=sum(f['end_fit_loss']<f['start_fit_loss'] for f in folds),
        held_plain_MSE_improved_folds=sum(f['long_held_MSE']<f['short_held_MSE'] for f in folds),
        operational_plateau_folds=sum(f['operational_plateau'] for f in folds),
        mathematical_convergence_proved=False, original_rules_unchanged=True,
        independent_confirmation=False, deployment_changed=False, submission_ready=False)


def figures(result, trace):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for f in result['folds']:
        rows = [r for r in trace if r['source']==f['source'] and r['job']==f['job']]
        axes[0].plot([r['extra_steps'] for r in rows], [r['weighted_fit_MSE'] for r in rows],
                     label=f['source'].split('_')[-1], linewidth=1.4)
    axes[0].set(xlabel='Additional optimizer updates', ylabel='Fitting normalized MSE',
                yscale='log', title='Fitting improves with more optimization')
    axes[0].legend(title='Excluded source', fontsize=8, ncol=2)
    x = list(range(len(result['folds'])))
    for name, key, color, marker in (
            ('Fit: 32 epochs', 'start_fit_loss', '#ca8a04', 'o'),
            ('Fit: continued', 'end_fit_loss', '#15803d', 's'),
            ('Source-held: 32 epochs', 'short_held_MSE', '#2563eb', 'o'),
            ('Source-held: continued', 'long_held_MSE', '#dc2626', 's')):
        axes[1].plot(x, [f[key] for f in result['folds']], label=name, color=color, marker=marker)
    axes[1].set(xticks=x, xticklabels=[f['source'].split('_')[-1] for f in result['folds']],
                xlabel='Excluded source', ylabel='Normalized MSE', yscale='log',
                title='Fitting is distinct from source-held error')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    directory = run.PRIVATE/'figures'
    directory.mkdir(exist_ok=True)
    fig.savefig(directory/'fitting_and_transfer.png', dpi=170)
    fig.savefig(directory/'fitting_and_transfer.pdf')
    plt.close(fig)


def source_coverage(analysis):
    s = run.source
    meta_path = s.checked(s.read(run.old.REG)['metadata'])
    metadata = s.read(meta_path)
    rows = []
    for c in analysis['contexts']:
        meta = next(m for m in metadata['contexts'] if m['job']==c['job'])
        s.checked(meta['metadata'])
        for f in c['folds']:
            row = next(m for m in meta['folds'] if m['source']==f['source'])
            parent = run.rel.base.read_checkpoint(s.checked(f['parent_checkpoint']))
            run.rel.base.exact(row['donor_sources'], parent['preprocess']['train_sources'])
            run.rel.base.exact(row['donor_rows'], f['fit_rows'])
            rows.append(dict(job=c['job'], **row))
    s.once(run.PUBLIC/'source_coverage.json', dict(result_source='cached_verified',
        metadata=s.artifact(meta_path), rows=rows, independent_physical_people_certified=False))
    lines = ['# Fitting-Source Coverage', '',
        'This is a cached-verified lineage diagnostic, not new data collection.',
        'Metadata hashes, parent training sources and fitting row counts agree.', '',
        '| Excluded source | Donor sources | Recordings | Windows | Conservative identity buckets |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['source']} | {len(r['donor_sources'])} | {r['donor_recordings']} | "
                     f"{r['donor_rows']} | {r['conservative_identity_buckets']} |")
    lines += ['', 'Two folds fit on one source component each; six fit on two components each.',
        'Many windows are overlapping observations, not independent domain evidence.',
        'Identity buckets conservatively merge matching local IDs across linked clips;',
        'they are not a certification of independent physical people.', '',
        'The small source076 fold fits 716 windows from two components and 112 identity',
        'buckets. Near-zero fitting loss on that bank does not identify conditional risk',
        'in a different component. This supports examining coverage and source shift;',
        'it does not prove that adding arbitrary data will solve the problem.', '',
        'Next: check whether additional already-authorized SOURCE forecasts have compatible',
        'teacher ancestry and risk-target definitions before pooling. Do not combine',
        'critic contexts blindly, relabel exposed data as independent, or open reserved data.']
    (run.PUBLIC/'source_coverage.md').write_text('\n'.join(lines)+'\n')


def main():
    s = run.source
    a = s.read(run.PUBLIC/'analysis.json')
    result = summarize(a)
    s.once(run.PUBLIC/'result_summary.json', result)
    lines = ['# Fixed-Budget Continuation: Optimization And Transfer', '',
        'I continued the eight saved risk heads while keeping architecture, features,',
        'optimizer settings, source separation, targets and decision rules fixed.',
        'Every fold received at least 4,096 additional updates, rounded to full epochs.',
        'Only the final checkpoint was evaluated after all eight fits and refits completed.', '',
        f"The registered information screen passes **{result['passing_contexts']}/3** contexts.",
        'This screen is not the original full-policy safety estimator. No deployment change.', '',
        '## Decision Results', '',
        '| Context | Model | Risk-MSE reduction % | Decision-cost reduction % | Admitted | Harmful |',
        '|---|---|---:|---:|---:|---:|']
    for c in result['contexts']:
        for name, v in c['arms'].items():
            lines.append(f"| {c['job'].split('_')[-1]} | {name} | {v['MSE_reduction_percent']:+.6f} | "
                         f"{v['cost_reduction_percent']:+.6f} | {v['admitted']} | {v['harmful']} |")
    lines += ['', 'Positive reductions are better; shared64 is the reference. Costs are source-macro',
        'known-proposal decision costs, not trajectory ADE/FDE improvements. Short and',
        'long mean the 32-epoch parent and fixed-budget continuation of the same network.', '',
        '## Fitting Versus Source-Held Error', '',
        '| Excluded source | Fit rows | Added updates | Final epochs | Fit loss, before | Fit loss, after | Held plain MSE, before | Held plain MSE, after |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for f in result['folds']:
        lines.append(f"| {f['source']} | {f['fit_rows']} | {f['extra_updates']} | {f['epochs']} | "
                     f"{f['start_fit_loss']:.6f} | {f['end_fit_loss']:.6f} | "
                     f"{f['short_held_MSE']:.6f} | {f['long_held_MSE']:.6f} |")
    lines += ['', f"Additional primary updates: {result['additional_updates']:,}. "
        f"Fitting loss improves in {result['fitting_loss_improved_folds']}/8 folds; "
        f"source-held plain MSE improves in {result['held_plain_MSE_improved_folds']}/8.",
        f"The registered last-quarter fitting plateau diagnostic holds in {result['operational_plateau_folds']}/8 folds.",
        'Neither completing the budget nor this operational diagnostic proves convergence.',
        'The full loss trace is retained; no checkpoint was chosen on held outcomes.', '',
        '## Evidence Boundaries', '',
        '- fresh_run: continued optimizer updates, full repeated continuations and final development readout.',
        '- cached_verified: parent networks, causal features, reference IDs, shared64 and track32 controls.',
        '- not_run: new trajectory training, new reserved evaluation, three-seed main result, deployment.',
        '- All 48,061 known proposals remain; 1,274 unknown outcomes are not made safe.',
        '- Original 2% all/easy harm, easy-net and support rules remain. Empty support is NA.',
        '- No identity exemption; the original 24-group primary remains NA, not passed.',
        '- Source-held own-B has already been exposed for development; replay is not independence.',
        '- Native obs8/pred12 image-coordinate silver labels, not metric or seconds-level.',
        '- No true3D, foundation, Stage5C, SMC or submission-ready claim.', '',
        '[Protocol](protocol.md) | [Loss trace](loss_trace.csv) | [Failure analysis](failure_analysis.md)',
        '| [Reproduction](operation_zh.md) | [Verification](verification_report.md)']
    (run.PUBLIC/'README.md').write_text('\n'.join(lines)+'\n')
    trace = []
    for path in sorted((run.PUBLIC/'fits').glob('*.json')):
        f = s.read(path)
        trace.extend(dict(job=f['job'], source=f['source'], **row) for row in f['extra_trace'])
    with (run.PUBLIC/'loss_trace.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(trace[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(trace)
    figures(result, trace)
    source_coverage(a)


if __name__=='__main__':
    main()
