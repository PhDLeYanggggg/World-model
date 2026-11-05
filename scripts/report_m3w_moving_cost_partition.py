"""Report every fixed partition comparison without promoting a secondary arm."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_cost_partition as run

source, PUBLIC = run.source, run.PUBLIC


def reduction(a, b):
    return 100*(1-b/a) if a > 0 else None


def summarize(a):
    if (a['cells'] != 24 or len(a['contexts']) != 3 or a['fresh_primary_fits'] != 16
            or a['cached_controls'] != 8 or sum(len(c['folds']) for c in a['contexts']) != 8):
        raise ValueError('Complete registered24-cell comparison required')
    rows, decisions = [], []
    for c in a['contexts']:
        if len({f['held_source'] for f in c['folds']}) != len(c['folds']):
            raise ValueError('Distinct within-context held sources required')
        means = c['summary']['source_macro']; baseline = means['mse_partition']
        for arm in run.method.ARMS:
            v = means[arm]
            rows.append(dict(job=c['job'], arm=arm, primary=arm == 'cost_partition',
                MSE_reduction_percent=reduction(baseline['normalized_MSE'], v['normalized_MSE']),
                joint_cost_reduction_percent=reduction(baseline['total_joint_cost'], v['total_joint_cost']),
                false_safe_cost_reduction_percent=reduction(baseline['false_safe_cost'], v['false_safe_cost']),
                absolute_false_safe_cost=v['false_safe_cost'], absolute_joint_cost=v['total_joint_cost']))
            counts = {key:sum(f['arms'][arm]['metrics'][key] for f in c['folds']) for key in
                ('known_proposals', 'true_joint_safe_proposals', 'accepted_known_proposals', 'false_safe_count', 'false_reject_count')}
            decisions.append(dict(job=c['job'], arm=arm, **counts))
    decompositions = [f['arms'][arm]['decomposition'] for c in a['contexts'] for f in c['folds'] for arm in run.method.ARMS]
    return dict(rows=rows, decisions=decisions, primary_passing_contexts=a['primary_passing_contexts'],
        neural_mechanism_followup_supported=a['neural_mechanism_followup_supported'],
        max_factorization_error=max(d['same_partition_factorization_max_abs_difference'] for d in decompositions),
        max_original_reconstruction_error=max(d['original_leaf_mean_reconstruction_max_abs_difference'] for d in decompositions),
        factorization_sign_changes=sum(d['direct_vs_factorized_joint_sign_changes'] for d in decompositions),
        reconstruction_sign_changes=sum(d['original_vs_reconstructed_joint_sign_changes'] for d in decompositions),
        own_B_rows=sum(c['own_B_rows'] for c in a['contexts']), known_rows=sum(c['known_proposed_rows'] for c in a['contexts']),
        unknown_rows=sum(c['unknown_proposed_rows'] for c in a['contexts']),
        fresh_models=16, cached_controls=8, neural_training=False, new_V_or_H=False,
        independent_confirmation=False, original_rules_unchanged=True, identity_certification=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)


def figure(summary):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'svg.hashsalt':'cost_partition_v1', 'svg.fonttype':'none', 'font.size':9})
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, key, title in zip(axes, ('joint_cost_reduction_percent', 'MSE_reduction_percent'), ('Joint decision cost', 'Risk MSE')):
        for i, arm in enumerate(run.method.ARMS[1:]):
            rows = [r for r in summary['rows'] if r['arm'] == arm]
            axis.bar(np.arange(3)+(i-.5)*.35, [r[key] if r[key] is not None else np.nan for r in rows],
                width=.35, label=arm, color=('#267F77', '#B35169')[i])
        axis.set(xticks=np.arange(3), xticklabels=['Context0', 'Context1', 'Context2'], ylabel='Reduction vs MSE partition (%)', title=title)
        axis.axhline(0, color='#444444', linewidth=.7); axis.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8); fig.suptitle('Source-held fitting diagnostics, not deployment')
    out = PUBLIC/'partition_comparison.svg'; fig.savefig(out, metadata={'Date':None})
    out.write_text('\n'.join(line.rstrip() for line in out.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'partition_comparison.png', dpi=150); plt.close(fig)


def render():
    a = source.read(PUBLIC/'analysis.json'); v = source.read(PUBLIC/'verification.json')
    run.rel.base.exact(v['analysis_sha256'], source.sha(PUBLIC/'analysis.json'))
    if not all(v[k] for k in ('all16_new_full_refits_exact', 'all8_legacy_full_refits_exact', 'all24_metric_replays_exact')):
        raise ValueError('All saved models must reproduce before reporting')
    summary = summarize(a); source.once(PUBLIC/'result_summary.json', summary); figure(summary)
    fmt = lambda x:'NA' if x is None else f'{x:+.4f}'
    text = ['# Cost-Directed Partitions', '',
        'I changed how the model divides fitting examples, keeping the same causal',
        'features, risk targets and zero decision boundary. The final readout remains',
        'the original weighted two-channel mean, not a classifier probability.', '',
        '| Context | Partition | MSE reduction % | Joint-cost reduction % | False-safe-cost reduction % |',
        '|---|---|---:|---:|---:|']
    for r in summary['rows']:
        text.append('| '+r['job'].split('_')[-1]+' | '+r['arm']+' | '+' | '.join(fmt(r[k]) for k in
            ('MSE_reduction_percent','joint_cost_reduction_percent','false_safe_cost_reduction_percent'))+' |')
    text += ['', 'Positive reductions are better. NA means a zero reference cost, not a pass.',
        f"The primary cost-partition follow-up screen passes **{summary['primary_passing_contexts']}/3** contexts.",
        'It requires two; the sign-only arm cannot replace the registered primary.', '',
        '![Fixed partition comparisons](partition_comparison.svg)', '', '## Decisions', '',
        '| Context | Arm | Known proposals | Joint-safe labels | Accepted | False safe | False reject |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in summary['decisions']:
        text.append('| '+r['job'].split('_')[-1]+' | '+r['arm']+' | '+' | '.join(str(r[k]) for k in
            ('known_proposals','true_joint_safe_proposals','accepted_known_proposals','false_safe_count','false_reject_count'))+' |')
    text += ['', 'These signs concern known source-held fitting labels, not a complete-policy',
        'safety certificate. Unknown outcomes remain unknown; zero intervention is',
        'not an identity exemption. Absolute costs are retained in result_summary.json.', '',
        '## Same-Partition Identity', '',
        'For each channel, P(positive) times positive severity minus P(negative)',
        'times negative severity reconstructs the original weighted mean when the',
        'partition and measure are unchanged. This arithmetic does not establish',
        'that separately trained hurdle models must behave identically.', '',
        f"Maximum factorization difference: {summary['max_factorization_error']:.4g}; sign changes: {summary['factorization_sign_changes']}.",
        f"Maximum original-mean reconstruction difference: {summary['max_original_reconstruction_error']:.4g}; sign changes: {summary['reconstruction_sign_changes']}.", '',
        '## Evidence Scope', '',
        f"Own-B rows: {summary['own_B_rows']}; known proposals: {summary['known_rows']}; unknown proposals: {summary['unknown_rows']}.",
        '- fresh_run: 16 new partition models, full refits and target/decision diagnostics.',
        '- cached_verified: eight MSE controls and frozen source inputs; controls also refit exactly.',
        '- not_run: new neural model, nested-C/H, independent confirmation or deployment.',
        '- One producer seed; three exposed contexts and dependent windows are not three independent seeds.',
        '- Original 2% harm, easy-net and support rules unchanged; no identity certification.',
        '- Empty support stays NA; prior failures and original24-group primary NA remain.',
        '- Moving20, DUT, DroneCrowd and H remain closed; Stage5C/SMC off.',
        '- Native obs8/pred12 silver image coordinates, not metric/seconds/true3D/foundation.',
        '- Not submission-ready. No scientific gate is established by exact replay.', '',
        '[Protocol](protocol.md) | [Failure analysis](failure_analysis.md) | [Reproduction](operation_zh.md)', '']
    (PUBLIC/'README.md').write_text('\n'.join(text))
    lines = ['# Every Source Fold', '',
        'Training complements overlap; these are descriptive source-held fitting comparisons, not independent tests.', '',
        '| Context | Held source | Arm | Held known rows | Fitting cost reduction % | Held cost reduction % | Fitting accepted | Held accepted | Held false safe |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for c in a['contexts']:
        for fold in c['folds']:
            base = fold['arms']['mse_partition']
            for arm in run.method.ARMS:
                v = fold['arms'][arm]; train, held = v['fitting_metrics'], v['metrics']
                lines.append('| '+c['job'].split('_')[-1]+' | '+fold['held_source'].split('_')[-1]+' | '+arm+
                    f" | {held['known_proposals']} | "+fmt(reduction(base['fitting_metrics']['total_joint_cost'],train['total_joint_cost']))+
                    ' | '+fmt(reduction(base['metrics']['total_joint_cost'],held['total_joint_cost']))+
                    f" | {train['accepted_known_proposals']} | {held['accepted_known_proposals']} | {held['false_safe_count']} |")
    lines += ['', 'Original full-policy safety/support is not evaluated or certified by these known-row signs.', '']
    (PUBLIC/'source_folds.md').write_text('\n'.join(lines))
    print(source.json.dumps({k:v for k,v in summary.items() if k not in ('rows','decisions')}))


def main():
    with (run.PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB); render()


if __name__ == '__main__':
    main()
