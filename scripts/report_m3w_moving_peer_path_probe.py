"""Report fixed source-held path probes, including negative and null controls."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_peer_path_probe as run

source, PUBLIC = run.source, run.PUBLIC


def reduction(a, b):
    return 100*(1-b/a) if a > 0 else None


def decision_counts(result):
    keys = ('accepted_known_proposals', 'true_joint_safe_proposals', 'false_safe_count', 'false_reject_count')
    return [dict(job=c['job'], known_rows=c['known_proposed_rows'],
        arms={arm:{k:sum(f['families']['extra_trees'][arm][k] for f in c['folds']) for k in keys}
            for arm in run.method.ARMS},
        source_macro_false_safe_cost={arm:c['families']['extra_trees']['summary'][arm]['false_safe_cost']
            for arm in run.method.ARMS}) for c in result['contexts']]


def summarize(result):
    if result['fits'] != 48 or len(result['contexts']) != 3 or result['primary_family'] != 'extra_trees':
        raise ValueError('Complete fixed48-cell study required')
    if (sum(len(c['folds']) for c in result['contexts']) != 8
            or any(len({r['held_source'] for r in c['folds']}) != len(c['folds']) for c in result['contexts'])):
        raise ValueError('Eight distinct within-context source folds required')
    rows = []
    for c in result['contexts']:
        for family in run.method.FAMILIES:
            a, b, null = (c['families'][family]['summary'][arm] for arm in run.method.ARMS)
            rows.append(dict(job=c['job'], family=family, own_B_rows=c['own_B_rows'], known_rows=c['known_proposed_rows'],
                unknown_rows=c['unknown_proposed_rows'], sources=len(c['sources']),
                MSE_reduction_percent=reduction(a['normalized_MSE'], b['normalized_MSE']),
                joint_cost_reduction_percent=reduction(a['total_joint_cost'], b['total_joint_cost']),
                false_safe_reduction_percent=reduction(a['false_safe_cost'], b['false_safe_cost']),
                MSE_reduction_vs_misaligned_percent=reduction(null['normalized_MSE'], b['normalized_MSE']),
                joint_cost_reduction_vs_misaligned_percent=reduction(null['total_joint_cost'], b['total_joint_cost']),
                screen_pass=c['families'][family]['passes_internal_screen']))
    return dict(rows=rows, primary_family='extra_trees', primary_passing_contexts=result['passing_contexts'],
        followup_neural_supported=result['followup_neural_supported'], fits=48, repeated_legacy_controls=8,
        known_rows=sum(c['known_proposed_rows'] for c in result['contexts']),
        unknown_rows=sum(c['unknown_proposed_rows'] for c in result['contexts']),
        own_B_rows=sum(c['own_B_rows'] for c in result['contexts']), source_folds=sum(len(c['folds']) for c in result['contexts']),
        neural_training=False, new_V_or_H=False, independent_confirmation=False, original_rules_unchanged=True,
        identity_certification=False, deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)


def figure(summary):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'svg.hashsalt':'moving_peer_path_probe_v1', 'svg.fonttype':'none', 'font.size':9})
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, family in zip(axes, run.method.FAMILIES):
        rows = [r for r in summary['rows'] if r['family'] == family]
        for i, (key, label, color) in enumerate((('MSE_reduction_percent', 'Risk MSE', '#187E78'),
                ('joint_cost_reduction_percent', 'Joint decision cost', '#B74C62'),
                ('MSE_reduction_vs_misaligned_percent', 'MSE vs mismatched', '#5577A0'))):
            ax.bar(np.arange(3)+(i-1)*.25, [r[key] if r[key] is not None else np.nan for r in rows],
                width=.25, label=label, color=color)
        ax.axhline(0, color='#444444', linewidth=.7)
        ax.set(xticks=np.arange(3), xticklabels=['Context0', 'Context1', 'Context2'],
            title=family+' (primary)' if family == 'extra_trees' else family+' (secondary)', ylabel='Reduction (%)')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle('Directional peer paths: source-held development probes, not deployment')
    out = PUBLIC/'probe_comparison.svg'; fig.savefig(out, metadata={'Date':None})
    out.write_text('\n'.join(line.rstrip() for line in out.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'probe_comparison.png', dpi=150); plt.close(fig)


def render():
    result = source.read(PUBLIC/'analysis.json'); verification = source.read(PUBLIC/'verification.json')
    run.rel.base.exact(verification['analysis_sha256'], source.sha(PUBLIC/'analysis.json'))
    if not verification['all48_refits_and_predictions_exact'] or not verification['all8_old_ridge_controls_exact']:
        raise ValueError('Full reproducibility checks required')
    summary = summarize(result); source.once(PUBLIC/'result_summary.json', summary); figure(summary)
    counts = decision_counts(result); source.once(PUBLIC/'decision_counts.json', counts)
    f = lambda v:'NA' if v is None else f'{v:+.3f}'
    text = ['# Directional Peer Paths', '',
        'I tested whether directional path information lost by proximity summaries',
        'helps predict intervention risk on an excluded fitting source. The primary',
        'nonlinear probe uses fixed ExtraTrees; ridge is a secondary diagnostic.',
        'A within-query mismatched association control checks whether correctly',
        'assigning the new geometry matters. It is not an independent test set.', '',
        '| Family | Context | MSE reduction % | Joint-cost reduction % | False-safe-cost reduction % | MSE reduction vs mismatched % |',
        '|---|---|---:|---:|---:|---:|']
    for r in summary['rows']:
        text.append('| '+r['family']+' | '+r['job'].split('_')[-1]+' | '+' | '.join(f(r[k]) for k in
            ('MSE_reduction_percent', 'joint_cost_reduction_percent', 'false_safe_reduction_percent', 'MSE_reduction_vs_misaligned_percent'))+' |')
    text += ['', 'Positive reductions are better. Source-macro costs retain their original',
        'sum-over-channels semantics; these are not held trajectory ADE improvements.',
        f"The primary internal follow-up screen passes **{summary['primary_passing_contexts']}/3** contexts.",
        'A passing information screen could justify a later neural experiment, not',
        'deployment. A favorable secondary ridge result cannot replace the primary.', '',
        '![Source-held comparisons](probe_comparison.svg)', '', '## Decisions Behind The Error', '',
        '| Context | Known candidates | True joint-safe labels | Summary accepted / harmful | Aligned accepted / harmful | Mismatched accepted / harmful |',
        '|---|---:|---:|---:|---:|---:|']
    for c in counts:
        text.append('| '+c['job'].split('_')[-1]+f" | {c['known_rows']} | "+
            str(c['arms']['summary120']['true_joint_safe_proposals'])+' | '+
            ' | '.join(f"{c['arms'][a]['accepted_known_proposals']} / {c['arms'][a]['false_safe_count']}"
                for a in run.method.ARMS)+' |')
    text += ['', 'These are descriptive per-row signs on known fitting outcomes, not a',
        'full-policy safety certificate. A joint-safe label is not a deployment pass.',
        'All rejected candidates can lower regression error without improving any',
        'decision. Zero harmful cost has an undefined relative reduction (NA),',
        'not a new safety certification. The context1 relative harmful-cost increase',
        'has a small nonzero denominator; absolute costs are retained in',
        '[decision_counts.json](decision_counts.json).', '', '## What Was Run', '',
        '48 fixed probe fits over eight source folds:24ExtraTrees and24ridge. Eight',
        'ridge cells reproduce previously completed summary120 controls exactly.',
        'A separate full64-tree pilot repeats exactly. All48model structures,',
        'coefficients, predictions and metrics are reproduced by fresh refits.', '',
        f"Own-B rows:{summary['own_B_rows']}; known proposals:{summary['known_rows']}; unknown proposals:{summary['unknown_rows']}.",
        'Unknown outcomes are counted, not imputed. Row features and the fixed',
        'misassociation are constructed before the outcome-availability filter.',
        'Overlapping windows and these three contexts are not independent seeds.', '',
        '| Context | Queries | Aligned-neighbor rows % | Singleton queries unchanged by mismatch |',
        '|---|---:|---:|---:|']
    for c in result['contexts']:
        text.append(f"| {c['job'].split('_')[-1]} | {c['complete_queries']} | {100*c['aligned_neighbor_row_fraction']:.2f} | {c['own_B_rows']-c['shifted_rows']} |")
    text += ['', '## Scope And Reproduction', '',
        '- fresh_run: causal path extraction,48probe fitting executions and deterministic refits.',
        '- cached_verified: source data, frozen forecasts, splits and earlier linear controls.',
        '- not_run: new neural branch, nested-C/H, reserved data and independent confirmation.',
        '- Original2% all/easy harm, easy-net and support rules unchanged; empty support remains NA.',
        '- No identity certification, threshold selection or deployment promotion.',
        '- Original24-group primary NA; Moving20,DUT,DroneCrowd,H remain closed.',
        '- Native obs8/pred12 silver image coordinates, not metric/seconds/true3D/foundation.',
        '- Stage5C/SMC off. Not yet submission-ready.', '',
        'Saved ExtraTrees models use n_jobs=4. The inherited runtime banner has',
        'a legacy forest_threads=1 label; the registered and saved estimator settings,',
        'not that generic label, describe this probe. No DataLoader multiprocessing.', '',
        '[Protocol](protocol.md) | [All metrics](analysis.json) | [Failure analysis](failure_analysis.md)',
        '| [Reproduction](operation_zh.md)', '']
    (PUBLIC/'README.md').write_text('\n'.join(text))
    lines = ['# Every Source Fold', '', 'Exposed source-held fitting diagnostic, not independent confirmation.', '',
        '| Context | Family | Source | Known rows | Queries | MSE reduction % | Joint-cost reduction % | MSE reduction vs mismatched % |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for c in result['contexts']:
        for fold in c['folds']:
            for family in run.method.FAMILIES:
                a,b,n = (fold['families'][family][arm] for arm in run.method.ARMS)
                lines.append('| '+c['job'].split('_')[-1]+' | '+family+' | '+fold['held_source'].split('_')[-1]+
                    f" | {fold['rows']} | {fold['queries']} | "+' | '.join(f(v) for v in (
                        reduction(a['normalized_MSE'],b['normalized_MSE']), reduction(a['total_joint_cost'],b['total_joint_cost']),
                        reduction(n['normalized_MSE'],b['normalized_MSE'])))+' |')
    lines += ['', 'All sources are retained. Very small source support and dependent queries limit inference.',
        'Known-label probes do not certify missing-label full-policy harm. No safety pass follows from this table.', '']
    (PUBLIC/'source_folds.md').write_text('\n'.join(lines))
    print(source.json.dumps({k:v for k,v in summary.items() if k != 'rows'}))


def main():
    with (run.PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        render()


if __name__ == '__main__':
    main()
