"""Summarize the fixed model-side diagnostic without revising a safety rule."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_candidate_neighbor_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_candidate_neighbor_v1'
sys.path.insert(0, str(ROOT))


def summarize(result):
    rows = []
    for c in result['contexts']:
        a, b = (c['result']['summary'][arm] for arm in ('history_relations', 'plus_candidate_neighbors'))
        folds = []
        for f in c['result']['rows']:
            x, y = (f['arms'][arm] for arm in ('history_relations', 'plus_candidate_neighbors'))
            folds.append(dict(source=f['held_source'], known_rows=f['rows'], queries=f['queries'],
                MSE_improvement_percent=100*(1-y['normalized_MSE']/x['normalized_MSE']),
                joint_cost_improvement_percent=100*(1-y['total_joint_cost']/x['total_joint_cost']),
                true_safe_fraction=x['true_joint_safe_proposals']/x['known_proposals']))
        rows.append(dict(context=c['job'], own_B_rows=c['own_B_rows'], known_proposals=c['known_proposed_rows'],
            unknown_proposals=c['unknown_proposed_rows'], folds=folds,
            MSE_improvement_percent=100*c['result']['relative_MSE_improvement'],
            joint_cost_improvement_percent=100*(1-b['total_joint_cost']/a['total_joint_cost']),
            false_safe_cost_improvement_percent=100*(1-b['false_safe_cost']/a['false_safe_cost']),
            aligned_neighbor_available_fraction=c['valid_aligned_neighbor_row_fraction']))
    return dict(contexts=rows, source_folds=sum(len(r['folds']) for r in rows),
        fitted_ridge_probes=2*sum(len(r['folds']) for r in rows),
        own_B_rows=sum(r['own_B_rows'] for r in rows),
        known_proposals=sum(r['known_proposals'] for r in rows),
        unknown_proposals=sum(r['unknown_proposals'] for r in rows),
        followup_screen_passed_contexts=result['passing_contexts'],
        followup_neural_training_supported=result['followup_neural_feature_training_supported'],
        source='fresh_run_own_B_leave_source_probe', new_neural_training=False,
        original_rules_unchanged=True, identity_certification=False, new_V_or_H=False,
        independent_confirmation=False, deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False)


def main():
    result = json.loads((PUBLIC/'analysis.json').read_text())
    verification = json.loads((PUBLIC/'verification.json').read_text())
    sha = hashlib.sha256((PUBLIC/'analysis.json').read_bytes()).hexdigest()
    if verification['analysis_sha256'] != sha:
        raise ValueError('Verified current analysis required')
    summary = summarize(result)
    (PUBLIC/'result_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    lines = ['# Candidate-Neighbor Relations', '',
        'I tested whether explicit forecast-to-neighbor geometry explains the remaining intervention errors.',
        'The fixed leave-source probe does **not** support expanding this feature branch.', '',
        '| Context | Own-B rows | Known proposed rows | MSE improvement | Joint cost improvement |',
        '|---|---:|---:|---:|---:|']
    for r in summary['contexts']:
        lines.append(f"| {r['context'].split('_')[-1]} | {r['own_B_rows']} | {r['known_proposals']} | {r['MSE_improvement_percent']:+.3f}% | {r['joint_cost_improvement_percent']:+.3f}% |")
    lines += ['', 'Positive values mean lower error/cost. The internal follow-up screen passes in **0/3** contexts.',
        'There are eight source folds and sixteen fitted linear probes, not sixteen new neural networks.',
        'The 62,466 rows include overlapping windows and are not independent observations.',
        '48,061 known proposals enter the probe; 1,274 unknown proposed outcomes are recorded but not imputed.', '',
        '## What Changed', '',
        'The control uses kinematic64 and rollout/history32. The treatment adds 24 features describing',
        'the candidate and floor paths relative to neighbors continued with their last observed velocity.',
        'Only neighbors aligned to the last two ego observation times are used. Normalization and',
        'target RMS are refitted on each leave-source fitting subset; no held labels enter fitting.',
        'Both probes use the fixed ridge penalty 0.01. The treatment has more coefficients, so this',
        'is a feature-utility diagnostic, not an equal-capacity neural ablation.', '',
        '## What It Means', '',
        'Adding this particular proximity representation does not provide stable linear risk information.',
        'Two contexts worsen both prediction and decision cost. The third reduces decision cost 1.164%',
        'while worsening MSE 5.124%. This does not prove that all interaction features or nonlinear',
        'models are useless, nor does it establish domain shift as the sole root cause.',
        'The source-held relationship remains unstable; the next model intervention should target',
        'cross-source risk learning instead of a threshold sweep or more versions of the same feature.', '',
        '## Evidence Status', '',
        '- `fresh_run`: feature extraction, 16 ridge fits, frozen-neural error slices, exact replay.',
        '- `cached_verified`: original forecasts, neural checkpoints, source roles and causal schema.',
        '- `not_run`: a new neighbor neural branch, reserved/H confirmation and deployment.',
        '- Original 2% all/easy risk, easy-net, reference/support rules unchanged; empty support remains NA.',
        '- No identity certification. Prior 17-view neural risk/support failure remains 0/17.',
        '- Native obs8/pred12 image coordinates, not metric/seconds/true3D/foundation evidence.',
        '- Stage5C/SMC off. The paper is not yet submission-ready.', '',
        '![Diagnostic contrast](probe_comparison.svg)', '',
        '[Protocol](protocol.md) | [All metrics](analysis.json) | [Source folds](source_folds.md)',
        '| [Failure analysis](failure_analysis.md) | [Reproducibility](validation_receipt.md)', '']
    (PUBLIC/'README.md').write_text('\n'.join(lines))
    lines = ['# Own-B Source Folds', '', 'Known outcomes only; not independent confirmation.', '',
        '| Context | Held source | Known rows | Queries | MSE improvement | Joint cost improvement | True safe fraction |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in summary['contexts']:
        for f in r['folds']:
            lines.append(f"| {r['context'].split('_')[-1]} | {f['source'].split('_')[-1]} | {f['known_rows']} | {f['queries']} | {f['MSE_improvement_percent']:+.3f}% | {f['joint_cost_improvement_percent']:+.3f}% | {100*f['true_safe_fraction']:.2f}% |")
    lines += ['', 'All sources retained, including source065 with only 44 known proposals. Counts are descriptive;',
        'overlapping windows do not create independent statistical replicates. No confidence interval',
        'or deployment guarantee is inferred from these fold counts.', '']
    (PUBLIC/'source_folds.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    with plt.rc_context({'svg.hashsalt': 'candidate-neighbor-v1', 'font.size': 10}):
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        index = np.arange(3); width=.34
        ax.bar(index-width/2, [r['MSE_improvement_percent'] for r in summary['contexts']], width,
            label='Risk MSE improvement', color='#187e78')
        ax.bar(index+width/2, [r['joint_cost_improvement_percent'] for r in summary['contexts']], width,
            label='Joint decision cost improvement', color='#b74c62')
        ax.axhline(0, color='#555555', linewidth=.8)
        ax.set_xticks(index, ['Context0', 'Context1', 'Context2'])
        ax.set_ylabel('Improvement over history-relation probe (%)')
        ax.set_title('Candidate-neighbor features: own-B leave-source probes')
        ax.legend(loc='lower right'); ax.spines[['top', 'right']].set_visible(False)
        fig.savefig(PUBLIC/'probe_comparison.svg', metadata={'Date': None})
        fig.savefig(PRIVATE/'probe_comparison.png', dpi=160)
        plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k != 'contexts'}))


if __name__ == '__main__':
    main()
