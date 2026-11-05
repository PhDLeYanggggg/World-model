"""Render the fixed retrieval comparison without fitting or selecting a model."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_track_analogue as run

ARMS = ('shared64', 'prior', 'track32', 'misaligned32')


def reduction(value, reference):
    return 100 * (1 - value / reference) if reference > 0 else None


def summary(analysis):
    contexts = []
    for context in analysis['contexts']:
        means = context['summary']['source_macro']
        old = means['shared64']
        arms = {}
        for arm in ARMS:
            values = means[arm]
            arms[arm] = dict(values,
                MSE_reduction_percent=reduction(values['normalized_MSE'], old['normalized_MSE']),
                cost_reduction_percent=reduction(values['total_joint_cost'], old['total_joint_cost']),
                admissions=sum(f['metrics'][arm]['accepted_known_proposals'] for f in context['folds']),
                harmful_admissions=sum(f['metrics'][arm]['false_safe_count'] for f in context['folds']))
        contexts.append(dict(job=context['job'], arms=arms,
            cost_reduction_vs_misaligned_percent=reduction(means['track32']['total_joint_cost'],
                means['misaligned32']['total_joint_cost']),
            passes=context['summary']['passes_information_screen']))
    return dict(contexts=contexts, passing_contexts=analysis['passing_contexts'],
        safety_certificate=False, independent_confirmation=False,
        deployment_changed=False, submission_ready=False)


def fmt(value, digits=6):
    return 'NA' if value is None else f'{value:+.{digits}f}'


def render(analysis, result):
    lines = ['# Cross-Source History Retrieval', '',
        'I tested whether nearby causal histories from other sources could identify',
        'useful interventions missed by the fitted risk forests. The fixed model',
        'uses one window per source/agent identity bucket and 32 buckets per query.',
        'It excludes the entire query source from references and preprocessing.', '',
        '**History matching helps relative to a misaligned control, but does not',
        'repair safe intervention.**', '',
        '| Context | Model | MSE reduction % | Decision-cost reduction % | Admitted | Harmful |',
        '|---|---|---:|---:|---:|---:|']
    for context in result['contexts']:
        for arm, row in context['arms'].items():
            lines.append(f"| {context['job'].split('_')[-1]} | {arm} | "
                f"{fmt(row['MSE_reduction_percent'])} | {fmt(row['cost_reduction_percent'])} | "
                f"{row['admissions']} | {row['harmful_admissions']} |")
    lines += ['', 'Positive reductions are better; reference is the shared64 forest.',
        f"The registered information screen passes **{result['passing_contexts']}/3** contexts.",
        'Two were required. Source-macro decision costs are not trajectory ADE',
        'improvements or the full scene-level safety estimator. No model is promoted.', '',
        '## Matching Versus The Null', '',
        '| Context | Cost reduction versus misaligned matching % |', '|---|---:|']
    for c in result['contexts']:
        lines.append(f"| {c['job'].split('_')[-1]} | {fmt(c['cost_reduction_vs_misaligned_percent'])} |")
    lines += ['', 'The fixed within-source permutation preserves the unweighted prediction',
        'multiset and admission counts, not all query-weighted moments. It is one',
        'null draw, not a significance test. Better than this null does not imply',
        'better than the strong control or compliance with the original risk rules.', '',
        '## Every Excluded Source', '',
        '| Source | Known rows | Track32 admitted | Harmful | Selected normalized easy harm/reference % |',
        '|---|---:|---:|---:|---:|']
    for c in analysis['contexts']:
        for f in c['folds']:
            selected = f['boundary']['decisions']['new']['accept']
            ratio = selected['selected_easy_risk']
            lines.append(f"| {f['source']} | {f['rows']} | {selected['rows']} | "
                f"{selected['true_joint_unsafe_rows']} | {fmt(None if ratio is None else 100*ratio)} |")
    lines += ['', 'These ratios use only known selected proposal labels, with causal row',
        'normalization and source/query weighting. They are descriptive, not the',
        'full recording-level physical risk or a replacement for the support gate.',
        'NA stays NA, including admitted rows without easy reference support.', '',
        '## Evidence And Scope', '',
        '- fresh_run: lineage metadata and all 48,061 source-excluded risk predictions.',
        '- cached_verified: original 120-feature caches, targets, weights and eight shared64 controls.',
        '- not_run: new neural trajectories, full-policy evaluation, new nested-C/H and independent confirmation.',
        '- All eight complete retrievals reproduce exactly with a different chunk size.',
        '- The 1,274 unknown proposals remain audited, not assigned safe labels.',
        '- Identity buckets may merge different people; relabeled duplicates may remain.',
        '- Overlapping windows and identity buckets are not independent samples.',
        '- Original 2% positive-harm/easy-net and support requirements are unchanged.',
        '- No identity-fallback certification. Original24-group primary remains NA, not passed.',
        '- Reserved Moving20, DUT and DroneCrowd remain closed.',
        '- Native obs8/pred12 image coordinates and silver labels; no metric or seconds claim.',
        '- No true3D, foundation, Stage5C, SMC, deployment or submission-ready claim.', '',
        '[Protocol](protocol.md) | [Distance diagnostics](distance_analysis.md) |',
        '[Failure analysis](failure_analysis.md) | [Reproduction](operation_zh.md)']
    return '\n'.join(lines) + '\n'


def render_distances(analysis):
    lines = ['# Distance And Source Diagnostics', '',
        'All bins were fixed before readout. Distances are standardized causal',
        'feature-space distances, not metres or calibrated semantic similarity.',
        'Unsafe here means at least one positive signed risk target, not collision.',
        'The following source/query-weighted fractions are descriptive. They do',
        'not establish independent calibration or irreducible uncertainty.', '',
        '| Source | Nearest-distance interval | Rows | Predicted unsafe % | Observed unsafe % |',
        '|---|---|---:|---:|---:|']
    for context in analysis['contexts']:
        for fold in context['folds']:
            for b in fold['diagnostics']['distance_slices']:
                hi = 'inf' if b['upper'] is None else str(b['upper'])
                p, y = b['predicted_unsafe_fraction'], b['unsafe_fraction']
                lines.append(f"| {fold['source']} | [{b['lower']}, {hi}) | {b['rows']} | "
                    f"{fmt(None if p is None else 100*p)} | {fmt(None if y is None else 100*y)} |")
    lines += ['', 'No nearest-distance cutoff is selected from this table. Empty bins stay NA.',
        'Source076 underestimates signed-risk exceedance even in the [0.25,0.5)',
        'and [0.5,1) bins; excluding only very distant histories would not explain',
        'away the observed failures. Source067 instead overestimates the probability',
        'of exceedance in those bins. A single global distance-to-risk interpretation',
        'is not supported. Probability and harm severity remain different quantities.', '',
        'Mixed neighbor labels are common, but 32 references make mixing likely.',
        'This does not prove that the future is fundamentally unpredictable.']
    return '\n'.join(lines) + '\n'


def main():
    analysis = run.source.read(run.PUBLIC / 'analysis.json')
    verification = run.source.read(run.PUBLIC / 'verification.json')
    run.rel.base.exact(verification['analysis_sha256'], run.source.sha(run.PUBLIC / 'analysis.json'))
    result = summary(analysis)
    run.source.once(run.PUBLIC / 'result_summary.json', result)
    (run.PUBLIC / 'README.md').write_text(render(analysis, result))
    (run.PUBLIC / 'distance_analysis.md').write_text(render_distances(analysis))


if __name__ == '__main__':
    main()
