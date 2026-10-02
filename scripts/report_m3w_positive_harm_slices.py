"""Deterministic development slices of the completed, fixed three-arm experiment."""
import json
from pathlib import Path
import numpy as np
from scripts.verify_m3w_positive_harm import verify, sha

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_positive_harm_v1'


def reduce_rows(rows):
    out = dict(heads=len(rows), arms={})
    for arm in ('original', 'additive', 'positive'):
        r = [row['result']['policies'][arm] for row in rows]
        upper = [v['easy_selected_risk_upper'] for v in r if v['easy_selected_risk_upper'] is not None]
        known = [v['selected_known_easy_harm_mass']/v['selected_known_easy_reference_mass']
                 for v in r if v['selected_known_easy_reference_mass'] > 0]
        out['arms'][arm] = dict(selected=sum(v['selected_count'] for v in r),
            selected_unknown=sum(v['selected_unknown'] for v in r),
            complete_support=sum(v['finite_completion_supported'] for v in r),
            easy_risk_defined=len(upper), easy_upper_violations=sum(v > .02+1e-12 for v in upper),
            known_easy_risk_defined=len(known), known_easy_violations=sum(v > .02+1e-12 for v in known),
            worst_easy_upper=max(upper) if upper else None,
            worst_known_easy_risk=max(known) if known else None)
    out['contrast_mean_within_fixed_views'] = {
        k: float(np.mean([r['result']['contrasts'][k] for r in rows]))
        for k in rows[0]['result']['contrasts']}
    return out


def report(rows):
    summary = dict(by_source={s: reduce_rows([r for r in rows if r['source'] == s])
        for s in sorted({r['source'] for r in rows})},
        by_head_seed={str(s): reduce_rows([r for r in rows if r['head_seed'] == s])
        for s in sorted({r['head_seed'] for r in rows})})
    summary['training'] = dict(
        max_gradient=max(r['training']['maximum_gradient'] for r in rows),
        max_training_mean_error=max(r['training']['train_relative_mean_error'] for r in rows),
        max_Newton_iterations=max(r['training']['max_iterations'] for r in rows),
        populated_tree_leaves=sum(r['training']['original_leaves_reconstructed'] for r in rows),
        zero_target_leaves_total_easy=np.sum([r['training']['zero_target_leaves_by_channel'] for r in rows], 0).tolist(),
        loss_before_mean_over_heads=np.mean([r['training']['training_loss']['before'] for r in rows], 0).tolist(),
        loss_after_mean_over_heads=np.mean([r['training']['training_loss']['after'] for r in rows], 0).tolist(),
        loss_units='normalized conditional Poisson deviance plus fixed regularization; not ADE/FDE',
        leaves_repeated_across_trees_and_source_views=True)
    summary['diagnostic_repeated_row_occurrences'] = {
        arm: {key: sum(r['diagnostic'][arm][key] for r in rows) for key in rows[0]['diagnostic'][arm]}
        for arm in ('additive', 'positive')}
    for mode in ('full', 'matched'):
        a = 'original' if mode == 'full' else 'original_matched_positive'
        b = 'positive' if mode == 'full' else 'positive_matched_original'
        summary[mode+'_support_transition'] = dict(
            retained=sum(r['result']['policies'][a]['finite_completion_supported'] and r['result']['policies'][b]['finite_completion_supported'] for r in rows),
            lost=sum(r['result']['policies'][a]['finite_completion_supported'] and not r['result']['policies'][b]['finite_completion_supported'] for r in rows),
            gained=sum(not r['result']['policies'][a]['finite_completion_supported'] and r['result']['policies'][b]['finite_completion_supported'] for r in rows))
    summary.update(result_source='fresh_run_aggregate_of_completed_registered_reports',
                   exploratory_development_only=True, independent_confirmation=False,
                   source_seed_selection=False, deployment_changed=False)
    return summary


if __name__ == '__main__':
    verification = verify()
    complete = json.loads((PUBLIC/'complete.json').read_text())
    rows = [json.loads((ROOT/r['path']).read_text()) for r in complete['groups']]
    result = report(rows)
    result['summary_sha256'] = verification['summary_sha256']
    result['complete_sha256'] = sha(PUBLIC/'complete.json')
    path = PUBLIC/'slices.json'; content = json.dumps(result, indent=2)+'\n'
    if path.exists(): assert path.read_text() == content
    else: path.write_text(content)
    print(json.dumps({k: v for k, v in result.items() if k != 'by_source'}, indent=2))
