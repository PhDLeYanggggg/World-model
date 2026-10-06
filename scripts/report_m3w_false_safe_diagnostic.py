"""Hash-check and summarize TRAIN-only false-safe diagnostics without new fits."""
import hashlib
import json
from pathlib import Path
import statistics

from scripts import diagnose_m3w_temporal_false_safe as run

HOME = run.ROOT/run.REL/run.DIAG


def summarize(rows):
    out = {}
    for arm in ('none', 'rowmean', 'temporal'):
        selected = [r for r in rows if r['arm'] == arm]
        known = [r['result']['known_selected'] for r in selected]
        defined = [r for r in known if r['known_positive_easy_risk'] is not None]
        components = {}
        for key in ('predicted_slack_mass', 'harm_underprediction_mass',
                    'reference_overprediction_budget_mass', 'realized_excess_mass'):
            localities = {}
            for row in selected:
                v = row['result']['known_selected']; den = v['actual_easy_reference_mass']
                localities.setdefault(row['identity']['source'], []).append(100*v[key]/den if den > 0 else None)
            valid = all(all(v is not None for v in vs) for vs in localities.values())
            means = {site: statistics.mean(vs) for site, vs in localities.items()} if valid else None
            components[key] = dict(equal_locality_mean_pp=statistics.mean(means.values()) if means else None,
                localities=means, defined_for_every_view=valid)
        if all(v['defined_for_every_view'] for v in components.values()):
            terms = [v['equal_locality_mean_pp'] for v in components.values()]
            assert abs(sum(terms[:3])-terms[3]) < 1e-7
        out[arm] = dict(views=len(selected), selected_occurrences=sum(r['result']['selected'] for r in selected),
            unknown_selected_occurrences=sum(r['result']['selected_unknown'] for r in selected),
            defined_easy_risk_views=len(defined), undefined_easy_risk_views=len(known)-len(defined),
            easy_risk_violations=sum(r['known_positive_easy_risk'] > .02 for r in defined),
            median_known_easy_risk_percent=100*statistics.median(r['known_positive_easy_risk'] for r in defined) if defined else None,
            harm_underpredicted_views=sum(r['harm_underprediction_mass'] > 0 for r in defined),
            reference_overpredicted_views=sum(r['reference_overprediction_budget_mass'] > 0 for r in defined),
            positive_query_excess_occurrences=sum(r['result']['query']['positive_excess_groups'] for r in selected),
            selected_known_query_occurrences=sum(r['result']['query']['selected_known_groups'] for r in selected),
            positive_recording_excess_occurrences=sum(r['result']['recording']['positive_excess_groups'] for r in selected),
            selected_known_recording_occurrences=sum(r['result']['recording']['selected_known_groups'] for r in selected),
            scalar_contributions_plus_sum_identity_checks=sum(r['result'][c]['scalar_checks'] for r in selected for c in ['known_all','known_selected']),
            scalar_sum_identity_assertions=2*len(selected),
            components=components)
    return out


def verified_rows():
    complete = json.loads((HOME/'complete.json').read_text())
    assert complete['registration_sha256'] == run.sha(HOME/'registration.json')
    rows = []; found = set()
    for ref in complete['heads']:
        path = (run.ROOT/ref['path']).resolve()
        assert path.is_relative_to(HOME.resolve()/'heads') and run.sha(path) == ref['sha256']
        row = json.loads(path.read_text()); key = row['identity']['group'], row['identity']['seed'], row['arm']
        assert key not in found; found.add(key)
        assert row['optimizer_updates'] == 0 and not row['validation_scored'] and not row['independent_roles_read']
        rows.append(row)
    assert len(rows) == 216 and len({r['identity']['source'] for r in rows}) == 12
    return rows, complete


def main():
    rows, complete = verified_rows(); summary = summarize(rows)
    ratios = {}
    for arm in ('none', 'rowmean', 'temporal'):
        ratios[arm] = {}
        for cohort in ('known_all', 'known_selected'):
            values = [r['result'][cohort] for r in rows if r['arm'] == arm]
            valid = [v for v in values if v['actual_easy_harm_mass'] > 0]
            ratios[arm][cohort] = dict(defined_views=len(valid), undefined_views=len(values)-len(valid),
                median_predicted_actual_harm_ratio=statistics.median(
                    v['predicted_easy_harm_mass']/v['actual_easy_harm_mass'] for v in valid) if valid else None)
    run.once(HOME/'harm_ratio_audit.json', dict(complete_sha256=run.sha(HOME/'complete.json'),
        result_source='fresh_run_aggregate_TRAIN_diagnostic', ratios=ratios))
    out = dict(result_source='fresh_run_aggregate_analysis_of_frozen_model_TRAIN_replay', arms=summary,
        complete_sha256=run.sha(HOME/'complete.json'), source_localities=12,
        views_per_arm=72, row_occurrences_not_independent=True, train_resubstitution_only=True,
        new_optimizer_updates=0, new_validation_evaluation=False, independent_confirmation=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    run.once(HOME/'summary.json', out)
    lines = ['# TRAIN False-Safe Component Diagnosis', '',
        'Fresh inference on the frozen 216 heads and their original 24 TRAIN packets.',
        'This is resubstitution, not generalization or independent confirmation.',
        'Source/seed views and selected occurrences overlap; they are not independent samples.', '',
        '| Arm | Risk-defined views | Known easy-risk violations | Median risk (%) | Unknown selected occurrences |',
        '|---|---:|---:|---:|---:|']
    for arm, s in summary.items():
        median = s['median_known_easy_risk_percent']
        value = 'undefined' if median is None else f'{median:.6f}'
        lines.append(f"| {arm} | {s['defined_easy_risk_views']}/72 | {s['easy_risk_violations']} | {value} | {s['unknown_selected_occurrences']} |")
    lines += ['', '## Signed Decomposition', '',
        'Equal-locality averages of source/seed-view components, divided by actual',
        'selected known easy reference mass, in percentage points. Negative terms',
        'are retained. Undefined support invalidates an aggregate, rather than dropping views.', '',
        '| Arm | Predicted slack | Harm underprediction | Budgeted reference overprediction | Realized excess |',
        '|---|---:|---:|---:|---:|']
    keys = ('predicted_slack_mass','harm_underprediction_mass','reference_overprediction_budget_mass','realized_excess_mass')
    for arm, s in summary.items():
        vals = [s['components'][k]['equal_locality_mean_pp'] for k in keys]
        lines.append('| '+arm+' | '+' | '.join('undefined' if v is None else f'{v:+.6f}' for v in vals)+' |')
    lines += ['', 'The first three columns sum to realized excess above the unchanged 2% budget.',
        'These are signed error contributions, not FDE improvements or physical safety probabilities.', '',
        '## Interpretation Limits', '',
        '- Failure on TRAIN would rule out pure unseen-domain shift as the sole explanation.',
        '- The decomposition locates numerical error; it does not prove a repair will generalize.',
        '- A reduced global quadratic loss does not guarantee calibration after action selection.',
        '- The frozen full development readout remains failed. No threshold or budget is changed.',
        '- Unknown outcomes remain in inference. Known risk is not a complete-support safety guarantee.',
        '- Detector-silver image-local raw-frame data; no metric, seconds, true-3D or foundation claim.',
        '- Stage5C and SMC remain off.', '',
        f"Job: {complete['job_id']}; elapsed: {complete['elapsed_seconds']:.3f} seconds;",
        f"independent scalar sum-identity assertions: {sum(s['scalar_sum_identity_assertions'] for s in summary.values()):,}.",
        'The raw per-head field `scalar_checks` counts scalar summands plus one identity,',
        'not that many independent assertions. This report distinguishes those counts.', '']
    path = HOME/'report.md'; text = '\n'.join(lines)
    if path.exists():
        assert path.read_text() == text
    else:
        with path.open('x') as f: f.write(text)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
