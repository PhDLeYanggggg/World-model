"""Pool verified diagnostic moment masses, never replace per-recording gates."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.evaluation.m3w_selected_cost_audit import ratio

OUT = ROOT/'outputs/publication_readiness_2026_09/moving_shared_motion_cost_v1/selected_cost_audit'


def main():
    audit = json.loads((OUT/'analysis.json').read_text())
    verified = json.loads((OUT/'verify_execution.json').read_text())
    if not verified['every_scalar_replayed_exact'] or len(audit['rows']) != 72:
        raise ValueError('Complete72role/arm views and exact diagnostic replay required')
    summary = []
    for role in ('critic_fitting', 'held'):
        for arm in ('local_history', 'shared_motion'):
            records = [r for r in audit['rows'] if r['role'] == role and r['arm'] == arm]
            if len(records) != 18:
                raise ValueError('All18jobs required in each role/feature arm')
            for policy in records[0]['policies']:
                values = [r['policies'][policy] for r in records]
                actual = np.sum([v['realized_selected_known_moments'] for v in values], 0)
                bounded = np.sum([v['bounded_label_selected_moments'] for v in values], 0)
                same_rows = np.sum([v['predicted_on_selected_known_moments'] for v in values], 0)
                predicted = np.sum([v['predicted_full_selected_moments'] for v in values], 0)
                percent = lambda h,r:None if r <= 0 else 100*float(h/r)
                summary.append(dict(role=role, arm=arm, policy=policy, views=len(values),
                    selected_row_views_not_unique=sum(v['selected'] for v in values),
                    predicted_full_all_risk_percent=percent(predicted[1], predicted[2]),
                    predicted_full_easy_risk_percent=percent(predicted[4], predicted[3]),
                    known_realized_all_risk_percent=percent(actual[1], actual[2]),
                    known_realized_easy_risk_percent=percent(actual[4], actual[3]),
                    known_H_over_predicted_H_same_rows=ratio(actual[1], same_rows[1]),
                    known_EH_over_predicted_EH_same_rows=ratio(actual[4], same_rows[4]),
                    bounded_H_over_predicted_H=ratio(bounded[1], predicted[1]),
                    bounded_EH_over_predicted_EH=ratio(bounded[4], predicted[4]),
                    predicted_all_budget_violating_query_views=sum(v['violations']['all']['predicted_budget_violating_queries'] for v in values),
                    known_all_budget_violating_query_views=sum(v['violations']['all']['known_realized_budget_violating_queries'] for v in values)))
    (OUT/'pooled_diagnostic.json').write_text(json.dumps(dict(rows=summary,
        population='pooled_repeated_job_views_not_independent_rows_or_official_macro_gate',
        result_source='cached_verified_scalar_formatting_not_new_training',
        no_policy_selection=True), indent=2, sort_keys=True, allow_nan=False)+'\n')
    with (OUT/'pooled_diagnostic.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    number = lambda x:'not estimable' if x is None else f'{x:.3f}'
    lines = ['# Selected-Cost Reliability Diagnosis', '',
        'This is a post-readout diagnostic, not a preregistered new test or',
        'independent confirmation. All72job/role/feature views replay exactly.',
        'No fitting, policy selection, threshold or source role changes.', '',
        'Each row pools18repeated job views. These are mass ratios, not official',
        'equal-source-group averages or independent sample counts. Future labels',
        'are used only after the frozen causal choices, to diagnose their errors.',
        'Known H/EH calibration compares actual and predicted masses on the same',
        'known rows; prediction budgets use all causally selected rows.', '',
        '| Role | Arm | Policy | Predicted all/easy risk (%) | Known realized all/easy risk (%) | Known H / predicted H | Known EH / predicted EH |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['role']} | {r['arm']} | {r['policy']} | "
            f"{number(r['predicted_full_all_risk_percent'])}/{number(r['predicted_full_easy_risk_percent'])} | "
            f"{number(r['known_realized_all_risk_percent'])}/{number(r['known_realized_easy_risk_percent'])} | "
            f"{number(r['known_H_over_predicted_H_same_rows'])} | {number(r['known_EH_over_predicted_EH_same_rows'])} |")
    lines += ['', '## Diagnosis', '',
        'Global moment fit does not ensure reliability after selecting predictions',
        'with very low estimated harm. The independent policy substantially',
        'underpredicts selected harm already on critic-fitting sources; held',
        'shift is not the sole explanation. Known-label violations also rule out',
        'missing outcomes as the sole cause. This does not uniquely establish',
        'whether sparse-tail weighting, decoder saturation, optimization or',
        'insufficient causal information caused the conditional failure.', '',
        'Do not add unrelated features or repeat a width/duration sweep from this',
        'table alone. The next controlled test should first check selected-set',
        'cost self-consistency on critic-fitting sources, retaining unknown bounds',
        'and the original2%requirements. Any prospective training repair needs',
        'a separate frozen definition; this diagnostic itself trains nothing.', '',
        'The original per-recording support/risk failures remain. Pooling does not',
        'rescue empty views or permit deployment. No new independent evidence,',
        'metric/seconds,true3D/foundation,Stage5Cexecution or SMC claim.']
    (OUT/'readout.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
