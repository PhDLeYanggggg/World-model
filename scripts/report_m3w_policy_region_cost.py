"""Read-only evidence export for the frozen region-moment training comparison."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import report_m3w_cost_decision_evidence as evidence

NAME = 'european_policy_region_cost_v1'
PUBLIC = evidence.PUBLIC / NAME


def build():
    previous = evidence.STUDIES
    try:
        evidence.STUDIES = (NAME,)
        value = evidence.build()
    finally:
        evidence.STUDIES = previous
    summary = json.loads((PUBLIC / 'summary.json').read_text())
    value.update(
        objective='fixed_teacher_record_region_cost_moments',
        region_minus_parent={scope: summary['scopes'][scope]['region_minus_parent']
                             for scope in ('fit', 'inner_held')},
        unchanged_primary_full_roster_budget_supported=summary['primary_full_roster_budget_supported'],
        unchanged_neural_increment_conditional_CI_positive=summary['neural_increment_conditional_CI_positive'],
        no_threshold_or_harm_multiplier_selected=True)
    return value


def interval_text(row):
    if row['point'] is None or row['CI95'] is None:
        return 'not_estimable'
    a, b = row['CI95']
    return f"{row['point']:.6f} [{a:.6f},{b:.6f}]"


def render(value):
    text = ['# Policy-Region Moments: Fixed Readout', '',
        'This export verifies completed training receipts,exact inference replay and',
        'the lossless aggregate bundle. It performs no training,threshold selection,',
        'new bootstrap or independent confirmation. The old scene head is the control.', '',
        '## Inner-Held Primary Banks', '',
        '| Bank | Known ADE gain % [conditional95%CI] | Supported record/head views | Undefined easy | Selected (unknown) |',
        '|---|---:|---:|---:|---:|']
    for row in value['policies']:
        if row['scope'] != 'inner_held' or row['policy'] not in ('physical_only', 'physical_neural', 'neural_only'):
            continue
        cell = interval_text(dict(point=row['known_ADE_gain_floor_percent'], CI95=row['conditional_CI95_percent']))
        text.append(f"| {row['policy']} | {cell} | {row['supported_record_head_views']}/{row['required_record_head_views']} | "
                    f"{row['undefined_easy_record_head_views']} | {row['selected']} ({row['selected_unknown']}) |")
    text += ['', '## Paired Mixed-Bank Contrasts', '',
        'Percentage points;completion bounds retain unknown outcomes. Matching is',
        'utility-only and does not certify the full policy safety requirement.', '',
        '| Contrast | Known | Completion lower | Completion upper |', '|---|---:|---:|---:|']
    contrasts = {
        'region minus old scene': value['region_minus_parent']['inner_held'],
        'mixed minus physical-only': value['inner_held_mixed_minus_physical'][NAME]}
    for label, rows in contrasts.items():
        for mode in ('full', 'matched'):
            cells = [interval_text(rows[f'{mode}_{kind}_pp']) for kind in ('known', 'lower', 'upper')]
            text.append('| ' + label + ' ' + mode + ' | ' + ' | '.join(cells) + ' |')
    fit = value['training'][NAME]
    text += ['', '## Actual Training And Unchanged Gates', '',
        f"{fit['fits']} new native Torch cost heads,{fit['updates']:,} primary updates,",
        f"{fit['seconds']:.6f}s,peak RSS {fit['peak_RSS_bytes']:,}bytes.",
        'All initializations,physical-fitting normalizers,draws and seeds are matched.',
        'The trajectory producer is frozen;three seeds refer to controller heads,',
        'not three complete independently retrained forecasting pipelines.', '',
        'Full required-recording all/easy2% gate: ' + str(value['unchanged_primary_full_roster_budget_supported']),
        'Positive completion-lower neural-increment CI: ' + str(value['unchanged_neural_increment_conditional_CI_positive']),
        '', 'The registered primary multiplier is1. Other fixed sensitivities are not',
        'selected for deployment. Empty risk remains not_estimable,not safe zero.',
        'SOURCE inner-held is exposed development,not independent confirmation.',
        'The3,000-locality intervals are conditional;record/head repetitions are not',
        'independent recordings. The fitting-region mechanism check is separate.',
        'Obs8/pred12/rawstride12 webcam detector-track pixel coordinates are not',
        'metric/seconds,human gold,true3D or foundation evidence. No deployment',
        'promotion,Stage5C execution,SMC or submission readiness is asserted.', '']
    return '\n'.join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    value = build()
    products = {PUBLIC/'evidence.json': json.dumps(value, indent=2)+'\n', PUBLIC/'evidence.md': render(value)}
    for path, content in products.items():
        if args.check or path.exists():
            if path.read_text() != content:
                raise ValueError('Preserve frozen region evidence; a new revision is required')
        else:
            path.write_text(content)
    print(json.dumps(dict(exact_check=args.check, new_training=False,
        full_roster_supported=value['unchanged_primary_full_roster_budget_supported'],
        neural_increment_supported=value['unchanged_neural_increment_conditional_CI_positive'])))


if __name__ == '__main__':
    main()
