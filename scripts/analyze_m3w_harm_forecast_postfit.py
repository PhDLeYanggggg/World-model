"""Reproducible diagnosis from fixed public current-producer metrics only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.m3w_harm_forecast_postfit import recording_budget_counts, paired_fit_losses

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_dimensionless_harm_forecast_v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def immutable(path, text, verify):
    if path.exists():
        if path.read_text() != text:
            raise ValueError('Post-fit analysis changed: ' + str(path))
    elif verify:
        raise ValueError('Verification requires an existing result')
    else:
        path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    load = lambda name: json.loads((PUBLIC/name).read_text())
    analysis, freeze = load('analysis.json'), load('training_freeze.json')
    support = load('support_diagnostic.json')
    verification, support_verification = load('verification.json'), load('support_verification.json')
    if (verification['analysis_sha256'] != digest(PUBLIC/'analysis.json')
            or verification['record_csv_sha256'] != digest(PUBLIC/'record_diagnostics.csv')
            or not verification['exact_analysis_replay']
            or support_verification['summary_sha256'] != digest(PUBLIC/'support_diagnostic.json')
            or not support_verification['exact_replay']):
        raise ValueError('Exact upstream replay required')
    with (PUBLIC/'record_diagnostics.csv').open() as f:
        records = list(csv.DictReader(f))
    risk = recording_budget_counts(records)
    losses = paired_fit_losses(freeze)
    primary = analysis['summaries']['ADE_all']['paired_improvement_pp_CV']
    localities = primary['by_site']
    result = dict(result_source='fresh_run_postfit_analysis_of_cached_verified_fixed_readouts',
        inputs={name:digest(PUBLIC/name) for name in ('analysis.json', 'training_freeze.json',
            'record_diagnostics.csv', 'verification.json', 'support_diagnostic.json', 'support_verification.json')},
        code_sha256=digest(Path(__file__)),
        api_sha256=digest(ROOT/'src/evaluation/m3w_harm_forecast_postfit.py'),
        unique_recordings=len({(r['source'], r['recording']) for r in records}),
        repeated_views_per_arm_subset_reference=978,
        counts_are_repeated_context_seed_views_not_independent=True,
        recording_diagnostics=risk, fit_loss_diagnostics=losses,
        source_localities_improved=sum(v > 0 for v in localities.values()),
        source_localities_worsened=sum(v < 0 for v in localities.values()),
        equal_source_ADE_paired_pp=primary,
        seed_ADE_paired_pp={k:v['ADE_all']['paired_improvement_pp_CV']['point']
                           for k,v in analysis['seeds'].items()},
        original_control_metric_checks=analysis['frozen_control_metric_checks'],
        support_old=support['old_budget_feasible'], support_new=support['new_budget_feasible'],
        support_gained=support['fixed_views_gained'], support_lost=support['fixed_views_lost'],
        support_is_label_aware_not_deployable=True, new_training_steps=0,
        independent_roles_opened=False, deployment_changed=False,
        full_policy_safety_certified=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False)
    immutable(PUBLIC/'postfit_diagnosis.json', json.dumps(result, indent=2)+'\n', args.verify)
    lines = ['# Fixed-Endpoint Failure Diagnosis', '',
        'Post-fit analysis of verified source-development results; no new model selection or threshold search.',
        'All nine endpoints, all three seeds, all twelve localities and all 163 recordings retained.', '',
        '## Matched Predictive Effect', '',
        f"Equal-locality ADE change: {primary['point']:.6f} pp versus the original model; "
        f"conditional 95% locality-bootstrap CI {primary['ci95']}.",
        f"Localities improved: {result['source_localities_improved']}; worsened: {result['source_localities_worsened']}.",
        f"Seed-specific ADE changes (pp): {result['seed_ADE_paired_pp']}.", '',
        '## Known Recording-Level Risk', '',
        'The 978 views repeat 163 recordings across two excluded producers and three seeds.',
        'These are unrestricted forecast diagnostics, not the final intervention policy or safety certification.',
        'Missing outcomes are not completed as zero. Known net benefit cannot cancel positive harm.', '',
        '| Arm / subset / reference | Views | Known <=2% | Known >2% | Not estimable | Net degradation >2% | Views with missing labels |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key, row in risk.items():
        lines.append(f"| {key} | {row['required_views']} | {row['known_budget_passes']} | "
            f"{row['known_budget_failures']} | {row['not_estimable']} | "
            f"{row['known_net_degradation_gt2']} | {row['views_with_unknown_labels']} |")
    lines += ['', '## Training Diagnostics', '',
        'Same scheduled minibatches; last ten logged batches only, not whole-TRAIN or validation loss.',
        'Compare the native ADE term, not total objectives with different penalty definitions.', '',
        '| Fit | Original native ADE | New native ADE | New selected harm | New CV harm |',
        '|---|---:|---:|---:|---:|']
    for row in losses:
        lines.append(f"| {row['trial']} | {row['old_native_ADE']:.6f} | {row['new_native_ADE']:.6f} | "
            f"{row['new_selected_positive_harm']:.6f} | {row['new_CV_positive_harm']:.6f} |")
    lines += ['', '## Fixed-Floor Support', '',
        f"Original {result['support_old']}/148; new {result['support_new']}/148; "
        f"gained {result['support_gained']}, lost {result['support_lost']}.",
        'The old oracle, frozen CV/damping.97 floor and original eligibility replay exactly.',
        'The remaining unsupported view is retained. An oracle is not an inference policy.', '',
        'The aggregate penalty reduces harm but sacrifices overall/hard prediction quality. ',
        'It is not a recording-level relative-risk constraint. Its train-selected reference is also ',
        'not the CV/damping.97 deployment floor. These facts motivate a training-only reference/constraint ',
        'repair, but do not prove a unique cause or a general impossibility result.', '',
        'Image-pixel/raw-frame detector-track evidence only. No independent confirmation, deployment, ',
        'Stage5C, SMC, metric, seconds, human-gold, true3D or foundation claim.', '']
    immutable(PUBLIC/'postfit_diagnosis.md', '\n'.join(lines), args.verify)
    print(json.dumps(dict(exact_replay=args.verify, diagnosis_sha256=digest(PUBLIC/'postfit_diagnosis.json'),
                          diagnosis_markdown_sha256=digest(PUBLIC/'postfit_diagnosis.md'))))


if __name__ == '__main__':
    main()
