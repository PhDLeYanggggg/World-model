"""Read verified cost-study receipts without selecting policies or changing gates."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.m3w_recording_lineage import sha256

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_shared_motion_cost_v1'


def read(path):
    return json.loads(path.read_text())


def value(number):
    return 'not estimable' if number is None else f'{number:+.4f}'


def interval(item):
    if item['point'] is None or item['CI95'] is None:
        return 'not estimable'
    return f"{value(item['point'])} [{value(item['CI95'][0])}, {value(item['CI95'][1])}]"


def render(summary, training, execution):
    if (summary['original24group_primary'] != 'not_estimable_not_passed'
            or not summary['matched_utility_is_not_safety_gate']
            or not summary['count_matching_within_arm_not_between_feature_arms']
            or not summary['feature_effect_uses_full_original_population']):
        raise ValueError('Preserve original primary and utility-versus-safety distinction')
    lines = ['# Shared Observed-Motion Cost Study', '',
        '## Evidence Status', '',
        'Fresh_run: complete real Torch fitting of 36 cost heads, excluded-source',
        'cost inference and original predicted-budget joint allocations.',
        'Cached_verified: complete saved-head prediction/scalar replay and saved',
        'solver feasibility. Replay does not claim bitwise MILP reoptimization.',
        'The first complete head in each feature arm has an exact independent refit.',
        'This report only formats verified receipts; it does not fit or select.', '',
        '**The original 24-group primary is not estimable and has not passed.**',
        'The 23 supported-group results are development secondary evidence.',
        'They are not independent geographic-site confirmation or deployment.', '',
        '## Fixed Information Contrast', '',
        'EqMotion was fixed before the parent forecast readout. Each cost head',
        'uses 380 causal local/rollout columns plus 48 observed-motion slots.',
        'Only the added 48 slots are zeroed in the local control after a common',
        'critic-fitting normalization. Capacity, initialization, loss, source draws',
        'and budget are paired. These are not camera pose or image features.',
        'Producer, critic-fitting and held source groups are pairwise disjoint.', '',
        '## Whole-Population Feature Effect', '',
        'Shared-motion minus local-history gain, percentage points. The fixed',
        'known reference denominator is retained; completion lower bounds account',
        'for unknown outcomes without imputed ground-truth positions.', '',
        '| Policy | Known-label paired effect [95% CI] | Completion-lower effect [95% CI] |',
        '|---|---:|---:|']
    for policy, metrics in summary['paired_shared_minus_local'].items():
        lines.append(f"| {policy} | {interval(metrics['known_pp'])} | {interval(metrics['completion_lower_pp'])} |")
    lines += ['', 'These are 3,000 source-group bootstrap resamples after producer and',
        'three actual training-seed averages, not IID-window confidence intervals.',
        'The joint-free completion-lower contrast was fixed prospectively; other',
        'policy views do not replace it after inspecting their scores.', '',
        '## Accuracy And Original Full Risk', '',
        '| Arm | Policy | Known gain (%) | Completion-lower gain (%) | Supported recording views | Known all/easy positive-harm violations | Worst known easy net degradation (%) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for arm, policies in summary['safety'].items():
        for policy, risk in policies.items():
            known = summary['known_accuracy'][arm][policy]['point']
            lower = summary['completion_lower'][arm][policy]['point']
            lines.append(f"| {arm} | {policy} | {value(known)} | {value(lower)} | "
                f"{risk['supported_record_views']}/{risk['required_record_views']} | "
                f"{risk['known_all_risk_violations']}/{risk['known_easy_risk_violations']} | "
                f"{value(risk['worst_known_easy_net_degradation_percent'])} |")
    lines += ['', 'Recording views repeat seeds and excluded producers; they are not that',
        'many independent places. Undefined support is not a known risk violation',
        'and is not a safety pass. All required full-policy views retain the',
        'unchanged 2% positive-harm and easy-degradation requirements.', '',
        '## Same-Count Utility Only', '',
        'Count matching is within each feature arm against its own independent',
        'policy, not between feature arms. Eligibility is the complete query where',
        'the causal independent intervention count is positive. Future labels do',
        'not determine membership. Empty controls are not estimable, not zero.', '',
        '| Arm | Nonempty / empty query views | Nonestimable source views | Known utility [95% CI] | Completion-lower utility [95% CI] |',
        '|---|---:|---:|---:|---:|']
    for arm, metrics in summary['within_arm_nonempty_matched_utility'].items():
        support = summary['matched_utility_support'][arm]
        lines.append(f"| {arm} | {support['nonempty_query_views']}/{support['empty_query_views']} | "
            f"{support['nonestimable_source_views']} | {interval(metrics['known_pp'])} | "
            f"{interval(metrics['completion_lower_pp'])} |")
    lines += ['', 'These controls measure utility and never replace whole-population safety.',
        'A not-estimable contrast is not silently averaged over remaining sources.', '',
        '## Compute And Reproduction', '',
        f"Complete heads: {training['complete_heads']}; primary updates: {training['optimizer_updates']:,}.",
        f"Actual train invocation: {execution['seconds']:.3f}s; peak RSS: {execution['peak_RSS_bytes']:,} bytes.",
        'Native arm64 Torch CPU4, interop1, workers0. Checkpoints and solver',
        'receipts stay private; code, configurations and light scalar receipts',
        'are versioned. Pilot work and first-head refits are additional verification',
        'work, not extra primary updates.', '',
        '## Limitations And Decision', '',
        'All-current crowd summaries may add information but are not calibrated',
        'ego-motion. Joint allocation pools predicted budgets; it does not impose',
        'pairwise geometric consistency or physical collision constraints.',
        'Partial-label ADE is not complete 12-step truth. Source components are',
        'not verified IID geographic sites. Calibration and confirmation remain',
        'closed. No promotion follows from a positive secondary mean alone.',
        'Image pixels/native annotation steps are not verified metres/seconds.',
        'No true3D, foundation, successful world-model or submission-ready claim.',
        'Deployment unchanged. Stage5C execution and SMC remain off.']
    return '\n'.join(lines)+'\n'


def main():
    summary, training, verification = (read(PUBLIC/name) for name in
        ('summary.json', 'training_freeze.json', 'verification.json'))
    execution = read(PUBLIC/'train_execution.json')
    if (training['complete_heads'] != 36 or not training['all18pairs_exact']
            or not execution['real_Torch_training']
            or not verification['all36cost_predictions_and_policy_scores_exact']
            or not verification['full_first_refits_exact']
            or verification['summary_sha256'] != sha256(PUBLIC/'summary.json')):
        raise ValueError('Complete matched real fits and independent replay required')
    read(PUBLIC/'evaluate_execution.json')
    read(PUBLIC/'verify_execution.json')
    for item in (*training['fits'], *summary['group_receipts']):
        if sha256(ROOT/item['path']) != item['sha256']:
            raise ValueError('Changed complete source receipt')
    for item in training['fits']:
        receipt = read(ROOT/item['path'])
        checkpoint = receipt['checkpoint']
        if sha256(ROOT/checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('Changed trained cost checkpoint')
    (PUBLIC/'readout.md').write_text(render(summary, training, execution))
    (PUBLIC/'readout_sources.json').write_text(json.dumps(dict(
        result_source='formatting_hash_verified_complete_fit_and_replay_receipts_not_new_training',
        summary_sha256=sha256(PUBLIC/'summary.json'),
        training_sha256=sha256(PUBLIC/'training_freeze.json'),
        verification_sha256=sha256(PUBLIC/'verification.json'),
        original_primary='not_estimable_not_passed',
        independent_confirmation=False, new_deployment=False,
        stage5c_executed=False, smc_enabled=False), indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
