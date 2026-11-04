"""Consolidate both completed neural loss studies without choosing an H winner."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import report_m3w_moving_adaptive_record_cost as report
from scripts.report_m3w_moving_harm_refinement import text_once
from scripts.report_m3w_moving_amplitude_cost import interval

source, PUBLIC = report.source, report.PUBLIC


def main():
    s = source.read(PUBLIC / 'summary.json')
    verification = source.read(PUBLIC / 'verification.json')
    if (not verification['all54neural_cost_arrays_and_policy_blocks_exact']
            or not verification['all_three_cached_parent_policy_blocks_exact']
            or verification['summary_sha256'] != source.sha(PUBLIC / 'summary.json')):
        raise ValueError('Complete full frozen comparison and replay first')
    rows = [row for ref in s['group_receipts'] for row in source.read(source.checked(ref))['rows']]
    sites = sorted({r['site'] for r in rows})
    if len(rows) != 138 or len(sites) != 23 or any(sum(r['site'] == site for r in rows) != 6 for site in sites):
        raise ValueError('All source, producer and seed views required')
    rates = {}
    for arm in ('point', 'shared_tree', 'frozen_teacher', 'adaptive'):
        by_source = {site: float(100*np.mean([r['policies'][arm]['joint_free']['switch_rate'] for r in rows if r['site'] == site]))
            for site in sites}
        rates[arm] = dict(source_macro_switch_percent=float(np.mean(list(by_source.values()))),
            by_source=by_source, role='descriptive_source_macro_not_IID_window_estimate_or_new_CI')
    parent = report.run.run.teacher.PUBLIC
    parent_summary = source.read(parent / 'summary.json')
    parent_v = source.read(parent / 'verification.json')
    if parent_v['summary_sha256'] != source.sha(parent / 'summary.json'):
        raise ValueError('Preserve complete parent result')
    support = source.read(PUBLIC.parent / 'moving_support_ceiling_v1/analysis.json')
    if support['required_record_views'] != 348 or support['zero_full_easy_reference_views'] != 15:
        raise ValueError('Do not change reference support or population')
    bindings = [source.artifact(path) for path in (PUBLIC / 'summary.json', PUBLIC / 'verification.json',
        parent / 'summary.json', parent / 'verification.json', PUBLIC / 'fitting_cost_diagnostics.json',
        PUBLIC / 'cost_diagnostics.json', PUBLIC.parent / 'moving_support_ceiling_v1/analysis.json')]
    source.once(PUBLIC / 'comparison_summary.json', source.stable(dict(bindings=bindings, rates=rates,
        primary_new_neural_cost_models=36, primary_optimizer_updates=72000, actual_seeds=[17, 29, 43],
        new_trajectory_models=0, independent_confirmation=False, deployment_changed=False, submission_ready=False,
        result_source='fresh_arithmetic_on_complete_hash_verified_fresh_neural_readouts_not_another_inference_run',
        stage5c_executed=False, smc_enabled=False)))
    effects = ['|Changed training objective and comparator|Known utility pp [95% CI]|Direct completion-lower pp [95% CI]|', '|---|---:|---:|']
    for label, value in (
        ('Frozen-teacher minus matched point', parent_summary['paired']['point']['joint_free']),
        ('Frozen-teacher minus shared tree', parent_summary['paired']['shared_tree']['joint_free']),
        ('Current-selection minus frozen-teacher', s['paired']['frozen_teacher']['joint_free']),
        ('Current-selection minus matched point', s['paired']['point']['joint_free']),
        ('Current-selection minus shared tree', s['paired']['shared_tree']['joint_free'])):
        effects.append(f'|{label}|{interval(value["known_pp"])}|{interval(value["completion_lower_pp"])}|')
    risks = ['|Joint-free arm|Support /348|Known all/easy violations|Worst defined easy-net %|Source-macro switch %|', '|---|---:|---:|---:|---:|']
    for arm in ('point', 'shared_tree', 'frozen_teacher', 'adaptive'):
        risk = s['safety'][arm]['joint_free']
        risks.append(f'|{arm}|{risk["supported_record_views"]}|{risk["known_all_risk_violations"]}/{risk["known_easy_risk_violations"]}'
            f'|{risk["worst_known_easy_net_degradation_percent"]:.4f}|{rates[arm]["source_macro_switch_percent"]:.4f}|')
    text = '''# Selected-Record Objectives: Two Complete Neural Tests

## Controlled Hypothesis

Lower row prediction error need not lower error in selected recording-level
cost masses. I tested that hypothesis with the existing 28,229-parameter neural
cost head, not a new trajectory, JEPA or multimodal dynamics architecture.
The frozen 9-forecaster EqMotion bank, physical floor, causal observations,
candidate fractions, original point loss, initialization, C-fit normalization,
query draws and 2,000-update budget stayed matched. The first arm added a
weight-one record-excess loss on the old teacher's fixed selection. A verified
C-only trigger then permitted exactly one follow-up: change that loss mask to
the current detached independent selection. No gradient passes through the
action choice. Both complete 18-model banks froze before new H readout.

The total is 36 fresh neural cost heads and 72,000 primary updates, three real
seeds, plus separately recorded pilots/refits. Original point controls are
cached trained models with fresh inference. Shared-tree costs are not freshly
inferred; their decisions are hash checked and all scalar controls replayed.
The adaptive comparison replays all 54 neural cost arrays and every control
policy block. A preallocation hash-format guard failure, corrected by a
separate runtime registration, is preserved; no models or thresholds changed.

## Complete Results, Not A Selected Success

''' + '\n'.join(effects) + '\n\n' + '\n'.join(risks) + '''

These are differences in utility **percentage points**, not relative-percent
improvements. Intervals use 3,000 source-group resamples after producer/seed
averaging. All 23 supported groups are exposed development, not independent
confirmation or verified IID physical sites. The original 24-group primary
remains not_estimable. None of these utility differences overrides risk.

The current-selection loss recovers known-label utility over the unsuccessful
frozen-teacher loss, but its direct lower interval includes zero. It still
loses to the original matched neural control. Its risk profile is not repaired.
Neither new head is deployed. The conservative shared tree preserves signed
easy net better, but also fails the complete original support/risk contract;
it is not promoted to a certified safe method either.

## The Failure Is Already Visible On C

The complete fitting diagnostic gives own-region known-harm actual/predicted
mass ratios 20.3885 for the frozen-teacher model and 36.6987 for the current-
selection model; easy-harm ratios are 29.5972 and 134.6385. The current mask
therefore has not repaired even its supervised fitting-region optimism.
On that same adaptive C selection, the frozen-teacher model has H/EH ratios
1.6363/0.7958. These cross-evaluations motivate separating proposal and risk
estimation; they do not prove that an ensemble or cross-fit will work.

On H joint selections, the adaptive model's own H/EH ratios remain
25.4586/72.6424. Ratios pool repeated-row masses and are descriptive, not
calibration or CIs. C uses independent masks and H uses joint masks, so their
difference is not an isolated domain-shift experiment. Selection-estimation
coupling is consistent with the evidence; exclusive causation is not proved.
The detached, sampled-record objective is neither a proper post-selection
calibration guarantee nor an unbiased full-video objective. Do not attribute
failure solely to new scenes, insufficient width or too few training minutes.

## Structural And Rare-Event Limits Remain Separate

Fifteen required recording views have zero full observed easy-reference mass.
Nonnegative selected reference cannot exceed full reference; at most 333/348
views could have positive support under unchanged labels/floor/event. This is
an upper bound, not achievable support. It does not extend the 330 single-row
witness diagnostic into a general joint-feasibility theorem. No view is dropped
and unavailable support does not become a safety pass.

The frozen-teacher worst defined easy-net view has only two easy windows. Its
reference mean is 6.7035 pixels and reconstructed mean ADE increase is 20.7533 pixels.
Sparse events matter, but that error is not merely a rounding-scale denominator.
These overlapping windows are not independent people. All costs are available-
label ADE in pixels, not full12-step error or physical danger. See the complete
denominator note; no H label from that case enters a new trained policy.

## What This Can Support In A Paper

The evidence supports a controlled negative result: adding record-excess loss
on teacher or moving self-selected masks does not establish safe admission.
It demonstrates that actual Torch training, exact recovery and a known-label
gain are different from completion-aware utility, selected-region calibration
and complete recording-risk support. General prediction/decision mismatch and
optimizer's curse are established ideas, not new discoveries here. See
[Smart Predict, then Optimize](https://pubsonline.informs.org/doi/10.1287/mnsc.2020.3922)
and [The Optimizer's Curse](https://pubsonline.informs.org/doi/10.1287/mnsc.1050.0451).
This is not SPO+ or a new safety theorem.

The next useful test is an honest source-excluded C proposal/risk evaluation,
with all learned preprocessing fitted only on its own fitting sources. Test
whether a separate critic detects optimistic proposals before committing a new
policy. Do not sweep more loss weights or thresholds on these H results.
Cross-evaluation alone cannot manufacture absent event support or independent
confirmation. New-source primary/calibration/test roles must be fixed before
their dependent outcomes; reserved 20 Moving groups, DUT and DroneCrowd stay
closed. Existing failures cannot be renamed as independent successes.

## Provenance And Limits

Fresh_run: both neural fits, registered held inference and complete replay.
Cached_verified: unchanged forecast bank, trained controls and solver choices.
Not_run: independent confirmation, new trajectory/JEPA training, and rebuilding
the main v9 PDF with this addendum. This Markdown supplement does not update
or certify the compiled paper or an anonymous cold-start package.

195,286 past-eligible SOURCE windows, 64 admitted recordings; 58 actual-query
recordings repeated across producer/seed views. Native observation8/prediction12
is supplied annotation-order causality, not verified sensor-as-of time. No
metric, seconds, human-gold, true3D, foundation, physical-safety, deployment or
submission-ready claim. Stage5C execution and SMC remain disabled.
'''
    text_once(PUBLIC.parent / 'evidence_manuscript_v9/selected_record_objective_addendum.md', text)
    print(source.json.dumps(dict(rates={a: v['source_macro_switch_percent'] for a, v in rates.items()},
        new_deployment=False, submission_ready=False), sort_keys=True))


if __name__ == '__main__':
    main()
