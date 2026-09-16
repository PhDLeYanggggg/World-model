# Matched Development Evaluation of the Deferral Control

2026-09-16. This slice connects the implemented cost-sensitive deferral head to the existing development evaluator. Verification is synthetic. Real fitting, development results and independent confirmation remain **not_run** under the unapproved real protocol.

## Why This Change Matters

A trained comparison head is not yet a comparison experiment. Previously the deferral control could be fitted and loaded, but could not be evaluated alongside the five M3W control arms. Listing the same producer models also does not establish that two heads saw the same training queries. Both gaps are now checked in the executable development path.

This remains an adaptation of existing regression-deferral work, not a new contribution. Its objective and literature boundary are documented in the [training control report](../deferral_control/method_and_limits.md). No superiority claim follows from adding this comparator.

## Input-Budget Identity

`m3w_oof_identity.py` fingerprints each query's recording, agent, current frame and requested horizon, its causal feature vector, producer ID and checkpoint hash, protocol, baseline and metric. Records are sorted before combination, so batch/fold order does not change the identity. Targets are excluded: gain/harm and absolute-cost targets differ by design. Duplicate queries, fractional IDs, inconsistent feature dimensions and nonfinite features are refused.

Both training CLIs export this identity. Deferral checkpoints bind it, and the development evaluator requires exact agreement with the gain/harm report. It also checks actual head/comparison-forecaster/producer seeds, producer architecture, fit-only preprocessing, parent artifacts and baseline identity before reading development labels. Old reports without this evidence cannot be silently promoted to matched-input results. Hashes detect drift and check declared provenance; they do not establish the truth of arbitrary manually forged declarations.

## Inference And Evaluation

Enable the comparison only through approved `development_evaluation.diagnostic_controls = ["cost_sensitive_deferral"]`. Every candidate then needs `deferral_head_id`, `deferral_report_path` and `deferral_report_sha256`. The unchanged real draft does not enable this option.

The evaluator verifies the entire learned family before opening any development labels and keeps one candidate model in memory at a time. Every scene query runs the forecaster once; all six arms use the same baseline/candidate arrays and past-only feature tensors. The deferral decision is made before label access. Agents with incomplete future labels remain in the decision population and intervention-rate denominator. Invalid candidate forecasts fall back without dropping agents; missing FDE endpoints remain missing.

Deferral uses its fixed argmax and finite-prediction support. It does **not** borrow M3W gain/harm thresholds to make itself appear safer. Its logit margin is not a calibrated probability. Reference support/budget satisfaction is measured afterwards, explicitly not enforced by this arm. Reports distinguish shared forecasts/queries/training inputs from shared budget caps. The earlier exact-coverage joint control remains a separate diagnostic and is not implied by this comparison.

The existing `budget_violation_agent_queries` count repeats the scene-level constraint flag for its members. It counts agent queries in reference-constraint-violating scene queries, not individual realized-harm events.

The diagnostic is not eligible for automatic deployment selection, and no new calibration or confirmation arm is registered. Existing five-arm selection remains unchanged. Formal comparator tuning and frozen final-family inclusion still require approved rules; a minimally trained diagnostic is not the strongest tuned deferral baseline.

## Paired Statistics

For each ordinary arm versus deferral, report paired mean error differences on all/easy/hard slices, with approved aggregation and a shared physical-scene bootstrap. Negative left-minus-right error favors the ordinary arm. One scene yields no interval, not a window-level pseudo-replication. Intervals are development-descriptive, conditional on fitted models and not multiple-comparison-adjusted, calibrated risk bounds or independent confirmation. Equal prediction inputs do not imply equal coverage, expected risk, actual risk or compute cost.

## Executed Evidence

[Synthetic integration evidence](synthetic_integration.json) records a production-CLI run in a temporary synthetic fixture: three forecasters trained for eight updates each, ridge fitting and an eight-update deferral head, identical **64 OOF rows / 306 features**, then six-arm evaluation on **27 agent queries / 16 scene queries**. Fifteen queries have complete ADE/FDE labels. One physical scene cannot supply a CI.

The undertrained synthetic deferral has normalized ADE 0.17265 versus floor 0.14323; this negative result is retained. All-past-supported intervention is 55.56%, versus 26.67% among complete labels. Differing denominators illustrate why future-complete windows must not define inference membership. These are wiring/negative-path checks, not evidence against the published method, for M3W superiority or for real generalization. The fixture has no hard-slice support.

Final related regression: **258 passed in 85.11 s**, including 15 new cases. Tests cover changed input budgets, fake normalization, seed mismatch, absent protocol opt-in, whole-family rejection before first labels, future corruption, invalid forecasts, exact completed resume and deliberate cache corruption. A separate two-scene constructed statistic verifies that 100 windows from one scene do not outweigh one window from another under equal-scene aggregation. The two initial regression cases failed on the previously missing interface, then passed after implementation. See [verification.json](verification.json).

## Remaining Evidence

Real preflight still refuses `Explicit protocol approval required`. No scientific choices were filled in, no final test labels were opened, and no historical result was rehabilitated. CREATE still needs the pending authentication/project details; no job was submitted. Unrelated staged changes are preserved and no cache, checkpoint or third-party data is committed.

The substantive study still needs the already-requested temporal/data-role decisions, independent scenes and a frozen matched experiment. Reuse artifacts only when source/protocol identities match; preserve old versions rather than editing their hashes. Real multi-seed gains, useful calibration and final public-baseline results remain missing. Stage5C/SMC are off; dataset-local/raw-frame results carry no new metric, seconds, physical-safety, foundation or CVPR-readiness claim.
