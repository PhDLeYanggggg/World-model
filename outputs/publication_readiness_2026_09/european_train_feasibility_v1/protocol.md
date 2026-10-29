# TRAIN Feasibility Before Another Policy Fit

## Question

Does the fixed candidate forecast admit useful actions that meet the approved
2% all/easy selected positive-harm and easy-degradation constraints in every
fitting recording? Or is the current objective unsupported even for an oracle?

No new split, budget, threshold, model, or prediction is introduced. All
source-qualified fitting recordings remain, including unsupported ones. The
frozen origin, exact-gradient, same-record and cross-record policies are replayed
on fitting data only. Three head seeds repeat each of 24 source/context packets;
oracle counts are also reported without those seed repeats. They are not
independent samples. Original-TRAIN inner-held recordings are not scored.

Physical legacy packets include both fitting and inner-held original-TRAIN rows.
The runner filters by the already fixed recording identities before preprocessing,
model scoring or oracle analysis. It does not claim the archive was physically
unread or that historical teacher exposure has been repaired. Independent
selection/calibration/confirmation roles remain unopened.

## Exact Diagnostic

The verified five-moment target is `(benefit,harm,reference,easy_reference,easy_harm)`.
Benefit and harm are mutually exclusive. This allows a closed-form **label-aware**
maximum-completion-lower-utility witness for a fixed set of allowed actions:

1. Include every allowed known row with strictly positive benefit. These incur
   zero harm and maximize utility before supported-denominator constraints.
2. If at least one has positive easy reference, all three budget checks pass.
3. Otherwise choose the single allowed known easy row with minimum harm among
   those with `easy_harm/easy_reference <= .02` (same numerical tolerance as the
   frozen evaluator). This anchors the easy denominator; it may have zero gain.
4. If none exists, supported feasibility is impossible for this action set.
   A weighted average of exclusively excessive easy ratios cannot be admissible.
5. If the anchor exists, the optimal value is total positive benefit minus its
   harm. Report nonpositive optimal values as no positive headroom, not success.

Unknown outcomes have nonpositive completion-lower gain and no known reference
support, so adding them cannot improve this optimization. They remain in the
cohort and its completion evaluator. Oracle exclusion uses labels and is explicitly
**not an inference rule**. Empty selected risk remains not_estimable, never pass.

This is exact for the present additive intervention-cost contract, not a solution
to future coupled collision constraints or a proof of learnability/generalization.
With a supporting easy anchor, its harm/reference bound also bounds total harm
and full-easy degradation, whose denominators are at least as large. Additional
harmful actions cannot improve utility. Tests compare every Boolean subset on
240 small random cases as an independent optimality check.

## Comparisons and Decision

Solve once over causal eligibility and once as a deletion-only oracle within each
frozen action mask. The latter quantifies whether useful safe choices survive in
the learned selection, not a deployable repair. Measure selected benefit retained,
benefit missed, known harm and missing-label envelope separately. Frozen actions
and ordinary scores must exactly match the previous fitting readout.

If useful feasible choices exist but policies violate fitting constraints, prioritize
an objective/decision-fitting repair. If they do not, report the precise recording
support or forecast limitation; do not drop recordings, relax 2%, or open held
data for tuning. An oracle cannot establish that causal features predict its choices.

## Execution

Native arm64 CPU4/interop1/workers0; local existing packets/checkpoints only.
Per-view resumable receipts, process lock, heartbeat, 10GiB disk reserve, 12h limit.
No optimizer steps, new checkpoints, CREATE submission, deployment, Stage5C or SMC.
Fresh_run labels apply only to this new diagnostic; source data, models and controls
are cached_verified. No metric/seconds, true-3D, foundation or paper-readiness claim.

```sh
.venv-pytorch/bin/python -m scripts.run_m3w_train_feasibility register
.venv-pytorch/bin/python -m scripts.run_m3w_train_feasibility run
.venv-pytorch/bin/python -m scripts.run_m3w_train_feasibility check
```

Append `--resume` only to an unfinished run. Source and protocol hashes are
committed before execution; old negative results are unchanged.
