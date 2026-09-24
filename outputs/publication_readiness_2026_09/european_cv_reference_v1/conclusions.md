# CV-Reference Repair: Completed, Not Deployable

## Question and What Changed

The preceding fallback was selected for average forecasting accuracy but itself
degraded positive-easy examples by 15.48% relative to constant velocity (CV).
Learning harm relative to that fallback did not answer the CV-relative safety
question. This version changes the decision reference to causal CV and rebuilds
the necessary rollout-disagreement features, gain/harm labels and fitting-only
cost scales. It does not retrain or alter the 18 neural forecasts.

`fresh_run`: nine ridge fits and nine real Torch cost heads, three seeds
(17, 29, 43), three nested source folds, 2,000 updates per neural head. All
18 endpoints complete before the comparative readout. The 100-update pilot is
included in the 18,000 updates, not an extra budget. No threshold, seed or
checkpoint is selected from these results. This is a source-development
reference-contract experiment, not independent confirmation or new dynamics.

`cached_verified`: predictors, histories, splits, previous policy outputs and
source identities. For each excluded source fold, the cost-label teacher chain
also excludes that fold. No target or future-validity mask enters decision inputs.

## Full-Cohort Findings

All 318,969 registered targets remain present. ADE has 311,922 supported targets
and 7,047 unknown targets; requested-final-step FDE has 240,269 supported targets
and 78,700 unknown. Missing futures are reported, not a retrospective eligibility
filter. The task is 8 observed / 12 requested steps at raw stride 12, in image
pixels. It is separate from the historical raw-frame t+50 task.

| Neural cost seed | ADE gain vs CV | 95% locality interval | FDE gain vs CV | Hard ADE gain vs CV | Easy degradation | Worst easy-locality degradation | Zero-CV harmed |
|---|---:|---|---:|---:|---:|---:|---:|
| 17 | 4.3015% | [2.4717%, 6.1509%] | 6.3022% | 6.5532% | 1.9050% | 8.7108% | 1 / 4 |
| 29 | 4.1848% | [2.1895%, 6.2669%] | 6.1406% | 6.6674% | 2.3297% | 12.5388% | 1 / 4 |
| 43 | 4.4308% | [2.3761%, 6.4805%] | 6.3118% | 6.7147% | 2.3757% | 11.2869% | 1 / 4 |

The corresponding old neural cost policies degrade easy by 13.3839%, 12.8069%
and 13.2583%. Correcting the reference materially reduces the observed harm,
but does not satisfy the unchanged worst-locality or exact-zero requirements.
Ridge produces larger CV-relative ADE gains (5.20--5.67%) but larger easy
degradation (9.50--9.77%); it is not a safe winner either.

The primary comparison cannot stop at CV. Neural pointwise gains over the fixed
damping-0.97 control are only 0.1980%, 0.0903% and 0.3568%, with intervals
[-1.7976%, 2.4436%], [-2.0006%, 2.3538%] and [-1.6217%, 2.4369%]. Stable
superiority to that strong control is not established. Direct gains over the
old pointwise policies are 1.5330%, 1.3912% and 1.4289%; those three intervals
also include zero. All intervals use 3,000 paired locality resamples conditional
on these fitted models and opened development sites, not window-iid resampling.

Metrics average locality-specific percentage gains with equal locality weight.
Subtracting two such percentages does not give the direct model-to-model gain;
read the paired contrast itself. Three seeds are reported, not averaged model
predictions or a post-readout selected seed.

The registered complete-future sensitivity has 193,705 targets and neural
CV-relative ADE gains of 5.2001%, 5.0129% and 5.2656%. This sensitivity does not
replace the partial-label population or excuse any safety failure. Per-locality
errors, p95/p99 model errors and masks remain in [analysis.json](analysis.json).

## Joint Decision Result

The fixed joint pilot contains 1,152 queries and 6,116 targets. Every query's
independent, exact-count unary and exact-count joint decisions is checked for
the same actual intervention count, not merely the same nominal budget.

Neural joint ADE gains versus CV are 1.0567%, 0.9323% and 0.5936%, but gains
versus the training-selected baseline are negative: -0.1170%, -0.2451% and
-0.5848%. Mean easy degradation is 1.0958%, 0.7591% and 0.1743%; the worst
locality degrades by 5.1690%, 6.0124% and 1.6473% respectively.

The pilot contains **no zero-CV cases**. An observed zero-harm count of zero
therefore does not validate zero-event protection, even where an internal
observed-support flag is true. Seed 17's exact-count joint advantage over
independent decisions is only 0.0114% [0.0000%, 0.0323%]. The other two seed
contrasts are undefined because a declared locality lacks supported nonzero
matches. Undefined values are retained; those localities are not dropped.
The interaction term has no stable predictive contribution in this study.

## Support Finding and Remaining Gap

All four zero-CV cases are in locality 008. Their excluded outer fold has no
zero-CV fitting cases. Each history previously moves but ends with zero causal
velocity. Each has only two observed future labels out of twelve, with no
requested-final endpoint. Zero masked ADE means zero error on that observed
prefix, not verified correctness over all twelve future steps.

These cases remain in the evaluation and are not used to fit a new test-time
filter. The posthoc finding identifies a relevant-event support gap and label
coverage limitation; it neither proves prediction impossible nor grants a
safety exception. See [failure_analysis.md](failure_analysis.md).

## Verification and Decision

- Complete metric reconstruction: `cached_verified`, same analysis SHA256.
- New checkpoint inference: `fresh_run`, 18 heads x 4,096 held rows, exact.
- Separate accounting: original neural and previous-policy locality errors
  unchanged; all 18 decision receipts pass actual matched-count checks.
- 145 relevant tests pass. Not the entire historical repository test suite.
- No training/runtime process remains active for this experiment.
- Independent model selection, risk calibration, confirmation, DroneCrowd,
  deployment and formal submission: `not_run` / unchanged.

**Decision: partial reference repair; do not promote any head or joint policy.**
The next version should test event-conditional risk with source-only support
checks, maintaining the same safety limits and independent roles. A predicted
unconditional harm budget is not an actual easy-case guarantee. More model
capacity or additional joint geometry is not yet the evidence-based priority.

No seconds, meters, verified online-sensor causality, human-gold annotation,
physical safety, true-3D, foundation-model or submission-ready claim. Stage5C
and SMC remain disabled. Historical Stage37 scores remain exploratory under the
later duplicate/teacher/test-selection audit and are not restored as clean
deployment evidence by this experiment.

Evidence: [registration](registration.md), [matrix](matrix.json),
[all controls](results.md), [losses](training_losses.md),
[metric replay](verification.json), [checkpoint replay](checkpoint_replay.json),
[accounting](accounting_audit.json), [figure](repair_contrasts.svg),
[execution](execution_notes.md), [Chinese operation](operation_zh.md).
