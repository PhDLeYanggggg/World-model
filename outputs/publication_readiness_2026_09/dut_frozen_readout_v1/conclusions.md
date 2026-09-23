# DUT Frozen-Chain Results and Limits

## What Was Run

The registered full readout completed, not a pilot: 27 recordings, two physical
sites, 9,147 query times and 420,364 past-eligible target windows. The inference
loop took 13,717.031 seconds (3.81 hours), excluding initialization and binding
checks, using the fixed arm64 CPU path. No interruption,
sample reduction, model refit, external threshold search or seed selection was
used. The independent original-CSV recount matches every model's complete receipt.

Of those targets, 399,729 have complete future paths, 18,898 have partial labels
and 1,737 have none. All remain in inference. The main descriptive metric is
complete-path native ADE with equal weight per physical site; it is not the
older SDD available-point ADE. Future labels are used only after decisions.
Existing source checkpoints are hash-verified, not freshly trained here.

Aggregate replay is exact. Fresh-process model/decision replay completed on
2026-09-24 local time: all 66 fixed chunks / 6,715 query times across 27 recordings
and twelve views reproduce exactly. The replay loop took 11,219.410 seconds,
excluding initialization and binding checks. This is not a second full readout
or an independent researcher verification. Both processes exited successfully.
The raw-CSV check independently implements population counting, not prediction.
The 234 scoped tests are not a claim that the entire legacy suite was run.

## Fixed Three-Seed Results

Changes below are relative to the frozen causal-CV floor, not the strongest
observed average-error DUT baseline. Values are seed mean and sample standard
deviation in percentage points. All fixed families and heads are retained.

| Predictor / cost head | Strict protected gain | Uncontrolled gain | Worst site/seed easy degradation, strict / uncontrolled |
|---|---:|---:|---:|
| Transformer / neural | 1.591 +/- 0.057% | 17.195 +/- 2.760% | 0.000 / 78.608% |
| Transformer / matched forest | 1.086 +/- 0.204% | 17.195 +/- 2.760% | 0.000 / 78.608% |
| EqMotion / neural | 1.708 +/- 0.739% | 41.447 +/- 1.166% | 0.000 / 82.018% |
| EqMotion / matched forest | 0.720 +/- 0.068% | 41.447 +/- 1.166% | 0.000 / 82.018% |

This supports a limited descriptive benefit/harm tradeoff, not a new deployed
model. Neural-head mean strict gains exceed their matched forest controls here,
but this is not independent confirmation of neural superiority or a calibrated
risk guarantee. The six forecasting endpoints are represented twice across cost
head views; these are not twelve independently trained forecasters.

The fixed damped_velocity_005 control has lower overall ADE (3.6348) than CV
(3.8434), a 5.428% average improvement. It also incurs 504.128% worst-site easy
degradation and worsens vehicle ADE. It fails the same easy-preservation condition;
it cannot be called a safe replacement just because its average is better.
Conversely, protected neural gains must not be advertised as beating this
strongest observed average-error control. No DUT winner is selected for deployment.

## Why This Is Not Yet the Main Contribution

1. **Joint selection has negligible incremental effect.** At half intervention
   count, joint differs from unary geometry on only 21 of 109,764 query/view
   instances. The changes in overall improvement are below 0.00025 percentage
   points in magnitude and have mixed signs. At full count all eligible candidates
   are required, leaving no identity-selection freedom; coincidence there is
   structural, not an independent demonstration that interactions never matter.
   The half-count comparison does not establish a useful joint contribution.
2. **Easy support is narrow.** Source-frozen easy cuts identify only 922 complete
   targets, about 0.231% of the complete population, across two sites. Zero observed
   easy degradation is not broad robustness or a 2% population-risk certificate.
   Thresholds must not be redefined using DUT to make support or results look better.
3. **Protection discards most unconstrained gain.** The strict rule intervenes
   sparsely. Uncontrolled neural forecasting improves averages substantially but
   fails easy preservation. These measurements motivate source-only development
   of better gain/harm decisions, not loosening the rule against DUT labels.
4. **Transport remains mismatched.** The SDD models use stride12 while DUT uses
   stride1; matching 8/12 step counts does not match physical duration. The source
   cost producers excluded one source site, whereas the final forecaster fits all
   four sites. The current result cannot isolate the contribution of either shift.

## Missing Labels and Uncertainty

The conservative full-population CV-minus-policy ADE-difference intervals remain
positive for all twelve strict views, including unknown future steps bounded by
prediction disagreement. These deterministic missing-label bounds do not identify
unobserved absolute ADE, and they are not confidence intervals or safety bounds.

The paired 3,000-resample bootstrap uses just two physical sites. Its sometimes
very narrow intervals reflect similar observed site means and coarse resampling
support, not high precision for a broad unseen-world population. Three seeds do
not create additional independent sites. Tail errors and physical-validity scores
are not reconstructed from aggregate means; they remain missing evidence.

## Next Research Action

Keep this external table fixed, with no DUT-guided threshold, seed or family
selection. Use the development sources to test whether joint decisions have a
nonadditive opportunity at useful intervention counts under the existing easy
constraint. A follow-up should compare the frozen neural policy with a similarly
protected simple-motion policy under matched budgets, not merely an unprotected
damped predictor. Any policy change needs a new source-only registration.

Do not open DroneCrowd just to accumulate a second favourable average. Its
separate whole-source confirmation role remains intact. A one-shot contract must
state the full chosen-in-advance comparison, observation/time mismatch, grouping,
and limits of two-site risk calibration; no prediction-based regrouping is allowed.

This is a dataset-local annotated-history 2.5D trajectory study, not true 3D,
metric or seconds-level prediction, foundation-model success or a safety proof.
It does not demonstrate a new scene-image, goal or multimodal ablation contribution.
Historical selection-exposed Stage37/43/44 results do not become independent
evidence because this new run completed. Stage5C and SMC remain disabled.
