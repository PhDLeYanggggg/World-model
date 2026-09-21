# Native Gain/Harm Heads: Useful Costs, Unsafe Intervention

Completed 2026-09-21. `fresh_run`: 12 ridge and 24 real Torch neural cost-head
fits, held-source inference, checkpoint replay and independent arithmetic.
`cached_verified`: frozen native forecasters and nested source training views.
`not_run`: independent calibration/confirmation, new external transfer, deployment.

## Research Decision

The useful route remains a focused test of **when a neural motion forecast is
worth replacing a causal baseline**, rather than a larger generative model.
The native forecaster already provides a positive controlled development result;
this experiment tests whether its benefit can be retained without easy-case harm.
It does not yet solve that problem or establish a publishable new method.

The task is eight observed annotation steps to twelve predicted steps. Errors
are SDD annotation pixels, summarized as the equal mean of within-scene ADE
improvements against causal constant velocity (CV). These are not t50 scores,
meters or seconds. All four source scenes have already informed research design.
Fitting exclusion is clean, but the readout is not independent confirmation.

## What Was Actually Fitted

- Four outer-held source scenes, three seeds, twelve physically separate cost
  training views. Upstream forecasts exclude both the row scene and the head's
  held scene. Normalizers, constant controls and quantile cuts use training only.
- 355 causal features: history/neighbor/rollout summaries, both complete causal
  candidate rollouts, and the observed native scale. No future label, future
  validity mask, scene ID or oracle decision is an inference feature.
- Twelve ridge fits, twelve width-64 neural MSE fits, twelve matched neural
  underharm4 fits. Underharm4 multiplies squared harm-underestimation loss by four.
  Its output is an upper-expectile score, not an expected mean, calibrated failure
  probability or statistical upper confidence bound.
- All neural heads complete 3,000 updates: 72,000 total, 18,432,000 sampled rows,
  22,914 parameters per head. Paired losses use identical initial states and
  sampled batches. Unknown labels are never sampled or filled with zero.
- All 36 endpoints complete before scoring. No checkpoint, seed or threshold
  is selected on the held source results. Both fixed rules are retained.

## Results

`Positive` switches if predicted benefit exceeds predicted harm. `Strict` also
requires predicted harm <= 0.1 * predicted benefit. Both exclude an exactly
unchanged candidate rollout. Neither rule is independently calibrated.

| Head / Fixed Rule | ADE Gain vs CV | Scene CI 95% | Hard Gain | Positive-Easy Degradation | Switch Rate | Zero-CV Safe Seeds |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Constant / positive | 7.633% | [5.963, 9.302] | 10.658% | 21.710% | 88.413% | 0/3 |
| Constant / strict | 0.000% | [0, 0] | 0.000% | 0.000% | 0.000% | 3/3 |
| Ridge raw / positive | 7.546% | [6.026, 9.256] | 10.249% | 15.875% | 70.679% | 0/3 |
| Ridge raw / strict | 0.133% | [0.013, 0.350] | 0.011% | -0.317% | 1.320% | 0/3 |
| Ridge clipped / positive | 7.539% | [6.020, 9.254] | 10.248% | 15.890% | 70.585% | 0/3 |
| Ridge clipped / strict | 0.126% | [0.012, 0.332] | 0.011% | -0.302% | 1.226% | 0/3 |
| Neural MSE / positive | 5.574% | [3.398, 7.807] | 7.762% | 7.111% | 30.246% | 0/3 |
| Neural MSE / strict | 1.292% | [0.511, 2.301] | 1.740% | 0.586% | 2.793% | 0/3 |
| Underharm4 / positive | 4.452% | [2.526, 6.028] | 6.235% | 3.833% | 15.811% | 0/3 |
| Underharm4 / strict | 0.340% | [0.019, 0.868] | 0.401% | 0.102% | 0.842% | 2/3 |

Hard is the training-CV q75 diagnostic; positive-easy is training positive-CV
q25, excluding exact-zero cases. These diagnostic cuts do not redefine or certify
the formal easy gate. Negative degradation denotes improvement. Three seed
errors are averaged before the equal-scene primary reduction. The 3,000-draw
bootstrap resamples four physical scenes, not overlapping rows. These conditional
intervals do not provide independent confirmation or a risk guarantee.

**No nontrivial rule satisfies strict zero-reference protection across all seeds.**
The passing constant/strict control never intervenes and has zero gain. It is
fallback, not learned success. Underharm4/strict still harms one zero-CV query
in deathCircle/seed43 by 0.45034 pixels ADE. Seeds17/29 have no observed zero-CV
harm, but choosing those seeds after readout would be inappropriate. There are
48 selected unknown-outcome query/seed instances under that rule; missing error
is not zero. See [full fixed table](results.csv) and [failure slices](failure_slices.json).

## What Changed and What Did Not

Mean harm MSE over the twelve outer-scene/seed groups is 8.7209 for the constant,
8.2393 for raw ridge, 8.2212 for clipped ridge, 8.1130 for neural MSE and 11.8983
for underharm4. Better average MSE does not make eligible interventions safe.
All twelve MSE strict-rule groups underpredict mean realized harm. Underharm4
reduces that count to six but still fails strict protection. Its larger MSE is
not directly comparable as a mean-estimation objective because it fits an expectile.

The asymmetric penalty reduces harmful interventions, but it also sharply
reduces intervention count and gain. **A matched-coverage control has not yet
shown that its ranking is better rather than simply more conservative.** No
causal claim about improved conditional judgment is justified by these two
unequal-coverage rules alone. Raw ridge has 10,330 negative harm predictions
across repeated outer/seed rows; both raw and clipped readouts remain visible.

## Gate Status and Next Experiment

| Check | Status |
| --- | --- |
| Real training, frozen budgets, clean nested fitting, exact score replay | Pass |
| Positive source-only accuracy for fixed intervention rules | Observed, developmental |
| Strict zero-CV protection with nonzero gain across all seeds | Fail |
| Independent risk calibration and untouched confirmation | Not run |
| New deployable policy or publication-ready method | Not established |

The next falsifiable comparison is **equal intervention coverage**, keeping the
same frozen forecaster and train-only cost views. Test whether underharm4 ranks
harm better than MSE at the same per-scene count, with a causal/random control.
Only then decide whether to change the conditional-risk objective. A new
easy-sensitive target should use easy labels only in training/evaluation, never
the future-defined easy status at inference. Independent calibration still needs
separate scene support. Do not fix the observed deathCircle row with an ad-hoc
rule, relax zero-reference tolerance or search this readout for a lucky seed.

Old Stage26/37 results remain exploratory after earlier exposure audits. This
experiment neither re-certifies those policies nor promotes a replacement.
M3W remains a 2.5D trajectory/world-state research track, not true 3D, metric,
seconds-level or a foundation model. Stage5C execution and SMC remain disabled.

## Verification

All 336 frozen bindings verify. All 36 endpoints and twelve train-only
preprocessors replay; 2,636,340 repeated cost-score rows match exactly. A separate
scalar-distance implementation checks 527,268 forecast rows, 720 scene metric
reductions and 120 fixed policy/scene slices. All checks pass. The 67 scoped tests
pass; this is not a claim that the entire historical test suite was rerun.
See [execution and provenance](execution_notes.md) and
[independent verification](independent_verification.json).
