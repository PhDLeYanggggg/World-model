# Geometric Cost Heads: More Intervention, No Safe Neural Advantage

## Conclusion

The registered experiment is complete. I trained 54 real Torch gain/risk heads
for 108,000 optimizer updates, retained all 144 policy views, and reproduced the
checkpoint scores and complete metrics. Constraining predicted gain and harm
by causal rollout distance changes the controller, but does not solve its main
failure: underestimating harm on the samples it chooses to change.

Six new neural-versus-matched-damping contrasts have a strictly positive
conditional ADE interval. **All six fail observed safety.** The old controller
has one such positive interval, which also fails safety. No arm is selected for
deployment, and this is not a successful neural world-dynamics contribution.

## What Changed

All forecasts, 355 causal features, training-only scalers, locality samplers,
training budgets and 2% predicted-risk limits were held fixed. Each new head
has 22,914 parameters, matching its old counterpart. Only the cost output
parameterization changed: predicted gain/harm is bounded by the maximum
candidate-versus-CV rollout distance over the requested future steps. This
distance uses two causal predictions, not future observations or their mask.

The maximum, rather than mean, is needed because future labels can be partial.
The triangle inequality gives a valid magnitude bound for any observed subset;
it does not guarantee a correct expected cost or a transported risk budget.
Reference error mass remains unbounded and can be positive when the two
forecasts coincide. No trajectory producer, JEPA or joint solver was trained.

## Complete Arm Summary

Each arm has 18 neural and 18 damping views: three fitting folds, three seeds,
and two event definitions. A safety pass here requires worst-locality easy
degradation at most 2% and no observed harm on zero-CV-error rows.

| Head arm | Neural observed safety | Damping observed safety | Positive neural ADE points vs damping | Strictly positive CI | Strictly negative CI |
|---|---:|---:|---:|---:|---:|
| Old | 9/18 | 16/18 | 2/18 | 1/18 | 15/18 |
| Utility only | 11/18 | 16/18 | 0/18 | 0/18 | 17/18 |
| Risk only | 8/18 | 18/18 | 7/18 | 4/18 | 11/18 |
| Both | 7/18 | 18/18 | 4/18 | 2/18 | 14/18 |

Utility-only changes do not establish an advantage over damping. Risk changes
allow more useful neural interventions, but also more harm. Damping's new-risk
views pass all 18 observed checks; that is a development result, not independent
safety certification or grounds for choosing a new deployed policy.

The largest positive neural ADE contrast is +1.7979% against matched damping,
with conditional 95% CI [+0.8246%, +2.8320%]. Its worst-locality easy degradation
is **17.6432%**, so it is not an acceptable winner. Across risk-only views the
worst easy degradation is 17.8014%. Other positive contrasts preserve average
easy accuracy but harm 2-4 zero-CV-error rows; those failures are also retained.

## Why the Repair Is Insufficient

An informative, post-hoc failure slice is fitting fold 2, seed 17, easy-event,
locality 110. The both-head policy increases overall-view intervention from
1.2889% to 37.5189%. On selected rows in locality 110, predicted positive-event
harm/reference mass is about 0.69%, while its observed ratio is about 47.09%.
This ratio is not the net easy-degradation statistic: easy degradation in that
locality is 17.6432%. Bounded magnitudes do not prevent badly underestimated
conditional harm. See [failure analysis](failure_analysis.md) and the complete
[reliability table](score_reliability.csv), not this one slice alone.

The preceding frozen-head scaling diagnostic found no sampled amplification of
training-constant features beyond one standardized unit. It did not justify
silently changing the old scaler. The preceding smaller-producer replacement
also failed to provide a consistent repair. These results narrow the next
experiment toward event support and selection-conditional risk estimation;
they do not prove a unique root cause.

## Evidence and Limits

- `fresh_run`: scaling diagnosis; 54 head fits; all-view inference/accounting;
  54 checkpoint replays; 54 exact sampler comparisons; complete metric replay.
- `cached_verified`: 18 inner OOF forecasters, nine final forecasters, 54 old
  heads, source geometry and role/producer lineage. Forecasts are unchanged.
- `not_run`: independent confirmation, new risk calibration, joint selection,
  remote training and deployment promotion. CREATE's specific M3W asset
  directory is still unverified; a dated queue observation is not a live check.
- 210 tests across 32 scoped files pass. The full legacy suite was not run.
- Every fit excludes eight localities from its complete producer chain. All
  twelve European Squares localities are nevertheless opened development data.
  The 144 views share localities; they are not independent replications.
- Three seeds and 3,000 locality-bootstrap draws are retained. Intervals are
  conditional, with no multiplicity adjustment or outcome-selected winner.
- Released detector tracks, image pixels, obs8/pred12, raw stride12. Not legacy
  t+50, seconds, metric, human gold, physical safety, true 3D or foundation.

The next controlled hypothesis is that factoring event support and conditional
harm, then evaluating selection-aware risk on source-excluded data, can improve
the gain/safety tradeoff. It must retain matched damping, old heads, zero-error
support checks and fixed scientific roles. This is a next experiment, not a
successful result. No threshold is changed using the present readout.

Deployment remains unchanged. Historical Stage35/37 scores are exploratory,
not recertified here. Stage5C and SMC remain disabled. The project is **not yet
a CCF-A submission candidate**.

[All results](results.md) | [All-comparison figure](head_ablation.svg) |
[Training losses](training_loss.svg) | [Gates](gates.md) |
[Execution](execution_notes.md) | [Chinese operation guide](operation_zh.md)
