# Completion-Aware Cost Supervision: Fixed Paired Training

## Question And Scope

Can matching the supervised cost target to missing-outcome accounting improve
safe selection, without changing the forecast bank, inputs or safety budget?
Use six original SOURCE contexts, three controller seeds, two banks
(physical-only and physical+neural) and two supervision modes: 72 fresh
26,309-parameter Torch cost heads, 2,000 updates each / 144,000 primary updates.
The trajectory producer seed 43 is frozen; three controller seeds are not
three independently trained forecasting pipelines.

This first experiment is fitting-only. No inner-held, independent selection,
calibration, confirmation or DroneCrowd readout. All required fitting sources
and recordings remain. Fitting success would motivate a separately specified
next experiment, not deployment or automatic access to held roles.

## Exact Target Difference

Keep the observed five-cost labels unchanged, including original NaN rows.
For known outcomes both modes use the same B,H,R,ER,EH values. For a wholly
unknown future outcome, use the causal canonical max-disagreement envelope E
to derive a distinct training proxy:

```text
benefit_lower = 0
harm_upper = E
reference_lower = 0
easy_reference_lower = 0
easy_harm_upper = E
```

These are finite interval endpoints, not observed or imputed errors. Actual
future reference can be unbounded; zero is only a lower bound. Positive harm
and easy harm are bounded by E for any nonempty future mask. No missing-at-
random assumption or inferred future event is used. Known partial-label
costs remain the original native masked ADE, not full-trajectory ground truth.

Both modes draw the same source-balanced past-eligible query stream,
including unknown-outcome queries. The known-only control masks their loss;
the completion mode supervises the explicit endpoint proxies. Counts of
unknown rows drawn and included in the objective must be measured, not a
hardcoded zero. Support masks define training supervision only and are never
features, causal eligibility rules, inference arguments or action filters.

The paired control is freshly trained because this common causal sampler
differs from the earlier known-label-only sampling measure. Do not call
cached old heads a matched control. Initial weights, optimizer, row draws,
steps, architecture, source weighting, batch size, original physical-fitting
feature/target normalizers, asymmetric cost loss and ranking weight are
identical within each pair. Normalization is not refitted to proxy endpoints.
The addition of unknown-row loss is the intervention, not matched loss size
or matched gradient norms. No extra clipping/alpha/threshold search.

## Engineering And Training Freeze

Before primary training, run all four bank/mode cells on the complete first
context for 20 updates, with 10+10/direct-20 exact recovery. Compare the
known-only method to the existing scene-head training core on that same new
common stream with unknown losses masked. This is an engineering neutral
control, not reproduction of the old differently sampled endpoint. Record
actual gradients, nonzero parameter updates, unknown supervision and runtime.

Estimate full cost and checkpoint space from that real pilot. Native arm64
CPU4, interop1, workers0. No Torch device/resource probing or multiprocessing.
Keep a 10 GiB disk reserve, <=512 MiB private checkpoints, process lock,
atomic checkpoints every 100 primary steps, heartbeat every 50 steps and a
12-hour boundary with explicit resume. Reduce neither the target roster nor
the scientific objective for slowness. Use CREATE only if the pilot actually
justifies it; reconcile scheduler/environment before any new submission.

Refit the first full 2,000-update primary endpoint in a separate checkpoint
and require exact scientific state, including Adam, sampler, RNG and trace.
Complete and commit all 72 endpoint receipts and their freeze before the
first fitting readout. No loss-based checkpoint/seed/target-mode selection.

## Readout And Gates

Evaluate both fresh controls, cached original static-scene policies, original
binary policy and the floor on every original fitting recording. The cached
neural-only static head is a secondary reference, not newly trained here.
Compare completion minus known-only separately in each bank, using actual
paired forecast paths and within-recording/frame count-matched utility.
Also report the mixed-minus-physical contrast within each new mode. Count
matching measures utility only, never safety.

Score against the untouched observed labels and canonical completion bounds,
not proxy training targets. Keep all unknown outcomes, undefined percentages
and empty selected-risk estimates. The all/easy positive-harm/reference and
easy-degradation upper bounds must each satisfy the original 2% requirement
on every required recording. Empty risk is not_estimable, not safe zero.

Report known ADE/FDE, paired completion-lower utility, hard/easy breakdown,
regret, selected unknown counts, full recording risk support, observed-known
moment diagnostics and separate proxy-target fit diagnostics. A proxy loss
or lower MSE is not a conditional-risk certificate. Preserve an unestimable
required FDE or hard view rather than dropping it.

Use all three head seeds and 3,000 locality resamples after context/head-seed
averaging. They are fitting-resubstitution intervals, not independent
generalization, whole-pipeline OOF or repeated-research-selection correction.
The fitting advancement gate requires complete mixed-policy risk support,
positive full and matched completion-minus-control utility intervals and a
positive mixed-minus-physical completion-lower interval. Report failures
without a new threshold sweep. Separate exact inference replay is required.

## Scientific Limits

EuropeanSquares SOURCE is oblique detector-track midpoint pixels, eight
observed and twelve predicted steps, stride twelve raw frames. Static
polygons are region proxies, not current scene-image encoding or verified
walkability/metric maps. No future endpoint, central velocity, future support,
test statistics or test-endpoint goal enters inference. These are cost heads,
not new trajectory dynamics or foundation models. No metric/seconds, human-
gold or true-3D claim. No Stage5C execution or SMC. Deployment stays unchanged.
