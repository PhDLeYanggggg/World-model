# Direct Tail Risk: Bounded Mechanism Experiment

The original evaluation rules remain binding. No identity-fallback certificate,
new denominator, relaxed 2% threshold, deleted empty control, or retrospective
pass is introduced. All original failures remain failures. This is exposed
SOURCE model development, not an independent test or a replacement primary.

## Hypothesis And Distinction From Earlier Failures

The completed proposal-conditioned critic overpredicts aggregate harm on all
proposals yet underpredicts it severely among its own admitted actions. Merely
fixing vanishing harm gradients, adding record/subset excess penalties, or
conditioning the loss on a foreign router has not repaired this selection.
Earlier signed-excess losses still formed differences of positive moment
outputs. This experiment directly emits two unconstrained signed outputs:
H - .02 R and EH - .02 ER. They are not separately identifiable cost moments.

Two new, identical 28,034-parameter context networks compare conditional-mean
squared error with fixed .9 quantile pinball loss. Only their objective differs.
The quantile is a modeling choice, NOT a finite-sample/conformal or 2% safety
guarantee. A negative predicted quantile does not guarantee an average-risk
constraint under source shift or adaptive selection. We test that empirically.

## Smallest Decisive Comparison

Choose the first already registered context in each of the three critic folds,
by registry order, before fitting. For each, frozen router A=1 proposes actions;
new critic B=0 learns on its own sources; original V=2 is read only after every
new fit is frozen. Six real heads receive 2,000 updates each. This reduced
context roster is explicitly a pilot, not the previous 108-head/full study.
It is chosen by fixed indices, not favorable prior outcomes. No new H, reserved
Moving group, DUT, or DroneCrowd data is opened. No new forecast is trained.

Reuse exact own-B preprocessing, all causal rows, query sampler, proposal mask,
and unknown-outcome completion contributions. Unknown H/EH remain causal
envelope bounds and R/ER remain zero; they are not observed labels. Normalize
each signed target using sqrt(rms_harm^2 + (.02*rms_reference)^2) from cached B.
Mean and quantile share initialization, sampler, budget, source roster and
inputs. The 48 shared-motion columns stay zero as in the parent control.

Keep the foreign public router's proposal and utility fixed. Admit an action
only when both new signed predictions are <=0 and the frozen point critic's
predicted R and ER are positive. Otherwise use the original floor. The observed
full-strategy, recording-level harm/support/easy-net rules are unchanged.
Keep same-query same-count/same-amplitude utility controls; empty utility is NA.

## Readout And Expansion Decision

Recompute old all-action and proposal-conditioned controls exactly. Report new
mean/quantile policies, original known and completion utility, intervention,
positive harm, easy net degradation, unsupported recording views, and signed
prediction versus realized excess on all proposals and admitted actions. Report
fit-B and held-V diagnostics separately. No V-derived threshold or refit.

This pilot does not establish population significance. Show all three context
results and all source/recording views; descriptive source-macro summaries are
not independent-test claims. Request a separate full registration only if the
quantile arm has positive completion-lower paired utility against the mean arm
in every pilot context, no more known all/easy violations in each, and at least
one actual intervention per context. Otherwise do not scale this hypothesis
merely because one slice is attractive. This is a compute-allocation rule, not
a changed scientific pass gate. Deployment is unchanged regardless of pilot.

Native arm64 CPU4/interop1/workers0. Checkpoint every100 updates, heartbeat100,
atomic resume, exact interrupted-vs-direct smoke test and first full refit.
Freeze all six before V; replay predictions/decisions/readouts afterwards.
No threshold sweep, secondary winner promotion, or test-set optimization.

obs8/pred12 native annotation order, local-image pixel, non-human-gold labels;
not SDD rawt50, metric, seconds, physical-safety, true3D or foundation evidence.
Stage5C/SMC off. Independent confirmation and submission readiness remain false.
