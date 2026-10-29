# Whole-Recording Constraint Updates

## Material Passport

New local neural policy experiment on original TRAIN, following a completed
negative direct-policy experiment. Old findings remain immutable. The scientific
question is whether empirical constraint optimization, not another risk-MSE
proxy, prevents retaining useful intervention decisions.

The parent primary violates known all-risk constraints in28 fit views and5 easy
views. Held failures are worse. Prior work already tried loss-scale repair,
moment constraints and soft/hard decisions; none is described as untried here.

## Diagnosis Before Training

At each of18 frozen parent hard_record states, compute exact full-record TRAIN
constraint excesses,256 repeated64-row sample estimates per recording, probability
of omitting every positive contributor, and16 full-versus-sampled gradient
cosines. No optimizer step or held-label scoring. Record every diagnostic,
including evidence against sampling noise. This is posthoc mechanism evidence,
not independent research confirmation or an extra deployment gate.

## Fixed Method Comparison

- stochastic: cached_verified parent hard_record,18 existing fits.
- exact_dual: identical initialization,2,000updates and original64-row recording
  draws for primal gradients; dual updates use exact full-record constraint means.
- exact_gradient: PRIMARY. Same model,initialization and2,000updates; both primal
  gradient and dual updates use exact full-record empirical means.
- original frozen gate and no-intervention floor remain explicit references.

36 new fits,72,000 optimizer updates. The predictive model remains385->32->1
with the original decision as a fixed offset. Inputs,preprocessing,source weights,
record weights,loss targets,2% budget,lr.001,dual lr1/cap100,gradient clip5,temperature
1->.25,hard-forward straight-through derivative and logit>0 decision are unchanged.
Unknown losses retain the parent completion envelope; unknownness is not an input.

Exact full-record calculations use8192-row chunks and one optimizer step after
all chunk gradients accumulate. Stats use float64 accumulation; neural weights
and gradients remain float32. Exact_dual preserves the parent's sampled primal
row stream exactly. Exact_gradient has larger deterministic gradient exposure,
so this is update-matched,NOT compute-matched or example-exposure-matched. Record
full forward/gradient rows and runtime explicitly. No efficiency superiority claim.

No record ID,future position,future knownness,test statistics or new goal input.
No source/record removal. No model capacity increase or new trajectory predictor.

## Readout and Decision

Freeze all36 finals before scoring existing inner-held recordings. No checkpoint
or threshold selection from held results. Evaluate72 fit and72 inner-held views.
The inner-held data were exposed previously and remain exploratory. Frozen
original and stochastic decisions,scores and policy metrics must replay exactly.

Primary exact_gradient is compared with original,stochastic and exact_dual.
Secondary exact_dual is compared with stochastic. Full-policy safety retains all
required scenes and the approved2% all/easy harm/reference plus easy-preservation
requirements. Equal-count per-record/frame comparisons assess utility only; empty
matched risk is not estimable,not a safety pass. Partial-label intervals are not
confidence intervals. No winner substitution or new MSE advancement gate.

Report fit/held constraint violations,useful benefit retained,selected/unknown
counts,empty cohorts,paired utility,source/context/head-seed slices and compute.
Successful empirical fitting alone is not constraint generalization or a paper
contribution. Current inference remains independent-agent selection,not joint
scene optimization. Independent selection/calibration/confirmation stay unopened.

## Relevant Method Boundary

[Cotter et al.,ALT2019](https://proceedings.mlr.press/v98/cotter19a.html) establish
that naive nonconvex constrained learning does not inherit a feasibility guarantee;
their proxy-Lagrangian method and stochastic-classifier results are not claimed
for our deterministic straight-through policy. Their
[ICML2019 constraint-generalization work](https://proceedings.mlr.press/v97/cotter19b.html)
separates fitting and constraint-validation data. This experiment diagnoses
empirical optimization; it does not replace independent calibration. Whole-record
gradient accumulation itself is standard,not a claimed novel method.

## Execution and Claims

Local native arm64 CPU4/interop1/workers0. Pilot20 updates per arm with exact10+10
resume versus direct20,80 pilot updates in total; estimate full cost before launch.
Checkpoint100,heartbeat50,12h execution boundary,resume identical scope.10GiB
reserve,256MiB private checkpoint cap. No scheduler jobs or duplicate runs.

obs8/pred12 rawstride12,pixel/local detector-silver. No metric,seconds,human gold,
physical safety,true3D,foundation or submission-readiness claim. Stage5C/SMC off.
Only code,configs,reports and aggregate metrics go to Git. Private data/models stay local.

Pre-execution engineering revision: a checkpoint can stop between scheduled
heartbeats. Trace entries now follow the configured update schedule rather than
the requested stop point, preserving exact resume including traces. An unexecuted
registration draft is retained; the binding registration follows the regression
test. No real-data diagnosis or training preceded this fix.
