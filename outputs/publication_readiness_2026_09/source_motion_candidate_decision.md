# Fixed Training-Side Zero-Target Gradient Control

## Material Passport

Registered before fitting any new candidate. Prior experiment: source_crossfit_v1,
commit 80114832. Twelve cold-start candidates lost to stationary CV on each held
site/seed; equal-site gain -5.01598%, fixed candidate/CV oracle +0.46765%.
False movement accounts for 4.32112pp of primary excess, but the nonzero-target
subset also loses. This follow-up is an optimization/candidate diagnostic,
not a new main protocol, split, independent test or proposed contribution.

## Hypothesis and Single Intervention

Hypothesis: zero-target ADE gradients shrink useful conditional motion forecasts,
leaving a near-zero candidate with too little gain for learned deferral. Compare
the existing unconditional objective with the same batch mean after multiplying
each zero-target row's loss by zero. Keep the full-batch denominator; do not
resample positives or increase their individual gradient weight. This tests
removing one class of gradient, not an isolated proof about conditional medians.

All train IDs, uniform sampling, initial seeds, model architecture, parameter
count, normalization, loss scale, step count, optimizer, clipping and rate
schedule are matched. No old weights are reused. The new terminal sampler RNG
and per-row draw counts must equal the cached control. The replacement engine
in unconditional mode must exactly reproduce the old two-phase engine on a
synthetic regression test before any real fitting.

Train-label conditioning is allowed only inside the loss. All inner-held and
training rows are forecast and scored, including zero targets. No future label
filters inference, selects a checkpoint, or defines a deployed guard. The
conditional candidate may invent large easy-case motion; this must be reported.

## Matrix and Resources

Four inner sites coupa/deathCircle/gates/hyang, seeds17/29/43: twelve new fits.
Bookstore excluded from every producer, fit and forecast. Main and sealed roles
unchanged and unscored. Use all15,430 stationary-history source queries for OOF
coverage;29recordings/545scopedagents, not independent windows. Eight observed
steps, twelve targets, stride12raw annotationframes, unchanged observation contract.

10,000 updates per model:2,000 constant plus8,000 cosine, 120,000newupdates total.
Same .0003rate, .0001weightdecay, batch64, clip5; random initialization only.
CPU4/inter-op1/workers0 with nativearm64 .venv-pytorch. Existing measured
120kbudget takes about96minutes and fits below3GiB; local66GiBfree observed.
An included100-update pilot will check the new engine. Slow training is not a
reason to truncate. Atomic checkpoints and heartbeats every200steps; exact resume.

Local/GitHub currentcommit80114832verified. CurrentCREATEjobs/assets remain
unverified; historicalSSHauthentication failure is not a fresh scheduler check.
This local-sized control does not require remote submission or duplicate jobs.

## Fixed Analysis and Decision

Always report original equal-site normalized ADE and all-site/seed errors,
zero-target absolute harm, nonzero-target error, per-producer training-defined
hard slices, nativepixelADE, and fixed CV/candidate binary future-oracle gain.
Report paired motion-minus-control differences in actual and oracle gain.
Use three seeds, 2,000 shared site/recording bootstrap draws, seed38113.
Intervals are conditional on four explored sites and overlapping fits.
Seed means are means of errors, not ensembles. No trial or threshold selection.

A positive oracle contrast would motivate causal benefit/risk learning; it is
not itself a deployable gain. If candidate direction/path utility remains weak,
do not train another cost head on the same low-headroom actions. If raw outputs
harm easy rows, do not deploy them. Even a useful oracle requires downstream
selection with nested producer exclusions and independent risk evidence.

## Boundaries

Control models/data are cached_verified; new training and analysis are fresh_run.
Risk-head fitting, joint intervention, main/external confirmation and deployment
are not_run. No physical time or metric scale claim; the historical annotations
are offline and may be interpolated, not certified sensor-as-of inputs.
This is not a new JEPA/Transformer, foundation, true-3D or human-gold result.
Stage5C execution and SMC remain off. Preserve negative results and prior files.
