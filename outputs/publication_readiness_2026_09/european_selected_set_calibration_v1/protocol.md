# Source-Selected-Set Calibration Control

This source-only development experiment is registered before the new calibration
fit. No target-transfer outcome is used by the algorithm. No new forest, neural
model, trajectory predictor, data role, scientific endpoint or risk budget.

## What Changes

The parent component calibrator fits whole-recording residual scores on the
original eligible pool, then changes eligibility. This control refits those SAME
four total residual scores on the CURRENT selected set. It takes the componentwise
maximum of old and refitted margins. Harm upper adjustments can only increase;
reference lower adjustments can only decrease. Benefit scores remain fixed.

Initialize with the parent's empirical90th-percentile raw-pool fit. Refit at
most8 rounds, using the same higher-order statistic and unchanged2% selected
positive-harm/reference budget. Stop at identical consecutive action sets: the
same score fit then cannot increase the total margins again. Missing support,
empty selection, or iteration-cap exhaustion disables this source calibrator.
No epsilon tuning, threshold grid, target cases, inference-time known-label
counts, or fitted-versus-target population statistics are used.

The original raw-pool calibration is the control. Harm-only, reference-only and
joint arms are all retained, with joint predeclared primary. Compare72 identical
source heads, three existing head seeds, the same past-only forecasts/support
and original source-validation partition. Leave one whole source-validation
recording out for BOTH algorithms. The held recording supplies no coefficient,
iteration or stopping information for its own fit. Full-source resubstitution
is a separate diagnostic, not validation. An OOF-screen pass is not evidence
that final full-source coefficients transfer safely.

## Readout and Falsification

Report known benefit/harm/reference mass, selected unknown count and causal
envelope, all/easy selected positive-harm bounds, net utility lower mass,
nonempty/undefined coverage, component counts, convergence, action hashes and
number removed. Complete finite-completion support uses the existing definition.
Primary question: can actual-selected-set calibration repair source-OOF risk
without simply deleting useful support? Do not count empty/undefined views as
passes. Compare both complete pass counts and common-defined risk/utility.
Average dependent views within source locality before3000 paired bootstrap
resamples; nominal development intervals, no independent confirmation.

The new action set must be a subset of its matched parent at every held row.
Every old OOF action hash and old completion screen must reproduce. Rerun all72
new calibrations and require exact output equality. Source-only results cannot
explain or fix persistent transfer harm by themselves. If only resubstitution
improves, the hypothesis is not supported; if all useful actions disappear,
this is abstention, not a successful risk controller. No transfer evaluation
or deployment promotion is authorized by this experiment's result.

## Resources and Evidence

Read verified local source arrays/checkpoints into memory and stream source-only
packets to the owned CREATE directory, bounded to2GiB. No new local numeric cache;
the10GiB cache reserve is unchanged. Small code/registration/log records are not
the numerical cache. CREATE uses a single4CPU/8GiB/1hour scheduled job, isolated
runtime, num_workers0. Save per-head calibrators and heartbeat for resume; preserve
submission intent after ambiguity and never duplicate a job. Remote outputs
are capped at64MiB. Full numerical collection waits for local reserve recovery.

Result provenance: source inputs cached_verified; new calibration/readout
fresh_run only after execution; transfer and independent roles not_run.
Protocol remains obs8/pred12, stride12 raw frames, image-local detector-silver.
No metric, seconds, human-gold, physical-safety, true3D or foundation claim.
Stage5C and SMC remain off.
