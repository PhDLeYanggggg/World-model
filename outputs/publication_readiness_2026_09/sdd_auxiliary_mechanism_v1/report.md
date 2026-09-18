# SDD Auxiliary Mechanism Controls: Final Report

## Scope and Provenance

Result source: **54 fresh real Torch fits**, plus **54 cached_verified parent
fits**, not 108 new fits. All fixed comparisons are complete. New training totals
270,000 optimizer updates, including the 100-update pilot resumed in the same
checkpoint. Summed fitting time is 8,251.39 seconds (2.29 hours); the full trainer
wall interval is about 2.33 hours. Native arm64 CPU, four Torch threads, one
interop thread and zero data workers were used. No HPC job was required.

The main task and primary remain unchanged: 11,966 fit windows, observe eight
and predict twelve native annotation steps, past-normalized ADE with equal
physical-site weighting. The three folds are ETH, Hotel and grouped Zara.
These are repeatedly exposed fit sites, not independent confirmation. Students,
development, calibration and confirmation roles remain closed.

The admitted SDD source is still the original 40 training recordings, 229,333
past-eligible windows, stride 12 raw frames and endpoint +144 raw frames. Source
and main steps are not assumed to represent equal time. Retrospective supplied
annotations are disclosed; this is not strict sensor-as-of prediction.

## Fixed Comparison

| Schedule | Supervision | Training updates | Evidence origin |
| --- | --- | ---: | --- |
| no_aux | Main fit only, two phases | 2,000 + 4,000 main | cached_verified |
| sdd_aux | Real SDD, then main | 2,000 source + 4,000 main | cached_verified |
| main4k | Random initialization, main only | 4,000 main | fresh_run |
| sdd_permuted | Permuted source residual labels, then main | 2,000 source + 4,000 main | fresh_run |

Each schedule has geometry, coverage-mask and past-RGB inputs, seeds 17/29/43,
and all three site folds. The 4,000-update main stream is matched exactly.
Source permutation preserves source draws and changes only training loss labels,
within original recording and exact future-label-support strata. No future label
enters inference. Final checkpoints are fixed; no held-score model selection.

## Primary Results

Positive numbers mean lower ADE than constant velocity (CV).

| Input | Main 6k | Main 4k | Real SDD + main 4k | Permuted SDD + main 4k |
| --- | ---: | ---: | ---: | ---: |
| Geometry | -1.34976% | -1.03110% | -0.80523% | -0.92421% |
| Coverage masks | -1.37243% | -1.00276% | -0.81626% | -0.80078% |
| Past RGB | -1.98266% | -1.52648% | -1.27335% | -1.33240% |

Every schedule/input aggregate remains worse than CV. Of 54 new individual
seed/site fits, three have tiny positive gains (+0.00418%, +0.00423%, +0.03958%),
but **0/54 pass easy preservation** and **0/54 are safe positive fits**. The
combined matrix has 0/108 safe positive fits. No checkpoint is promoted.

New-fit easy degradation ranges from 195.12% to 11,392.47%; the corresponding
per-fit absolute normalized ADE harm ranges from 0.02593 to 0.29810. Near-zero
baseline errors inflate percentages, so both are retained. Training-cohort gains
are positive (0.53496% to 1.90869%), but do not establish held-site transfer.

## What the Controls Resolve

1. Shorter main exposure itself helps: the main-6k models are worse than main4k
   in all nine seed-averaged input/site comparisons. This does not select 4k as
   an optimal stopping point or certify it as safe.
2. Real source supervision has small gains relative to main4k: geometry
   +0.22356%, mask +0.18465%, RGB +0.24933%. Only the RGB descriptive site
   interval excludes zero in this comparison; it still loses to CV and its mask
   control. Three exposed sites cannot provide independent confirmation.
3. Shuffled source labels also help relative to main4k. Real versus shuffled
   source is +0.11789% geometry, -0.01536% mask and +0.05827% RGB. All three
   descriptive site intervals cross zero. Correct input-to-future source pairing
   has not shown a stable advantage in this tested design.
4. In an arithmetic decomposition of the old source-versus-main6k absolute ADE
   difference, the main4k-versus-main6k component accounts for 58.52%, 66.47% and
   64.31% respectively. These are not causal mediation fractions; percentage
   improvements are not additive.

| Real source vs permuted source | Gain | Descriptive 95% site interval |
| --- | ---: | --- |
| Geometry | +0.11789% | [-0.12649%, +1.89699%] |
| Coverage masks | -0.01536% | [-0.15366%, +0.15464%] |
| Past RGB | +0.05827% | [-0.17034%, +0.27215%] |

The 2,000 bootstrap draws resample only three reused physical sites after seed
averaging. Overlapping windows are not independent uncertainty units. Failure
to distinguish real and permuted source is not proof of exact equivalence.

## Failure Evidence and Next Test

The [failure analysis](failure_analysis.md) separates direct observations from
hypotheses. A train-only source audit finds a stationary normalization mismatch:
the fixed 0.001 native-unit floor is not invariant between pixel and other local
coordinates when history motion is zero. Broad static-to-any-motion normalized
target medians are 1,125 in source versus 24.702 in main fold-0 training data.
The analytic scalar log-loss sensitivity differs strongly; actual parameter
gradients and causal responsibility for transfer failure are not established.

The shortest justified next repair is a registered source-representation and
loss-sensitivity control for stationary/near-stationary histories, with synthetic
unit-rescaling invariance checks before further training. Keep the main primary,
cohort and closed roles fixed. A main-task/primary change would require a separate
scientific decision. Do not launch another threshold search on these weak
forecasts or treat more overlapping source windows as new independent starts.

## Verification and Limits

All 54 new checkpoint predictions replay exactly. Checkpoint steps, finite
parameters/losses/gradients, main sample counts and final sampler states are
verified; source sample counts match for permuted controls. Completed-run resume
preserves 166 immutable artifact hashes and adds zero updates. Twenty-three
focused tests pass; the full legacy suite was not rerun.

See [all fits](fit_metrics.csv), [contrasts](contrasts.md),
[machine-readable analysis](analysis.json), [source audit](source_support.md)
and [reproduction](reproducibility.md). The source cache and checkpoints stay
local and are excluded from Git. Main observation roles, split lineage and
future-label separation are unchanged, not a new sensor-causality certification.

No deployable neural candidate, independent generalization claim or submission
readiness is established. No metric, seconds-level, true-3D or foundation claim.
Stage5C and SMC remain disabled. The long-term research goal remains active.
