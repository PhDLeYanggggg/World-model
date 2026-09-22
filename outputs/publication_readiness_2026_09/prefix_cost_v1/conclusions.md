# Prefix Supervision Does Not Yet Repair Scene-Wise Protection

## Evidence Status

2026-09-22. **fresh_run**: 24 Torch risk-head fits, fixed development evaluation,
checkpoint replay, separate arithmetic verification and a post-readout veto audit.
**cached_verified**: existing nested source-excluded forecasts, features, training
preprocessing and scalar reference. **not_run**: new full forecaster training,
independent calibration/confirmation, external evaluation and raw-frame t+50.
Registration `562b41d1` preceded fitting. No threshold, seed or role was selected
after readout. No model is promoted; Stage5C and SMC remain off.

The main task remains 8 observed / 12 predicted annotation steps, raw stride 12,
SDD annotation pixels. This is not a metric, seconds-level, true-3D or foundation
world model result. All four physical sites have informed development. No new
human-gold labels, physical-safety guarantee or independent test claim is made.

## What Changed

The earlier diagnosis found that zero full-trajectory harm does not imply zero
harm on every shorter prefix. I tested whether explicitly supervising that
profile repairs the decision problem. Both arms use the same 48,792-parameter
network: 356 inputs, 128 hidden units, 24 outputs. The control learns twelve
copies of the terminal benefit/harm pair; the other learns one pair per prefix.
Same initialization, complete fitting rows, sample draws, fixed training emphasis,
optimizer, budget and train-only preprocessing. Existing forecast producers and
their nested exclusions are unchanged. No future availability enters inference.

Each of 4 source sites x 3 seeds has both arms: 24 heads, 288,000 updates and
73,728,000 sampled draws, with zero unknown-label draws. Recorded fitting time
is **403.41 seconds**, excluding preparation, hashing, evaluation and earlier
forecaster training. A real 100-step pilot was resumed. This is not 24 full
world-model retrainings. Native arm64 CPU4/interop1/workers0 was sufficient.

## Fixed Results

Improvements below are equal-physical-site percentages over causal constant
velocity, after averaging errors across seeds. Negative easy degradation means
improvement. Counts sum repeated seeds; they are not independent samples.

| Fixed policy | ADE gain | FDE gain | Hard gain | Aggregate easy degradation | Worst scene/seed easy degradation | Switches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Capacity-matched terminal-repeat control | 4.19347% | 4.51128% | 4.53075% | -0.45698% | 3.96042% | 28,808 |
| Prefix-trained, terminal rule only | 4.08941% | 4.37295% | 4.37969% | -0.42379% | 3.77976% | 27,964 |
| **Prefix-trained, all-prefix guard: primary** | **1.20187%** | **1.32828%** | **1.60045%** | **0.60967%** | **2.32962%** | **993** |
| Terminal control, equal switch count | 1.92842% | 2.12417% | 2.76467% | 1.05982% | 3.25098% | 993 |
| Prefix terminal ranking, equal switch count | 1.94323% | 2.12515% | 2.77846% | 0.97894% | 2.28043% | 993 |

The primary guard loses **2.99160 percentage points** versus the capacity-matched
terminal control; paired 3,000-site-resample CI **[-3.70943, -2.41675] pp**.
At equal intervention count, it loses **0.72655 pp** to the control's ranking,
CI **[-1.05392, -0.39235] pp**. The prefix terminal ranking likewise retains more
gain at equal count. None of these policies satisfies all registered conditions.
The verified preceding scalar log policy remains 4.18873% ADE gain, with its own
protection failure; the new guard differs by -2.98686 pp, CI [-3.72748, -2.36853].

The guard's positive CV contrast, 1.20187% with CI [0.11462%, 3.24726%], is not
success against the stronger registered reference. No secondary winner is promoted.

| Primary guard, seed-averaged | ADE gain | Easy degradation |
| --- | ---: | ---: |
| coupa | 0.06422% | 0.06379% |
| deathCircle | 4.25580% | 1.42885% |
| gates | 0.22163% | 0.89284% |
| hyang | 0.26583% | 0.05319% |

Every scene's seed average is below 2%, but **gates seed17 degrades 2.32962%**.
The registered every-scene/every-seed condition therefore fails; averaging away
this failure is not permitted. The primary gain is also concentrated in one site.

## Why the Guard Fails

The following is fresh **post-readout diagnosis of frozen decisions**, not another
policy. Training and held-source results are separated, and native cost means
are reported within individual physical sites in `veto_diagnosis.json`.

1. The profile terminal rule nominates 27,964 decisions; the all-prefix guard
   vetoes 26,971 and retains 993. **26,952 of those vetoes first trigger at k=1**,
   about 99.9%. Eight first trigger at k=2, nine at k=3, two at k=6. Adding all
   prefixes is effectively a severe first-step veto in this experiment.
2. Among the 23,348 vetoed complete-label decisions, **19,890 (about 85.2%) would
   improve terminal ADE**. Of those, 7,645 do have harm on some earlier prefix:
   part of the full-versus-prefix tension is real, not just bad threshold arithmetic.
   But 12,245 vetoed complete decisions have no realized prefix harm at all.
   The learned guard is not identifying that distinction reliably.
3. Retained complete decisions include 605 terminal-beneficial and 103
   terminal-harmful outcomes; 313 have some prefix harm. Learned expected cost
   ratios are not an individual-outcome safety guarantee.
4. At fixed complete-label terminal nominations, first-prefix mean harm is
   underestimated in 21/36 fitting-site groups and 8/12 held-source views.
   Terminal mean harm is underestimated in 17/36 fitting groups and 12/12 held
   views. This remains a learning/generalization problem, not only a future-label
   support mismatch. Neither one favorable average nor fewer switches repairs it.
5. Keeping the terminal rule with prefix training also fails to improve the
   control (4.08941% versus 4.19347%). The result cannot be blamed exclusively on
   the extra guard, although the guard causes the largest loss of useful gain.

These overlapping views/seeds are descriptive reused instances, not independent
counts for a significance test. No harmful track is removed, and no future mask
is used to choose an inference prefix. Diagnosis narrows the next experiment; it
does not rescue this experiment's failed primary claim.

## Missing Outcomes and Reproducibility

Of the 993 selected decisions, 285 have incomplete future labels and 41 have no
future label at all. The veto audit separately reports 44 with no contiguous valid
prefix, which includes gapped cases and is not the same definition as entirely
unknown ADE. Complete exact-zero-CV outcomes have zero harm, but that does not
certify the missing population. Full-grid lower bounds are negative for gates in
all seeds and in additional site/seed slices. Do not discard these rows or replace
the primary supported ADE with complete-only scores.

All 24 checkpoints replay exactly, covering 1,054,536 scored profiles. Separate
formulas verify 144 fitting prefix reductions, 1,581,804 reused fitting-row
instances, 60 policy choices, 480 scene reductions, draws, bounds and gates.
The verifier's initial contrast argument error is transparently corrected in a
versioned script; the frozen original, model and analysis remain unchanged.
Same-agent/shared-source engineering verification is not independent confirmation.
See [execution record and commands](execution_notes.md). The full legacy suite
was not rerun; no full-suite claim is made.

The 3,000 bootstrap resamples use only four development-exposed physical sites.
They establish a conditional development contrast, not population-wide calibration.
Historical Stage26/37 results are not recertified by this experiment.

## Decision and Next Method Test

**Do not deploy or promote this repair.** Preserve existing artifacts without
recertifying them. Prefix labels corrected a genuine representational omission;
they did not establish a better decision policy or a publishable main contribution.

The next defensible method question is whether the all-or-nothing trajectory
action is itself too restrictive: keep the reliable motion baseline at the first
steps and introduce the neural forecast gradually, without reading future label
length. Test a preregistered deterministic temporal blend against matched uniform
shrinkage, using the same forecasts, history and fitting budget, and retain path
smoothness/interaction checks. This is a proposed next experiment, **not run** and
not permission to choose a favorable blend on these results. No latent generation.

Independent scene-level calibration/confirmation and meaningful joint-agent
evidence still remain separate priority gaps. Pending source-role and data-access
decisions must not be bypassed. No extra training duration alone closes them.
