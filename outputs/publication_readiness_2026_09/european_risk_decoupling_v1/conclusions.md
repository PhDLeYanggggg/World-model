# Protected Risk Decoupling: Fitting Screen Failed

## Material Passport

Fresh real-Torch training:36 matched continuations,72,000 new optimizer updates.
Fresh frozen fitting readout:72 context/source/head-seed views. Original controls
are cached and hash-verified, not newly trained forecasters. All data remain
exposed original TRAIN. No new held inference, outer development, independent
selection/calibration/confirmation, deployment, Stage5C or SMC.

## Question and Answer

Does separating conditional risk estimation from soft-action optimization
repair risk estimates while preserving frozen benefit/reference predictions?

**It improves average fitting MSE but fails the registered conditional-risk
screen and does not establish safe selection.** Seven of nine fitting routing
conditions pass; the original selected-cohort easy-risk condition and the
12/18 paired-improvement condition fail. A failed condition is not waived because
the numerical change is small. No held comparison follows this fit-only run.

## Fixed-Cohort Results

Each entry is a source-normalized signed-risk MSE, averaged equally across72
views. Signed risk is H-0.02R, with the corresponding easy moments for easy risk.
These are not trajectory ADE/FDE or raw error percentages. The earlier
conditioning diagnosis reported pooled residual-square sums divided by known
occurrences; those occurrence-weighted MSE values must not be compared directly
with this protocol's equal-view averages. Original predictions and cohort
hashes reproduce exactly despite the different aggregation.

| Fixed cohort and target | Frozen original | Coupled continuation | Estimation-only continuation |
|---|---:|---:|---:|
| All eligible, all risk | 0.380675189 | 0.286251349 | 0.075039640 |
| All eligible, easy risk | 0.005080316 | 0.004187225 | 0.002253197 |
| Original selection, all risk | 0.079458745 | 0.194612684 | 0.078065818 |
| Original selection, easy risk | 0.000309373416 | 0.000333339974 | 0.000309426746 |

Relative to the original, estimation-only reduces eligible all/easy MSE by
80.29%/55.65%. Original-selection easy MSE instead worsens by0.01724%, essentially
unchanged in practical magnitude, not a successful repair. Only4/18 context/head-
seed pairs improve that target over BOTH comparators; the fixed minimum is12.

All eligible cohorts contain894,975 repeated occurrences:876,135 known and18,840
unknown. Original selection contains62,902:61,335 known and1,567 unknown. These
are repeated context/seed occurrences, not independent sample sizes. Unknown
outcomes are counted in inference/support, not imputed into supervised MSE.

On the identical original selection, observed normalized easy-harm mass is
95.560642. Predicted mass is1.375607 original,16.934458 coupled and2.268195
estimation. Estimation still labels2,799 known occurrences predicted-safe when
their observed signed easy risk is positive, versus2,825 original. Total-mass
underestimation therefore persists despite a substantially better global MSE.
Masses sum source-normalized repeated occurrences; they are not pooled ADE/FDE.

## Policy Diagnostics

The causal decision rule, eligibility and2% criteria are unchanged. Each arm's
policy chooses its own cohort, so these are complementary diagnostics, not the
fixed-cohort MSE comparison above.

| Diagnostic | Original | Coupled | Estimation |
|---|---:|---:|---:|
| Selected repeated occurrences | 62,902 | 123,436 | 419,270 |
| Selected unknown outcomes | 1,567 | 2,161 | 7,799 |
| Selected / eligible occurrences | 7.03% | 13.79% | 46.85% |
| Known easy positive-risk violations | 25/72 | 13/72 | 62/72 |
| Undefined easy selected risk | 7/72 | 6/72 | 4/72 |
| Full completion support | 1/72 | 0/72 | 0/72 |

Estimation minus coupled completion-lower utility is-1.401134 percentage points
for full policies and-0.046503pp after matching intervention counts. These are
descriptive fitting estimates without confirmatory CI. They are not evidence
that either continuation is safe. The matched deficit means that increased
coverage is not the whole explanation, although its magnitude is small.

## Interpretation and Boundary

The action-dependent losses influence average cost fitting and intervention
coverage. Removing them is not enough to estimate the rare harm in the selected
region. It also removes an action-constraint training signal, which may explain
part of the large coverage increase. This experiment removes regret and dual
terms together and starts from action-trained weights; it cannot isolate them
individually or establish from-scratch superiority.

The result is narrower than "risk regression is useless" or "causal features
contain no signal." The positive average fitting result is real, but the desired
conditional safety result is absent. No threshold search, favorable seed/source
selection or new independent readout was used to rescue the candidate.

## Execution and Verification

PilotPID67431:300 updates,21.953444s,exact resume and protected predictions pass.
TrainingPID67973 exit0:370.459281s,peak RSS2,901,655,552 bytes. All36 encoders
actually updated; final private checkpoint payload17,675,280 bytes. Training
freeze committed in`1419e149` before inference. ReadoutPID69017 exit0:100.449795s.

Full frozen-model inference is replayed using `check`; the separate verifier
reconstructs864 cohort-target,216 policy and72 matched-count checks plus all nine
routing conditions. See[verification](verification.json) and
[reproduction](reproducibility.md). Thirty-two scoped tests pass; no legacy
artifact-writing full suite was run. The local figure was visually checked;
only its receipt, not the image or checkpoints, is committed.

The project remains pixel/local,rawstride12,obs8/pred12,detector-silver. No metric,
seconds-level,human-gold,physical-safety,true3D or foundation claim. Historical
contaminated stage scores are not recertified. The long-term goal remains active.
