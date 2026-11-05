# Source-Held Forecast Donors Inside Producer Training

I trained six new EqMotion controls to expand risk-learning source coverage
without moving outer calibration or evaluation sources into training.
Each model completed 4,000 updates; primary total 24,000, with a further 40
updates for the direct recovery control. One seed, not a three-seed result.

**The donor bank is complete. Better safe intervention is not yet established.**

The bank contains 125,540 rows from 14 source components, partitioned
by parent: 6 sources for producer A0 and 8 for producer A1, never blindly pooled
across parent contexts. 122,716 rows have some future labels,
90,965 have all 12, and 2,824
have no valid future labels. Unknown is not safe. Row IDs, teacher ancestry and
input/label arrays are recorded separately. Inputs remain past-only by annotation
index; the source annotation-generation process is not proved temporally causal.

## Training

| Model | Fit sources | Donor sources | Donor rows | First / final batch loss | Fit seconds |
|---|---:|---:|---:|---:|---:|
| eqmotion_A0_inner0_seed17 | 4 | 2 | 6197 | 7.027591 / 0.663961 | 1309.3 |
| eqmotion_A0_inner1_seed17 | 4 | 2 | 2734 | 7.279579 / 0.442291 | 1389.0 |
| eqmotion_A0_inner2_seed17 | 4 | 2 | 45025 | 6.910533 / 0.532024 | 1302.3 |
| eqmotion_A1_inner0_seed17 | 5 | 3 | 5521 | 7.021617 / 0.619711 | 1399.4 |
| eqmotion_A1_inner1_seed17 | 5 | 3 | 19742 | 13.816051 / 0.705646 | 1372.8 |
| eqmotion_A1_inner2_seed17 | 6 | 2 | 46321 | 10.865404 / 0.693550 | 1371.8 |

Summed primary fitting time including the reused pilot prefix: 8144.5s.
First/final values are sampled batch losses, not an epoch-level comparison or
proof of convergence. Every final checkpoint is fixed at 4,000 updates.
All nonfitting source sample draws are zero. Six forecast caches reproduce
exactly from final checkpoints. Replay is verification, not independent evidence.

## Why Not Pool The Old Caches?

The metadata-only compatibility matrix rejects direct cross-context pooling.
A shared teacher does not make another context's outer H a training donor.
Different teachers can have seen the intended critic sources and can use
different training-defined risk targets. The new forecasts instead come from
models fitted only inside the appropriate parent A, excluding each donor source.

These are smaller source-trained teachers: each fits four to six sources,
versus the original six or eight. Their error distribution can differ.
More sources do not automatically yield calibrated risk estimates.
OOF donor ADE is lower than causal CV on 10/14 sources and higher on
4/14: moving_source_component_047, moving_source_component_062, moving_source_component_076, moving_source_component_082. These negative sources are retained, not
filtered after observing their errors. This is not a strongest-baseline result.

## Evidence Status

- fresh_run: subset packing, six real Torch fits, OOF predictions and donor diagnostics.
- cached_verified: pinned source admission, partitions, code and source hashes.
- not_run: augmented risk-head comparison, new outer C/H readout and deployment.
- The original 2% all/easy harm, signed easy-net and support requirements remain.
- Empty controls remain NA. No identity-fallback certificate. Old failures remain.
- Original 24-group main estimand remains NA; this does not repair group068.
- Native obs8/pred12 image-coordinate silver labels; no meters, verified seconds,
  true3D, foundation, Stage5C execution, SMC or submission-ready claim.

## Next Experiment

Compare the same risk model and fixed decision rule with own-B-only donors
versus parent-A OOF augmentation, on the already-exposed own-B development
screen. Preserve query-source exclusions and compare teacher-size effects.
Do not open a new test or select thresholds on outcomes. A failed donor bank
comparison would reject the coverage remedy; a successful one still needs
independent confirmation under the original complete policy contract.

[Protocol](protocol.md) | [Source diagnostics](donor_diagnostics.md)
| [Loss trace](loss_trace.csv) | [Reproduction](operation_zh.md)
