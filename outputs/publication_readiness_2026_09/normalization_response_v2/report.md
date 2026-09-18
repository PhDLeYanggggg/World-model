# Actual Gradient Response and Observed-Unit Repair

## Scope and Provenance

Fresh autograd diagnostics, zero optimizer updates. Existing arrays and frozen
training checkpoints are hash-verified. The v1 measurement took 95.22 seconds
(432 records); the explicit v2 repair took 49.02 seconds (216 records).
Neither run evaluates forecast improvement. Original reports are preserved.

Main gradient analysis uses only fold-0 training: 9,352 Hotel/Zara windows.
The auxiliary population is the approved original 40 SDD training videos,
229,333 windows; 188,358 have complete future labels for these gradient probes.
Main cache files are integrity-hashed, but held-main target rows are not analyzed
by this diagnostic. Development/calibration/confirmation and source val/test
remain closed. Future targets enter losses and event analysis, never inputs.

For each of five supervised motion events, select at most 32 anchored,
complete-label windows with a fixed seed. Selection hashes and track counts
are saved. Repeat three seeds and geometry/mask/RGB inputs. These correlated
training samples are not independent validation units. Partial-label and
unanchored rows are not silently removed from the training population.

## Parameter Gradients, Not Scalar Derivatives

Mean per-row full-parameter gradient norms, averaged across three seeds and
three modalities; measured before gradient clipping:

| Domain / condition | Static history, later movement | Other motion |
| --- | ---: | ---: |
| Main / legacy initialization | 0.018786 | 0.289135 |
| Main / legacy final training checkpoint | 0.032870 | 0.737401 |
| Main / v2 radius decoder, primary log loss | 79.537275 | 39.667727 |
| Main / v2 radius decoder, internal log loss | 0.220162 | 0.312384 |
| SDD / legacy initialization | 0.001522 | 0.304346 |
| SDD / legacy final training checkpoint | 0.003087 | 1.449280 |
| SDD / v2 radius decoder, primary log loss | 280.277629 | 4.066242 |
| SDD / v2 radius decoder, internal log loss | 0.255504 | 0.314430 |

The legacy trained model has small sampled start-event gradients, so the
previous scalar-derivative concern was not merely an algebraic speculation.
However, the measurements do not show that increasing gradients improves
prediction. Direct radius decoding under primary log loss can instead greatly
amplify gradients. Internal log loss reduces this imbalance at initialization.
Checkpoint response is measured on training data, not out-of-sample performance.
Gradient cancellation and clipping prevent norm ratios from being literal
optimization-influence ratios. Zero encoder gradients at zero-head
initialization are expected; this is not representation collapse.

## Invariance Repair

The original observed-unit frame still inherited baseline rollout diagnostics
computed with a native-unit turn-speed cutoff. A tiny curved isolated history
rescaled by 1e-4 exposes the counterexample. v2 recomputes rollout *features*
after past-only internal conditioning. Main evaluation baselines and the
primary normalization are unchanged. Preserve v1 as a documented limitation,
not a silently corrected positive result.

The internal radius uses observed ego/neighbor geometry only, without the
old fixed native-unit floor. Exact-zero spatial context has no identifiable
positive length: main fold-0 train has eight such rows; SDD has 68, of which
51 have complete labels. Predictions for unsupported rows must fall back to CV.
v2 tests cover the curved counterexample and independence from cached rollout
mutation. Combined transform, actual-gradient, training-resume and adapter
checks: 22 passed. These are engineering checks, not scientific confirmation.

## Next Falsifiable Test

The fixed geometry-only comparison separates unit inputs, radius decoding and
internal loss. It preserves all main/source populations, primary evaluation,
three physical-site folds and three seeds. All outcomes will be retained, with
no held-score model selection. A 100-update real pilot completed and saved
state without held evaluation; it resumes into the full 27-fit matrix.

This remains offline annotated-history, dataset-local/pixel raw-frame research.
No metric, seconds, true-3D, foundation, independent confirmation, deployment,
Stage5C or SMC claim follows from the audit.
