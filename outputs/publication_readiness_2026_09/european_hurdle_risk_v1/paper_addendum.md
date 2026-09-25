# Development Addendum: Separating Harm Occurrence from Severity

25 September 2026. This addendum complements the
[evidence-bearing manuscript](../evidence_manuscript_v1/manuscript.md) without
pooling its SDD results with European Squares. It is not a new paper abstract,
independent confirmation, or submission-readiness claim.

## Method and Prior Art

We test whether a baseline-relative risk controller benefits from explicit
supervision of harm occurrence and conditional severity. For causal inputs X,
candidate and constant-velocity forecasts define a deterministic envelope D,
the maximum Euclidean separation across the requested prediction steps. The
future-label mask is not used in D. For a predefined evaluation event E, let
H_E be positive candidate ADE increase multiplied by its event indicator.
Unknown labels are excluded rather than assigned zero.

The head predicts reference-error event mass B, joint probability
P(H_E > 0 | X), and conditional normalized severity
E[H_E / D | H_E > 0, X]. Its harm moment is D times the latter two outputs.
The architecture-matched control optimizes moment MSE only; the hurdle arm adds
unweighted occurrence cross-entropy and positive-only severity MSE. The utility
head, forecasts, candidate families, fitting draws and 2% decision budget are
fixed. This is risk-head supervision, not a new trajectory predictor.

Two-part modeling of a zero mass and a positive continuous response is
established, including [Yiu and Tom (2017)](https://arxiv.org/abs/1703.09147).
Our adaptation to forecast-relative harm neither establishes structural novelty
nor inherits a risk guarantee. The study asks about the resulting decisions.

## Controlled Results

We fit 72 heads across three seeds and three source-role folds and retain 144
policy views. Each fit uses four localities and excludes eight throughout the
producer chain; all twelve localities have already been opened for development.
Intervals use 3,000 paired locality resamples and are conditional, dependent
across comparisons and not multiplicity adjusted.

Explicit factor supervision reduces the maximum positive-easy degradation
across neural views from 17.2546% under matched product MSE to 0.6746%.
Nevertheless, 12/18 views harm zero-reference-error cases; the other six contain
none. Only four distinct zero-reference rows exist, concentrated in one locality.
The limited and uneven support prevents a claim of generalized protection.

Against product MSE, all-event neural controllers improve all-ADE in 8/9
comparisons, seven with positive intervals. Easy-event controllers worsen
all-ADE in 9/9 comparisons, all with negative intervals. Against identically
protected damping, none of the 18 all-ADE intervals is positive and 15 are
negative. Two hard-subset intervals in one fold favor the neural candidate by
0.5874% [0.2603%, 0.9032%] and 0.3308% [0.0601%, 0.6326%]; 15/18 hard intervals
favor damping. All results, rather than a selected winner, are retained.

A posthoc fitting-prior comparator supports some learned occurrence signal:
neural easy-event Brier scores improve in all nine views, eight with positive
intervals. This diagnostic does not isolate matched-coverage ranking, certify
selected-tail calibration or establish neural dynamics value.

## Limitations and Reproducibility

The experiment supports a narrower conclusion than the original world-model
hypothesis: supervision decomposition can alter protection and coverage but
does not establish stable safe neural superiority. Complete source-chain
exclusion does not turn repeatedly inspected development sources into a final
test. Four zero-reference rows are insufficient evidence for rare-stratum safety.
The next comparison must separate risk ranking from reduced intervention count
and test any changed rule with producer-excluded calibration.

All 72 checkpoints and samplers replay within the stated scope; all 144 full
metric views and 72 prior control views reproduce. The scoped suite passes
218 tests. See [execution notes](execution_notes.md), [all results](results.md)
and [failure analysis](failure_analysis.md). Data and checkpoints remain private
local artifacts, so the public aggregate package is not a stand-alone training
reproduction bundle.

Released detector tracks use image pixels, eight observed and twelve predicted
steps at raw stride 12. These are not legacy t+50, seconds, metric coordinates,
human-gold labels or physical-safety outcomes. No true-3D, foundation-model,
independent-generalization or deployment claim is made. Stage5C and SMC remain
unexecuted.
