# Method Positioning and Unproven Claims

2026-09-24. A bounded primary-source check during the registered source-training
run, not an exhaustive novelty review or a change to its experimental settings.

## Existing Work We Must Not Reclaim

SelectiveNet jointly learns a predictor and reject function, including regression,
and studies error versus retained coverage. Merely learning when not to use a
neural prediction is therefore not a new contribution. Our proposed comparison
differs in evaluating every indexed agent through an explicit, strong causal
fallback, and measuring added error relative to that fallback, rather than
reporting only the accepted subset. This is a task distinction to validate, not
proof of novelty. [Geifman and El-Yaniv, ICML 2019, Sections 2 and 7.2](https://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf).

Conformal Risk Control gives an expected-risk result for exchangeable,
upper-bounded, monotone loss families, with additional conditions stated in its
theorem. A threshold on a learned harm score is not that procedure. Raising a
joint optimizer's budget can change which agents are selected, so realized
relative error need not be monotone. We cannot attach the theorem to the present
optimizer without establishing an appropriate loss family and sampling unit.
This source-only study performs no such calibration.
[Angelopoulos et al., ICLR 2024, Theorem 1 and Section 2.3](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf).

Learn then Test frames calibration as hypothesis testing with valid p-values
and family-wise error control. Its setup uses independent calibration examples;
it does not make overlapping windows independent. A policy ordering must not
be chosen using the same calibration outcomes without the corresponding
data-splitting construction. It is a possible existing calibration tool, not
our theoretical invention. Twelve reserved localities may still provide too
little power for the desired risk target.
[Angelopoulos et al., arXiv:2110.01052, Sections 1.1 and 2.3](https://arxiv.org/pdf/2110.01052).

A search also surfaced a newer item titled *Conformal Policy Control*. Its
OpenReview full-text access hit a browser-verification page in this check.
That item has **not** been substantively assessed and must be resolved before
claiming an exhaustive comparison. Search-result text is not a completed review.

## What This Experiment Can Actually Test

1. A neural candidate must add value beyond history-OLS and damped motion, not
   merely beat a noisy last-difference reference.
2. Cost labels must be produced out of fit, excluding the eventual outer
   locality at both the predictor and cost-head levels. No training exposure
   can be repaired by simply renaming a prediction file.
3. Independently predicted benefit and harm can fail separately. Report their
   errors and correlations; high net utility does not imply easy protection.
4. At one predicted harm cap and exact intervention count, joint selection must
   improve on both independent scores and additive geometry. Otherwise there
   is no evidence for a nonadditive interaction contribution.
5. A lower predicted overlap proxy is not evidence of fewer real collisions.
   Image coordinates, detector boxes and forecast-only proxies limit the claim.

## Zero-Reference Boundary

The existing zero-reference allowance remains zero. For Euclidean loss, if a
candidate differs from the baseline at a scored point, the possible outcome
equal to that baseline makes the baseline exact and the candidate harmful.
This elementary counterexample is not a new theorem: without restrictions on
possible outcomes, a pointwise zero-harm guarantee cannot come from a learned
confidence score alone. An observed absence of zero-reference harms also does
not establish a population guarantee.

The experiment therefore reports actual zero-CV harms separately, including
harm already introduced by the training-selected fallback. It does not relax
the tolerance, discard zero-error rows, or insert an epsilon into percentages.
If the selected fallback itself harms those rows, describing the entire
pipeline as zero-reference safe would already be false.

## Claim Status

The current method is a falsifiable proposal for baseline-relative, scene-level
intervention on detector-track forecasts. It is not yet a new calibration
theory, validated physical safety system, multimodal foundation model or
submission-ready main contribution. Reserved confirmation stays closed until
method selection and the applicable risk protocol are frozen.
