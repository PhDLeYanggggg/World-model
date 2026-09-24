# Method Positioning and Claim Limits

Primary-source reading on 2026-09-24. This note changes neither the registered
experiment nor its thresholds. It is not an exhaustive novelty search.

## Relevant Prior Methods

**Conformal Risk Control (CRC).** Theorem 1 controls expected bounded loss with
exchangeable loss functions, monotonicity, right-continuity and a safe endpoint.
Its finite-sample adjustment is part of the method, not a fitted risk model.
These assumptions must be checked for the actual unit of inference; overlapping
trajectory windows do not supply thousands of independent physical scenes.
[Primary paper, Theorem 1](https://arxiv.org/pdf/2208.02814).

**Learn then Test (LTT).** This separates model construction from an i.i.d.
calibration sample and casts admissible parameter selection as multiple testing.
Its family-wise error control allows selection from an accepted parameter set
under its assumptions. That is different from repeatedly inspecting development
outcomes and choosing a rule with the best bootstrap interval. The original
paper's formal setup and risk-controlling prediction theorem were inspected.
[Primary paper, Sections 1.1 and 2](https://arxiv.org/pdf/2110.01052).

**Conformal Policy Control (CPC).** The inspected March 2026 v1 studies regulating
an optimized explicit policy relative to a safe reference via calibrated
likelihood-ratio clipping. It is directly relevant to baseline-protected
intervention. Its nonmonotonic-loss theorem has smoothness and stability
conditions; the policy result additionally requires appropriate weights and a
safe reference. The paper distinguishes exact theoretical weights from practical
approximations. We cannot import its guarantee by renaming deterministic forecast
selection a conformal policy. This version-specific reading is not a verification
of every later revision or implementation.
[Primary paper, Sections 4.1-4.2, Theorems 4.2/4.5](https://arxiv.org/html/2603.02196v1).

## What Is Different Here, and What Is Not Yet a Contribution

Our present object is a deterministic subset of simultaneous agent forecasts in
one recording/frame. The experiment measures two elementary accounting choices:
sharing signed predicted benefit and borrowing denominator mass from agents left
on CV. It preserves the forecast bank and reports same-query count controls.
This is a useful mechanism test, not a new risk-control theorem. Shared budgets,
knapsack selection, and a fallback policy are not by themselves novel.

Predicted feasibility is not observed easy preservation. In particular,
`max(E[net harm | X], 0)` is not `E[positive harm | X]`; expected benefit can still
mask within-agent harm. A positive-CV easy-subset average also omits zero-CV
harm unless that is reported separately. Unknown labels cannot be counted safe.
Even zero observed harm would not create a population guarantee.

The controller's realized loss need not vary monotonically as its allowed
intervention budget changes: its selected identities change, not just coverage.
The current learned moments, fixed numerical budget, and four-site bootstrap are
not CRC or LTT calibration. The four physical sites have influenced method
development. Excluded-site model fitting does not undo that exposure.

## Evidence Needed Before a Stronger Claim

1. Freeze the complete predictor, controller, loss, inference unit and source
   eligibility before admitting genuinely independent calibration/confirmation.
2. Establish meaningful gains over predictor-matched independent selection and
   protected simple motion at matched intervention or risk levels. A neural
   forecast is not indispensable merely because its protected score is positive.
3. Demonstrate any multi-agent interaction contribution separately from a shared
   scalar budget. The preceding negligible pairwise effect remains a negative
   result, not a reason to relabel budget sharing as learned interaction dynamics.
4. If claiming formal risk control, specify a bounded loss and justify the chosen
   method's assumptions at the physical-site sampling level. A source bootstrap
   supports descriptive uncertainty only; it does not replace calibration.

No independent labels are opened for this comparison. No CPC sampling, latent
generative execution, SMC, deployment change, metric/seconds claim, foundation
claim, or submission-readiness claim is made.
