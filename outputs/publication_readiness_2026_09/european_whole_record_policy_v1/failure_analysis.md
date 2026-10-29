# Failure Taxonomy

## 1. Minibatch Coverage Was a Real, Insufficient Explanation

The frozen TRAIN diagnostic contains 277 positive record/channel occurrences.
256 have nonpositive estimates in at least half of 256 repeated 64-row draws.
Median sample/full-gradient cosine is 0.183337; 94/288 cosines are negative.
Separate reconstruction from the original recording masses verifies all 888
risk means, with maximum absolute difference 4.01e-10. These are repeated states,
not independent experiments or proof that every failure has this cause.

Exact dual preserves the original primal draws yet does not improve either
full-policy safety or matched held utility. Therefore correcting dual statistics
alone is insufficient. This is a completed negative controlled comparison.

## 2. Full Gradients Improve Optimization, Not All Constraints

Primary fit budgets increase from 35/72 to 53/72; known easy violations fall
from five to zero. Fit utility versus stochastic is [2.981433, 2.982839] points.
However, 13 fit views still violate the known all-risk budget and only 178/444
recording views pass the full requirements. Empirical fitting is not solved.

The final pre-update traces contain 284 positive record/channel excesses and
20/888 dual entries at the cap for the primary. Exact dual has 287 positive
excesses and 32 capped entries. A final cap count does not identify whether the
cap caused a given violation. These traces precede the final optimizer step;
the frozen readout, not the trace, determines the reported policy result.

Changing dual weights also changes the training objective during optimization.
A negative final Lagrangian is not evidence that every constraint passes, nor
is it a fixed validation loss. Straight-through derivatives and finite-step
nonconvex optimization provide no empirical or population feasibility guarantee.

## 3. Recording Generalization Is Now the Dominant Observed Failure

Known all/easy violations change from 13/0 on fit to 64/60 on held views.
Full-policy budgets fall from 53/72 to 0/72. All twelve source slices have 0/6
held view budgets, and all three head seeds have 0/24. These slices overlap and
must not be counted as separate experiments.

Known benefit/harm fractions retained relative to the original policy change
from 59.36%/28.53% on fit to 52.96%/52.60% on held data. The policy learns useful
fitting decisions but does not reliably separate benefit from harm elsewhere.
This is consistent with constraint overfitting or missing predictive information;
the comparison does not separately identify those causes or a specific feature.

The worst definite full-utility source contrast versus original is locality126:
[-9.700885, -8.750454] points. It remains in every aggregate. Sources020 and082
have unresolved full utility versus stochastic because their intervals span zero;
they are not dropped to manufacture universal improvement.

## 4. Missing Labels Are Not the Sole Blocker

Held known-only violations are already 64 all and 60 easy. Five additional all
and six additional easy views pass their known-only ratio but fail the unknown
completion upper bound. Channels overlap; three easy ratios are undefined.
There are 49 easy-degradation upper-bound violations. Missing outcomes retain
the registered envelope and never become zero loss or an inference filter.

## 5. Utility and Safety Must Remain Separate

Primary beats stochastic on full and matched aggregate utility, but its safety
is worse. Primary loses matched utility to the original reference across all
three head seeds. Original also fails safety in this cohort. None of these
comparisons justifies a new deployment claim. Empty matched risk is not
estimable; its zero matched utility difference is not a safe-policy certificate.

## What to Test Next

Reconcile existing cross-fitting roles and test separate recording donors for
utility fitting and constraint updates, wholly inside original TRAIN. Compare
against this exact-gradient control with fixed complete-policy readout. Retain
the original 2% budget and all source/recording slices. The mechanism question
is constraint generalization, not another risk-MSE threshold or held-tuned cap.
This follow-up is not yet implemented or run; no outcome is promised.

Whole-record gradient accumulation is standard. The present experiment supports
a specific optimization-versus-generalization distinction, not a new theoretical
guarantee or a completed paper contribution. These are historically exposed
TRAIN recordings, not independent confirmation. No new forecaster or joint
scene policy was trained. Pixel/local, raw-frame, detector-silver only.
Stage5C/SMC remain off; deployment is unchanged.
