# Why Better Component Scores Still Fail The Complete Policy

## Material Passport

Experiment: `moving_complete_donor_policy_v1`, native image-coordinate obs8/pred12.
Scope: 8 already-exposed own-B sources, 24 recording views. Fresh full-policy
inference and evaluation; frozen cached models. No new final test or training.
Status: completed negative original-risk/support result; deployment unchanged.

## Three Different Failures

1. **Actual positive harm, not just missing denominators.** Mean violates known
   easy positive-harm risk in 15/24 views, unanimity in 9/24. Unanimity has no
   signed easy-net violations, but this does not erase the positive harm.
2. **Censored labels still contribute risk.** Mean admits 81 wholly unknown
   rows; unanimity admits 11. The original causal path-disagreement envelope
   raises completion-all violations from 9 to 18 views for mean, and 4 to 9 for
   unanimity. Unknown outcomes are not fabricated labels or safe observations.
3. **Selected easy-reference support is lost.** Mean has 6 undefined easy
   ratios, unanimity 12. All 24 recordings contain observed easy reference in
   the unselected population. Thus this cohort's missing selected support is
   not a structural absence of easy events in the raw recordings. Forest has
   22 undefined easy views and is not a passing comparator either.

The mutually exclusive diagnostic gives mean 16 observed-violation views,
4 completion-only violations, 3 remaining unsupported views and 1 pass.
Unanimity has 10, 4, 8 and 2 respectively. Categories use observed violation
first, completion second, missing support third; overlapping risk counts in
the main table should not be added.

## What Improved

Mean improves available-label source-macro ADE 1.3130% versus the training-defined
floor, unanimity 0.3359%. Conservative completion lower gains are 0.8238% and
0.3056%. Paired lower contrasts versus forest have positive exploratory
source-bootstrap intervals. This is useful trajectory information on exposed
development data, not a complete policy certificate.

The earlier donor treatment improves risk learning across seeds, but that
objective was largely fitted to observed outcomes. Its signed excess scores
and the fixed zero decision rule do not guarantee selected reference mass or
the recording-level positive-harm ratio. Shared seed errors also survive
unanimity. Removing intervention therefore cannot be equated with passing.

## Model-Side Repair

Keep the original evaluator, 2% limits, support requirements and NA semantics.
Compare known-only risk learning with complete-risk donor supervision that
includes the existing conservative envelopes for wholly unknown rows.
Preserve whether a target is observed or an upper bound. Fit preprocessing
on training sources only, exclude the entire query source, and keep the same
causal feature interface, budgets, seeds and zero threshold. Report both
per-channel errors and complete recording-risk/support results.

The envelope treatment alone may remain insufficient: reference-moment and
selection-aware risk modeling are a separate unresolved issue. Do not silently
mask unknown queries using their future validity, discard unsupported views,
add an identity exception or select thresholds on this development readout.

## Limits

This is not independent confirmation. Only eight exposed source components
support the exploratory bootstrap. Available-label ADE is not official
complete-12 ADE. The original 24-group main estimand remains NA, not passed.
Source annotation-generation causality remains unverified. No metric/seconds,
true-3D, foundation, human-gold, Stage5C or SMC claim follows.
