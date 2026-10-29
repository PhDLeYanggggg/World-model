# The Selected Tail Is Dominated by Underestimated Harm

Fresh replay of144 views completed in86.30s; all six original prediction hashes
and selection counts match. No new model was fitted. The preceding turn was
progress; this turn adds a new controlled accounting result, not a safety pass.

## What the Numbers Distinguish

For isolated harm on held recordings, normalized own-cohort easy excess is:

18.075413 = -12.735580 +38.375982 -7.564989.

The three right terms are predicted excess, harm underestimation and0.02 times
reference overestimation. The reference term is negative in aggregate: reference
error is not the dominant positive term for this policy/cohort. It can still
matter on individual rows and in the joint/local controls.

Of1,589 known selected easy-violating row occurrences,1,573(98.99%) would still
pass the risk check if only reference error were replaced by its observed
value. Replacing harm instead would reject1,456(91.63%). These are illegal
future-informed counterfactual checks, not proposed inference inputs or a claim
that these outcomes can be predicted. Unknown outcomes remain separate.

1,343/1,589(84.52%) violations lie below the inner-fit90th percentile of
standardized causal feature radius. A simple far-radius filter would not
address most observed violations. Radius is coarse; this does NOT rule out
local-support gaps, missing features or recording distribution shift.

The isolated branch's +5/+5 log-odds supremum leaves a minimum realized
easy-harm shortfall of30.113113 out of39.489891 total own-cohort harm(76.26%).
1,185 selected known row occurrences exceed this supremum. The architecture
cannot reproduce those realized labels, even with maximal shifts. Conditional
means need not reproduce individual realized outcomes, so this is NOT proof
that the ceiling is the sole cause or that lifting it will generalize.

## One Discriminating Repair

Test a full-range bounded additive probability adjustment against the frozen
odds-shift arm, with the same anchor, base freezing, width, parameters, sampling,
2,000-step continuation, loss and dual constants. Preserve the causal envelope
and nested easy-harm bound. No unbounded trajectory correction or new forecaster.
The change alters both reachable harm values and gradient geometry; even a
positive result would not uniquely attribute causality to range alone.

Retain utility/joint/local/anchor/old-risk controls and the failed isolated
control. Require matched-count utility and original full risk support; do not
promote an empty policy, pick a favorable source, tune on held or relax2%.
Do not repeat a generic hurdle/ranking objective under a new name.

## Verification Limits

89 scoped tests passed; full legacy suite not run because some tests overwrite
frozen scientific outputs.144 receipt hashes, prediction hashes, original
actions, known/unknown counts and additive decompositions replay exactly.
No independent review agent was used. No new calibration guarantee, independent
test result or submission-readiness claim follows.

Operational correction to protocol wording: heartbeat uses atomic replacement;
per-view receipts use exclusive creation, not atomic rename. A torn receipt
fails validation and must be diagnosed, not silently counted as complete.
This run completed normally, and every receipt validated. Frozen code remains
unchanged; future runners should use atomic per-view writes.

Obs8/pred12, raw stride12, detector-track local/pixel coordinates. Not metric,
seconds, physical safety, human gold, true3D or foundation. No Stage5C/SMC.
