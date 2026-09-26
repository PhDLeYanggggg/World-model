# Failure Analysis

## Hypothesis Tested

The prior strong-control reconstruction ruled out an accidental architecture
or sampling change as the sole reason for auxiliary failure. This study then
tested whether end-state shared-gradient interference supplies a supported
repair direction. It does not pass the registered allocation screen.

## Evidence-Supported Findings

1. Raw task conflict is present but not universal. In the full cap-auxiliary
   states, 120/576 four-cost comparisons conflict; median cosine is positive.
   The auxiliary/main shared norm-ratio median is 0.39594, and 0.14888 against
   easy-harm. These are dependent fitting diagnostics, not causal coefficients.
2. The actual intervention is inconsistent. Even within those 120 conflicts,
   66 projected updates worsen positive-envelope easy-harm probes versus the
   unprojected update. Projection is along the four-cost gradient, not a
   guarantee for that individual component. AdamW moments, global clipping,
   finite steps, curvature and differing probe batches can all matter; this
   study does not isolate their contributions.
3. Task information remains unsupported. Projected true vs projected shuffled
   gives one positive, two negative and three overlapping easy-harm intervals.
   A geometric adjustment has not made the useful part of the label reliable.
4. The tested change is tiny at this late state. No inference from one-step
   fitting effects to trajectory accuracy or thousands of future updates is
   warranted. The cost-only model's initially zero auxiliary head also limits
   what a single newly introduced auxiliary step can reveal.

## Remaining Unknowns

Earlier easy-membership gradient and severity-transport diagnostics already
failed to justify a broad conflict/resampling repair. This cap-event study
uses different labels and paired projection interventions; it is not evidence
that no one previously tested task interference. Prior negative results remain.

- Early/middle training gradients were not saved in the parent checkpoints.
  The final-state diagnostic cannot rule out a conflict earlier in training.
- The relative contributions of rare-label severity, squared-loss tail
  concentration, nested-reference transport and optimization budget remain
  unresolved. Binary-event discrimination need not imply severity prediction.
- Source fitting probes cannot settle cross-scene or independent-scene utility.
- No new scene-joint policy, intervention-rate matched comparison, independent
  calibration or confirmation was run. Historical exposed results remain exploratory.

## Engineering Failures Kept Separate

The report originally concatenated integer assignment IDs, then strict JSON
serialization rejected a NumPy boolean. Both were repaired with explicit
source amendments and regression tests. They did not affect any virtual
update, metric formula, criterion or data split. Engineering repair does not
turn the negative scientific result into a passed gate.

## Decision

Do not launch a full projection sweep under a claim that the mechanism is
already established. Keep all current scientific gates and deployment status
unchanged. Test the remaining training-stage/severity hypotheses with fixed
fitting-only diagnostics and matched controls before selecting a repair.
No independent role, metric/seconds claim, Stage5C execution or SMC is allowed.
