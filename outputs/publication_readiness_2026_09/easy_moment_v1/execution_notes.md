# Execution Record

## Pre-fit repairs

The first preflight used an incorrect parent field name (`archives` rather than
`decision_archives`); it stopped before writing an experiment identity or fitting.
The corrected parent field uses the actual frozen analysis schema.

The first pilot then stopped on an intentionally strict cutoff identity check,
before fitting any tree. For coupa/seed17, the existing shared reporting easy
cutoff is 6.246571682255203, while the bounded learner's complete-case
preprocessing cutoff is 6.727639217728246. These are distinct existing
training-only reductions, not an external-label mismatch. The new joint-risk
target must use the same shared cutoff as the evaluation population. Its
implementation now explicitly receives that cutoff; standardization and
draw-count matching retain the existing bounded learner's preprocessing.

No target, outer outcome, risk tolerance or policy threshold was selected to
improve a result. No new tree/checkpoint existed at either failure. The abandoned
preflight identity is retained locally as `aborted_preflight_identity.json` in
the experiment output directory; the repaired source/registration receives a
new identity. This is a recorded implementation repair, not a second scientific
trial or an unreported model-selection attempt.
