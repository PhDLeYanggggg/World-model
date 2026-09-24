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

## Completed run

The corrected 16-tree pilot resumed into the same registered 128-tree budget.
Training PID 75451 exited successfully after all 36 fits; evaluation PID 76620
and exact replay PID 76735 also exited successfully. The independent arithmetic
verifier and report exporter completed with no post-fit training-code repair.
Every forest, old cost score and selected decision was checked against its hash
or exact replay. The independent verifier covers 216 decision arrays, 1,728 scene
reductions, 27 contrasts and the incomplete-future bounds.

The completed readout is negative for the proposed utility improvement. New
pointwise joint-moment gates nearly suppress neural intervention; they do not
improve the matched product rule or old strict neural policies. There is no
post-readout threshold repair or winner deployment. The exporter adds a clearly
labelled descriptive rejection analysis without changing any fitted model or
decision. Generated figure labels were visually checked for legibility.

Registration/start commit: `3e2b232557edf2a160ea19ee3e8804177d89869d`.
Analysis SHA256: `51f3a8fb50b21ec1c02e7aaee37baf42ebd4e52310d05573ca7f3c75f064800c`.
