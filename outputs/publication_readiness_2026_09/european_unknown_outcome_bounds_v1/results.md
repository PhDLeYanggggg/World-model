# Missing-Outcome Bounds: Partial Support, Not a Learned Policy Result

Result source: **fresh_run** reconstruction and bound calculation over frozen
**cached_verified** models. No new neural training or transfer evaluation.

## Main Readout

| Frozen source-validation candidate views | Number | Bound-supported | Trained-step supported |
|---|---:|---:|---:|
| All three heads across 72 fits | 216 | 15 | 3 |
| Previously rejected only for unknown selected outcomes | 41 | 9 | 3 |
| Previously supported | 6 | 6 | 0 |

The other 32 unknown-only candidates all exceed the easy selected-risk upper
budget; ten also exceed the all selected-risk upper budget. This means their
support is insufficient under the conservative bound, not that their actual
missing outcomes have been shown harmful. Existing known-outcome risk failures
are not repaired by missing-outcome accounting.

The 15 supported views belong to nine source-fit groups in three localities.
They include duplicate heads and dependent seed views, not 15 independent
successful models. The three trained views are the final/MSE duplicate at
locality008 seed17, step2000, and locality119 seed43 MSE step200. The corresponding
initial priors have higher known-source-validation gain. Thus this diagnostic
does **not** establish an advantage of the learned head over its prior.

Example: locality008 seed17 MSE/final selects 17,269 validation occurrences,
including 177 wholly unknown outcomes. Unknown envelope mass is 293.75146;
known easy budget slack is 352.74968, so it can cover that worst-case mass.
Selected all/easy harm upper ratios are 0.98096% / 1.77911%, within the unchanged
2% budget. Net gain lower mass is 5,208.54860 in summed image-local ADE units.
This is not a guaranteed percentage gain or a transfer-safety statement.

## Evidence Boundary

The 99,498 unique source-validation row IDs include 37,668 with nonempty partial
future masks. Known native partial-label costs stay fixed. The proof applies
to arbitrary nonempty labels on wholly unknown rows, using maximum causal
forecast disagreement. It does not impute trajectories, modify inference
eligibility or replace partial-label ADE with full-grid ADE.

The run independently checks 1,512 aggregate quantities with scalar summation,
reconstructs all old source choices and verifies causal envelopes. Initial
runtime: 58.92 seconds; peak resident memory: 8,352,546,816 bytes. No new row
cache or checkpoint. Exact replay and scoped-test receipts are stored alongside
this report. No statistical CI is attached to an algebraic bound: these are
deterministic conditional development calculations, not population estimates.

The 12 localities remain exposed development sources. Source validation was
already used for checkpoint selection and is not independent calibration.
No deployment upgrade; independent selection/calibration/confirmation closed.
Stage5C and SMC off; no metric, seconds, true3D, foundation or human-gold claim.

## Next Contrast

Register a separate comparison that changes only the source-level unknown-label
veto to this finite-completion screen, retains the existing candidate ordering
and known-validation utility ranking, and freezes actions before transfer
readout. This will test whether recovered source support has decision value;
it must not be presented as new neural learning.
