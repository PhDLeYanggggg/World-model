# Query-Head Conditioning: Complete, No Deployment Change

Result source: fresh_run frozen-head fitting inference/autograd and disposable
optimizer counterfactuals; parent36 checkpoints cached_verified. All72 inner-fit
views,36 head groups,288 matched batches and576 disposable AdamW steps completed.
PID65213 exit0,141.240318s, peak RSS2,947,530,752bytes. Native arm64 CPU4/inter-op1,
workers0. No persistent model training, held inference or independent role opened.

## Identical-Cohort Risk Estimation

The primary cohort is the frozen self policy's selection. Both heads see exactly
62,902 repeated occurrences:61,335 with known outcomes and1,567 unknown. These
are not independent agents, windows, scenes or a held test sample. Each mass
below sums errors divided by its original source's frozen TRAIN scale; these
are neither pooled ADE/FDE nor physical units.

| Measure on fixed self cohort | Self head | Peer-context head |
|---|---:|---:|
| Observed all-harm mass |4979.711114|4979.711114|
| Predicted all-harm mass |220.525318|14089.895730|
| All signed-risk residual MSE |0.164117|0.437189|
| Observed easy-harm mass |95.560642|95.560642|
| Predicted easy-harm mass |1.375607|23.342870|
| Easy signed-risk residual MSE |0.000684616|0.000728945|
| Predicted easy excess over2% reference |−28.909064|−6.941800|
| Observed easy excess |45.286108|45.286108|
| Near-zero easy output/logit Jacobian |47,239/62,902 (75.10%)|42,741/62,902 (67.95%)|
| Positive easy underprediction at near-zero Jacobian |87.76%|78.67%|
| Exact-zero easy Jacobian |259|231|

Unknown outcomes remain in inference/context/derivative counts; residual MSE
uses only the61,335 known occurrences. Near-zero is L1 Jacobian <=1e-6 times
the causal harm cap, unchanged from the previous audit. Signed easy excess is
H_easy−.02R_easy, not net easy ADE degradation. Reference estimates can also
contribute to signed error; this is not a pure harm-only residual.

Peer context raises estimated harm mass, but does not solve the same rows'
prediction problem. All-harm mass overshoots while2,766.756 units of positive
underprediction remain; easy squared residual is worse. On peer's own selected
cohort, predicted/observed easy harm is only0.006065, with78.82% near-zero rows.
Thus improving aggregate mass is not sufficient for conditional decision quality.
This problem already occurs on fitting data, not only under held-recording shift.

## Does Removing Clipping Fix It?

Every registered probe activates clip5. Median gradient scaling is0.05925 for
self and0.07896 for peer. Median weighted-risk/data gradient-norm ratio is9.79
and6.66; negative cosine occurs72/144 and70/144 times. These final-state batches
are not the complete training path.

Both disposable updates inherit the exact same saved AdamW moments and step
counter; the original parameters and checkpoints never change.

| Unclipped versus original clipping | Self: better / worse / tied | Peer: better / worse / tied |
|---|---:|---:|
| Total same-batch objective |120 /24 /0|129 /15 /0|
| Fixed-cohort all signed-risk MSE |70 /74 /0|86 /58 /0|
| Fixed-cohort easy signed-risk MSE |30 /35 /79|52 /39 /53|

Median parameter-step norms increase0.02181→0.09594 (self) and0.02392→0.09077
(peer). Median total loss favors unclipped by0.32303 and0.27518, but worst
unclipped-minus-clipped total differences are+256.47 and+144.15. Easy-risk
changes are mostly tied or mixed; the tolerance is max(1e-10,1e-6|before|).

**Conclusion:** clipping materially limits these local steps and removing it
often lowers total loss. This is not evidence of a reliable easy-risk repair,
longer-run stability, held lift or deployment safety. Do not promote an
unclipped model or choose a new threshold from these diagnostics.

## Verification and Decision

The first attempt stopped at a float32 additivity assertion. Its numerical
reproduction and all partial receipts are preserved. v2 repairs verification,
not any model/loss/optimizer setting. Native gradient decomposition error is
at most4.90e-7 relative to component magnitude; double-precision checks are
at most1.86e-15. All12 old fit views and4 old probe groups reproduce exactly,
excluding only version binding and the new verification field.

Separate arithmetic checks cover1,152 view/target records,272 summary fields
and288 finite probe batches.30 scoped tests pass2.83s. The full legacy suite
was not run because it contains historical-artifact-writing tests. The local
four-panel figure was visually inspected; only its receipt is committed.

No new candidate is deployed. The earlier72-view context comparison remains
negative; this diagnosis does not overturn it. See [failure analysis](failure_analysis.md),
[next-step boundary](project_gap.md) and [verification](verification.json).

Obs8/pred12 rawstride12, detector-silver pixel/dataset-local evidence only.
No verified metric/seconds/physical-safety/true3D/foundation claim. Stage5C and
SMC remain disabled. This is progress in diagnosis, not submission readiness.
