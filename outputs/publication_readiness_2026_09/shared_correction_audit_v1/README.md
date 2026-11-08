# Does Shared Motion Explain the Hindcast Risk Reversal?

**Fresh diagnostic.** Keep the same three-origin half-blend recipients
and the original 2% rule. For each recipient, subtract the mean
OLS4-minus-CV velocity among the other complete-history agents in that
query, then cap the adjusted vector at the original norm. The control
keeps the original direction and uses that same norm. Singletons use
the original correction. The common term is a proxy for shared
image-plane motion, not recovered camera pose.

The two new arms preserve intervention counts in both cohorts and
match every velocity-correction norm to floating-point tolerance
(maximum difference 4.45e-16 on the opened recordings). The 40
opened-recording forecasts were hashed before scoring; the original
hindcast scores and risk counts reproduce. Fitting uses the same
69,746 rows in 24 recordings and nine groups, with original query
memberships reconstructed for the motion proxy. The opened cohort has
40 recordings in 15 groups. There are no new learned parameters.

| Policy | Fitting passes / 24 | Opened passes / 40 | Opened conservative gain vs CV, pp |
|---|---:|---:|---:|
| Frozen hindcast | 0 | 40 | +26.477 |
| Reduce magnitude only | 0 | 40 | +25.052 |
| Remove shared correction | 0 | 39 | +24.592 |
| Full OLS4 | 0 | 8 | +51.288 |

The primary direct conservative contrast, shared removal minus
magnitude only, is **-0.464 pp [-0.661, -0.293]** on the opened
cohort and **-1.051 pp [-2.960, +0.232]** on fitting sources.
Intervals are 3,000 paired source-group resamples within each cohort.
The repaired direction does not improve upon shrinkage, and even
the shrinkage does not repair the fitting-source risk checks.

## Why the reversal is not just a mix of motion bins

The positive-harm-to-CV-reference ratio, pooled on known selected
rows without the missing-label debit, is 8.057% on fitting sources
and 0.309% on opened DroneCrowd. This pooled diagnostic is **not**
the official per-recording pass criterion.

Query coherence is the squared norm of the mean OLS4-minus-CV
velocity divided by the mean squared norm. Below 0.25, the known
ratios are 4.519% versus 0.281%; between 0.25 and 0.75 they are
5.620% versus 0.988%. The high-coherence bin has 4,959 fitting
recipients and no opened recipients. The two overlapping bins cover
47.05% of the fitting reference mass and all opened reference mass.
Within this common support, the -4.663 pp gap decomposes exactly
into **-4.328 pp within bins** and **-0.336 pp from different
reference weights** (rounded). The easy overlap covers 96.81% of
fitting reference mass; its -13.405 pp gap is likewise mostly within
bins (-12.281 pp). The split uses average cohort weights/rates, so the
two terms sum to the observed gap. It is descriptive, not causal;
it cannot compare the unrepresented high-coherence bin.

**Decision:** Do not deploy shared removal or spend sealed confirmation
on it. Test sensitivity to past sampling and observation stability on
identical anchors and future labels before another scene-level model
change. The [MovingDroneCrowd++ paper](https://arxiv.org/abs/2601.12500)
and [DroneCrowd paper](https://arxiv.org/abs/2105.02440) describe
different source releases; the current result does not establish
matched physical cadence, camera pose, or sensor-time annotation
provenance. The task remains offline annotated image coordinates,
eight observed and twelve predicted annotation steps. No metric,
seconds-level, physical-safety, true-3D or foundation claim.

~~~sh
p=scripts/run_m3w_shared_correction_audit.py
for phase in register freeze score replay; do .venv-pytorch/bin/python "$p" "$phase"; done
.venv-pytorch/bin/python scripts/audit_m3w_shared_risk_mixture.py
~~~

Native arm64, local per-recording receipts and heartbeat. Exact replay
uses only scalar artifacts, without reopening targets. Eighteen scoped
tests passed. Original risk rules and confirmation roles are unchanged;
all 30 independent groups remain sealed. Stage5C and SMC are off.
[Registration](registration.json) | [Analysis](analysis.json) |
[Risk decomposition](risk_mixture.json).
