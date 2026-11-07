# Directional ADE Scoring Does Not Reliably Improve Matched Utility

Fresh development fit on exposed Moving A: 14 sources, 34 recordings,
102 views, three previously frozen forecaster seeds. Use 8 observed and
12 future annotation steps in native image coordinates. In each of 42
cells, only seven fitting sources enter the direction and remainder
estimates. The seven other sources, including the held source, do not.
The target and valid-label mask enter training and scoring only, never
inference. Candidate trajectories, estimated risk costs, missingness
envelope and the original risk rule are unchanged.

The new head predicts which direction would reduce baseline ADE, then
scores the displacement of each candidate along that direction. An
additional candidate-specific nonnegative term accounts for the
remainder between this linear approximation and exact ADE gain. Neither
head emits or modifies a forecast. The original direct-gain policy and
metrics replay exactly. All arms have 6,952 switches matched per query.

| Comparison | Known ADE gain difference (pp) | Completion lower (pp) | 95% source interval for lower |
| --- | ---: | ---: | ---: |
| Direction plus remainder vs direct (primary) | +0.01879 | +0.00916 | [-0.04245, +0.06149] |
| Direction only vs direct | +0.01750 | +0.00565 | [-0.04978, +0.06339] |
| Direction plus remainder vs direction only | +0.00129 | -0.00707 | [-0.01620, +0.00186] |

| Policy | Full risk passes / 102 | Easy upper >2% | Signed easy net >2% | Undefined easy risk |
| --- | ---: | ---: | ---: | ---: |
| Direct | 2 | 65 | 8 | 35 |
| Direction only | 2 | 65 | 10 | 35 |
| Direction plus remainder | 4 | 63 | 7 | 35 |

The full head changes 8,988 row assignments (including switches moved
between rows) relative to direct. This is not a nearly identical
allocation, but neither the primary nor the remainder ablation supports
a reliable utility improvement. Source bootstraps (3,000 draws after
averaging forecaster seeds) reflect paired development uncertainty.
Still only 4/102 views pass the complete observed rule. No independent
confirmation, deployment change, Stage5C execution or SMC.

Score and replay have the same analysis hash. The comparison is included
in the [working manuscript appendix](../../../evidence_manuscript_v10/utility_tradeoff_appendix.md),
with the calibration and source-stable risk ablations in separate rows.
