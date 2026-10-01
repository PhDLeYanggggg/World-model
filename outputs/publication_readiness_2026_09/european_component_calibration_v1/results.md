# Source-Only Component Calibration: Negative Matched-Utility Result

## Evidence Status

Completed on 12 already-exposed European development localities. Source-only
calibration, action generation and outcome readout are **fresh_run**, followed
by exact replay. The 72 fixed parent cost heads and upstream predictors are
**cached_verified**. No new forest or neural dynamics model was trained here.
Independent selection, calibration and confirmation roles remain closed.

The task is obs8/pred12 at stride12 raw frames in image-local coordinates with
detector-silver labels. These are not metric, seconds-level, true3D, foundation,
human-gold or physical-safety results. Stage5C execution and SMC remain off.

## Frozen Comparison

The design retained raw and source-screened parent policies, harm-only,
reference-only and joint adjustments, their source-screened versions,
per-query count-matched parent/joint controls, and the fixed floor. All three
head seeds (17, 29, 43) share upstream43; no seed or arm was chosen on transfer.

Registration was committed at `85f6a3ac`, the source calibration freeze at
`22162228`, and 216 causal action views at `0fbc3c60` before outcome readout.
No per-row future-label availability or future endpoint enters inference.
Source screens include finite-completion bounds for unknown selected outcomes.

Empirical 90th-percentile recording margins raise harm estimates and lower
reference-error estimates. Whole-recording OOF screens exclude each held
recording from its own coefficient fit. Full-fit coefficients use all source
validation recordings. The OOF and final policies differ; OOF support is not
a safety guarantee for the final policy. The original 2% selected-reference
risk budget is unchanged. This procedure is not CQR or conformal risk control.

## Primary Result

At the same per-query intervention counts, joint calibration has **-0.0030862%**
ADE improvement over the parent, nominal locality-bootstrap 95% interval
**[-0.0056737%, -0.0010450%]**. There are 3,000 draws over 12 equally weighted
locality means: nine negative, two positive, one zero. Head/direction repetitions
and overlapping windows are not independent samples. Because these localities
have already been used for development, this interval is not confirmation.

| Policy | ADE gain vs floor (%) | Hard ADE gain (%) | FDE gain (%) | Intervention (%) | Easy selected-risk violations / defined | Undefined |
|---|---:|---:|---:|---:|---:|---:|
| Raw parent | 0.052592 | 0.000416 | 0.077881 | 4.778357 | 27/140 | 76 |
| Source-screened parent | 0.047605 | 0.000398 | 0.070749 | 4.246264 | 18/90 | 126 |
| Harm-only, source-screened | 0.047542 | 0.000398 | 0.070649 | 4.197606 | 10/71 | 145 |
| Reference-only, source-screened | 0.034204 | 0.000275 | 0.050758 | 3.188412 | 6/60 | 156 |
| Joint, source-screened | 0.034200 | 0.000275 | 0.050750 | 3.188115 | 6/60 | 156 |
| Parent, matched count | 0.037282 | 0.000284 | 0.055290 | 3.188115 | 6/60 | 156 |
| Joint, matched count | 0.034200 | 0.000275 | 0.050750 | 3.188115 | 6/60 | 156 |
| Floor | 0 | 0 | 0 | 0 | 0/0 | 216 |

Gains are relative to the same fixed floor, not Stage37's historical task.
Detailed fixed-arm results are in `summary.json` and `readout.json`.
Floor-only and zero-denominator views have undefined selected risk, not passes.

## Risk and Failure Mechanism

Joint source screening has worst easy selected positive-harm/reference ratio
**2.278217%**, above the 2% budget. All-agent selected risk fails 2/60 defined
views, worst 2.104243%. There are 1,940 unknown selected occurrences across
repeated views; these are not zero-error outcomes or distinct independent rows.
No head seed passes: easy-risk violations are 2/21, 1/18 and 3/21 for 17/29/43.

Worst whole-easy degradation is only 0.003339%, but does not certify selected
positive-harm risk. Fewer violations coincide with reduced coverage and more
undefined views. The matched-count parent also fails 6/60, so the reduction
does not establish better risk ranking. Joint is essentially indistinguishable
from reference-only in this fixed comparison, without an incremental harm-arm
benefit established.

For source008 -> target020, single1/controller0, head43, switching less raises
actual selected easy risk from 1.649600% to 2.104243%. Removing low-harm reference
mass can increase the ratio on the retained group. Conservatively changing
predicted components is not monotone control of actual selected risk.
See [the failure analysis](failure_analysis.md) for all six failing directions.

## Source Support and Runtime

- 72 calibrators; 43 supported and 29 with no known raw-eligible source rows.
- 348 leave-one-recording-out folds over 58 distinct validation recordings;
  121 folds lack component support.
- OOF source screens pass 26 harm-only, 23 reference-only and 23 joint fits.
  Full-fit resubstitution counts 30/27/26 are diagnostic, not admission criteria.
- Eleven full fits have only one contributing recording for their least
  supported component, and 17 have two. A coefficient is not a reliable bound.
- Full fit 54.28 s; full calibration replay 54.76 s; readout 138.02 s;
  complete readout replay 136.47 s. Coefficient records total 947,602 bytes.
- Real pilot peak RSS 6,090,293,248 bytes. Native arm64, CPU4/interOp1/workers0;
  no new CREATE job or large trajectory materialization was needed.

## Verification and Decision

All 72 calibrators and the entire outcome readout replay exactly. The run has
59,400 native metric checks, 747,900 query-count checks and 648 exact unchanged
parent metric views. The 31 scoped tests passed; the full legacy suite is
**not_run**, not implied by scoped verification. Numeric verification seal:
`2ea6c547ade4668c8d4f3f8750a13e68e168cf76d53ef0e509abb9bb7ca6f7d4`.

**Do not promote this policy.** Keep the deployment floor unchanged. The study
rules out this specific marginal component correction as a safe utility upgrade;
it does not establish the impossibility of risk-aware neural intervention.

Next, reuse prior selected-subset and scene-joint assets and register source-OOF
versus transfer accounting of retained/removed benefit, positive harm and
reference mass. Separate post-selection shift within source recordings from
cross-locality drift before another policy fit. That diagnostic is **not_run**.
Do not tune on the six transfer failures, relax the risk budget, or open
independent roles. [Reproduction and resume instructions](operation.md).
