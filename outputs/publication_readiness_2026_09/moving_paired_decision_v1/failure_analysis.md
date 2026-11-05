# Failure Analysis: Ordering Is Not Safe Acceptance

## Result

The registered paired-rank repair fails its matched utility comparison and
the original full-policy safety requirement. No deployment changes. All nine
real fits and frozen readouts are complete and exactly replayed; this is not
an unrun experiment or a runtime failure. The original rules remain unchanged.

| Policy | Known ADE gain % | Hard gain % | Worst easy-net % | Original full risk/support |
|---|---:|---:|---:|---:|
| Joint control | 0.894772 | 1.169812 | 8.633364 | 0/17 |
| Matched pointwise | 1.467386 | 1.840246 | 8.464957 | 0/17 |
| Paired ranking | 0.836321 | 1.129416 | 8.337269 | 0/17 |

These are source-macro partial-outcome numbers on seven already exposed
development sources, not independent test estimates or complete-roster ADE.
The main rank-minus-point contrast is -0.631065 percentage points, with
missing-label completion bounds [-0.866418,-0.395712] pp. They are not sampling
confidence intervals. One producer seed does not establish multi-seed stability.

## Failure Taxonomy

### 1. Relative Ranking Does Not Fix The Acceptance Boundary

Rank improves own-fitting combined AUROC over joint in all three contexts:
0.9446 to0.9586,0.8641 to0.8765,0.7200 to0.7281. Yet its total fitting decision
cost is46.5534%,19.6743%,1.7278% worse than point. Ordering and the fixed
zero-boundary decision are different targets. Rank's common-score translation
invariance is verified analytically and by synthetic tests; it does not prove
that a particular learned common shift caused every observed error. Original
supervised anchors remain present. See [loss geometry](loss_geometry.md).

### 2. More Rejection And More Intervention Have Different Costs

In critic0, rank reduces harmful admissions from18 to4 versus joint but raises
useful rejections from2655 to2873. Total fitting decision cost rises8.2240%.
In critic1/2, rank lowers total cost slightly versus joint, by3.6430%/0.3400%,
but is still inferior to point. The effect is not uniform across contexts.

Point reduces useful rejection in every context, but harmful admission counts
rise18/35/162 to19/118/551. Corresponding weighted false-safe cost increases
65.66%/251.51%/207.32%. Its larger average development utility is therefore
not a safe upgrade. Counts alone cannot replace weighted cost or the original
per-recording risk/easy/support checks.

### 3. Heterogeneous Fitting Difficulty Remains

The largest context has rank AUROC0.7281 versus0.9586 in critic0. Its risk-MSE
trace is also strongly query-dependent. This identifies heterogeneous fitting
difficulty, not proof of missing data or a unique feature failure. Existing
two-agent pooling alias evidence covers only five queries and cannot explain
the overall result. Future-feature, test-goal or threshold shortcuts are not
remedies and remain prohibited.

### 4. The Negative Difference Is Not Only One Source

Six of seven exposed sources lose utility for rank versus point. SOURCE078
has the largest contrast, -3.3202pp; SOURCE051 is -0.7724pp. SOURCE077 improves
only0.00765pp. These are repeated-development slice descriptions, not a new
scene-selection rule or justification to remove difficult sources.

### 5. Actual Violations Remain In Addition To Missing Support

Rank has11 known all-risk violations,13 easy-risk violations and4 undefined
selected-easy views. Its worst defined easy-net degradation is8.3373%, above2%.
Thus this is not merely a denominator or empty-control problem. Removing the
decision veto increases known gain to3.3212%, but worsens easy-net degradation
to41.7732%. More permissive switching is not a valid repair. Missing support
stays NA, not pass; no identity-fallback certification is added.

### 6. Engineering Reproducibility Is Not Scientific Confirmation

The first process terminated without an observed exit code. Same-binding
resume completed; all three fresh controls exactly reproduce prior models,
the full treatment refit is exact and all new metrics/predictions replay.
This rules out nondeterministic restart drift in the checked comparison, not
all possible implementation defects. Reserved20 Moving, DUT, DroneCrowd and H
remain closed. The original24-group primary remains NA.

## Next Model Question

Before another fit, inspect own-B score magnitudes and the gradients of the
paired term versus supervised risk/decision anchors, particularly useful rows
that remain blocked. Distinguish a conditional magnitude error from loss-gradient
conflict; neither mechanism is established by the current aggregate results.
Check earlier objective controls before registering one distinct repair. Keep
the zero threshold, original2% rules, data roles and independent-data boundary.
Do not retrospectively promote point, sweep thresholds or relabel old failures.

This experiment trains intervention-risk heads, not new forecasts or a JEPA
world model. Native obs8/pred12 image coordinates are not metric, seconds,
true3D, foundation or human gold. Stage5C and SMC remain disabled. The broader
research goal remains incomplete, including an independently confirmed safe
method, three-seed primary evidence and a submission-ready package.
