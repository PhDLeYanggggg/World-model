# Matched Interventions and Nonadditive Pair Gain

This note defines the experimental target; it does not claim that
the trained policy passes calibration.

## Decision

Keep the causal CV forecast as the default. At a fixed query
with two available edit slots, an assignment chooses two distinct
recipients and one legal candidate path for each. All comparison
arms use the same past-only candidate menu and two peak offsets.
The independent arm sums per-action values, the scene-uniform
arm uses the same values and requires a common path family, and
the slate arm uses the estimated probability of passing the
unchanged dual-accuracy and risk checks.

The slate head estimates the positive smaller of the trajectory
gains against CV and OLS4 conditional on qualification. Its
risk-only score is qualification probability times that gain.
The interaction ablation uses the same checkpoint, probability
and trajectory term, adding the conditional nonadditive pair
gain divided by the roster's pair count. This is a ranking
objective, not a bound on realized risk. Risk is measured with
the original outcome-based checks on the separate calibration
city. In the optional two-arm comparison, both policies edit the
same queries at the same peaks.

## Four-Path Contrast

Let `E_0`, `E_A`, `E_B` and `E_AB` be pair-relative distance
errors with neither, either or both changes. The additional
gain of making both changes, beyond the sum of singleton gains,
is

```text
E_A + E_B - E_AB - E_0.
```

Terms from every pair other than the two edited agents cancel.
For that remaining pair, write its reference residual as `z`
and the edited agents' displacements as `u` and `v`.
The contrast is

```text
||z + u|| + ||z - v|| - ||z + u - v|| - ||z||.
```

Group by either `u` or `v`. The reverse triangle inequality
bounds each of the two corresponding norm differences by the
same edit norm. The magnitude of the contrast is therefore at
most `2 min(||u||, ||v||)`, independent of the unknown target.
On missing pair steps subtract this quantity; average known
terms and missing debits over the original full step grid. The
same target completion must underlie all four terms; subtracting
separately bounded forecasts would not give this contrast.
This is about forecast error, not an effect of intervening on
the physical agents or a collision guarantee.

## Relation to Prior Methods

[Learning to defer](https://proceedings.mlr.press/v119/mozannar20b.html)
motivates learning relative to a reference rather than treating
model accuracy as the entire objective. Here the reference is a
motion forecast and the actions are coupled edits, rather than
an instance-level classification handoff.
[ScePT](https://arxiv.org/abs/2206.13387) addresses consistent
joint forecast generation. This experiment holds the predictor
and edit budget fixed to test the incremental value of
joint selection, including when trajectories have incomplete
future annotations. The four-term difference uses the familiar
inclusion-exclusion construction; the contribution under test
is its use for matched forecast edits with incomplete labels,
not the algebra alone.
