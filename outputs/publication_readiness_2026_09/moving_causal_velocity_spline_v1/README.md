# A causal velocity spline versus OLS4

I trained a direct forecaster rather than a new selector. It reads eight
positions, predicts three bounded velocity controls in a past-motion
frame, and integrates them over twelve annotation steps. It starts
exactly at OLS4. The ego and peer arms share the same training draws
and architecture. The completion arm also charges disagreement with
OLS4 at each missing future step. The network never receives the
future target or the availability mask as input.

The run trained 27 PyTorch models, totaling 43,200 updates, in three
source-role rotations and three seeds. The full roster has 69,746
windows from 24 recordings in nine exposed development source groups.
The fixed CV, OLS4 and OLS8 controls and the verified source-excluded
EqMotion and Past Context Transformer forecasts use the same held
rows. The frozen comparators used six fitting sources per fold versus
three for the spline, so this is not a matched training-budget comparison.
They are local-protocol comparisons, not reproductions of
[Scholler et al.](https://arxiv.org/abs/1903.07933) or [EqMotion](https://arxiv.org/abs/2303.10876).

| Forecast | Available ADE / endpoint FDE (px) | Observed gain vs. OLS4 (pp) | Conservative gain vs. OLS4 (pp) |
| --- | ---: | ---: | ---: |
| CV | 94.11 / 246.62 | Reference for normalization | Reference for risk |
| OLS4 | 91.60 / 234.45 | 0.00 | 0.00 |
| OLS8 | 101.68 / 242.37 | -15.52 | -59.66 |
| EqMotion, fixed head | 86.10 / 207.58 | +3.51 | -41.84 |
| Past Context Transformer | 81.44 / 192.05 | +6.74 | -120.07 |
| Ego spline | 87.15 / 217.63 | +2.54 | -55.36 |
| Peer spline | 91.23 / 216.06 | +2.54 | -57.44 |
| Completion spline | 86.09 / 222.66 | +4.21 | -9.12 |
| Validation-selected spline | 92.14 / 230.33 | +4.89 | -39.98 |

The gain columns pool valid step error with equal query weights and
normalize by observed CV error, not OLS4 error. The ADE column first
averages within a trajectory, so its sign need not match observed
gain. Endpoint FDE is conditional on the last label being available.
Intervals use 3,000 paired source draws after averaging seeds.

The completion arm has observed gain **+4.21 pp [0.93, 7.64]**, but a
conservative gain of **-9.12 pp [-16.99, -3.89]**. The selected arm's
observed interval crosses zero **[-6.38, 16.10]**, and its conservative
interval is **[-67.66, -18.14]**. The loss of a positive lower bound is
not evidence that missing trajectories got worse: the debit is
unobserved uncertainty. For the completion arm, partially labeled
tracks contribute 11.35 pp of debit and wholly unobserved tracks 1.98 pp.
Its complete-track observed gain contribution is not clearly positive.

With the same observed loss, peers change observed gain by +0.001 pp
[-10.69, 10.39], and the validation-selected spline worsens observed
pair displacement by 6.08 px [2.05, 13.02] versus OLS4. The completion
arm's known hard-case improvement versus OLS4 is +1.74% [-2.44, 5.05].
The direct forecasts pass no full-recording original risk checks.
These are diagnostics, not a new deployment protocol.

## Candidate capacity and the missing-tail repair

At the previous 21,242 exact query edit counts, adding the three frozen
forecasts at amplitudes .15, .35 and 1.0 leaves discovery at **10/72**
verified recording-seed views. The original and augmented banks have
the same statuses, including 36 nonpositive upper bounds and two
unresolved views. Learned paths supply 699 of the actions in the
verified witnesses, but no new view verifies. This outcome-aware search
does not show that a causal selector can recover the witnesses.

I then tested whether concentrating the same total per-row displacement
earlier in the forecast could address the missing-step debit. Returning
gradually to OLS4 improves the absolute conservative score against
OLS4 from -9.12 to -2.35 pp, versus -3.30 pp for a uniform shrinkage
control with identical displacement. The direct paired conservative
contrast of taper versus uniform is -3.68 pp [-6.00, -1.97]; it uses
its own missing-step debit, so it is not the subtraction of two OLS4
lower bounds. Six calibration cells choose a taper, three retain OLS4.
The validation-frozen held gain is -0.04 pp [-0.35, 0.34] against
OLS4, and no recording passes the original CV-relative rule. This is
an exploratory repair after the raw development readout, not an
independent result.

I keep the existing fallback. The next priority is to compare
stronger candidate forecasts on source-separated development tracks
with adequate label support before another selector fit. The exposed
complete-track contribution in this study is not itself decisive,
and the original rule and reserved confirmation roles stay unchanged.

Training and readout are `fresh_run`; OOF comparator trajectories,
source roles and the checkpoint replay are `cached_verified`. The
ARM CPU run finished in 91 seconds with four Torch threads, no workers,
and about 1.44 GiB peak RSS. Checkpoints preserve model, optimizer and
draw state every 100 updates. No external calibration, independent
confirmation, latent rollout or SMC was run.

[Protocol](protocol.md) | [Direct readout](readout.json) |
[Error decomposition](slices.json) |
[Matched candidate capacity](capacity.json) | [Tail repair](tail_readout.json)
