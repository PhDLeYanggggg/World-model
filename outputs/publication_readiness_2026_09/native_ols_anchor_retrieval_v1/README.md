# Selecting an OLS4 position correction

I tested whether a short backtest within the observed history can identify
which agent would benefit from an OLS4 position correction. Each eligible
agent has the same kind of alternative: a fixed half-amplitude position
ramp over the first three predicted steps, with the OLS4 slope unchanged.
Each selection method intervenes on exactly one agent per eligible query.
The learned models use the same training draws; the motion control has the
five prefix backtest features zeroed. The oracle sees outcomes and only
measures hindsight headroom. See the [registered protocol](protocol.md).

## Results

Fresh run: 54 native Torch fits, three seeds (17, 29, 43), nine
leave-one-fitting-source-out folds, 69,746 windows, and 1,176 queries.
Each intervention arm makes 3,528 selections summed across seeds.
The intervals resample the nine fitting sources 3,000 times after seed
averaging. Gains and harm are normalized by observed CV error mass; they
are not percent changes in FDE or measurements of the official 2% risk.
The lower score debits wholly unknown labels by maximum trajectory
disagreement and uses available labels for other rows. It is not a
full-grid bound for partially observed futures.

| Selection | Lower vs OLS4 [95% CI], points | Positive harm, points | Benefit-label AP |
|---|---:|---:|---:|
| OLS4, no switch | 0 | 0 | N/A |
| Learned with prefix backtests | -0.00804 [-0.01400, -0.00230] | 0.02610 | 0.394 |
| Learned motion control | -0.00897 [-0.01495, -0.00313] | 0.02378 | 0.349 |
| Untrained prefix rule | -0.01439 [-0.02488, -0.00647] | 0.03025 | 0.331 |
| Random | -0.01638 [-0.02578, -0.00848] | 0.03206 | 0.336 |
| Same-budget hindsight oracle | +0.05043 [+0.02231, +0.08405] | 0.00393 | 0.992 |

The added features improved descriptive ranking AP, but did not establish
a paired gain over the matched motion control: the conservative difference
is -0.00197 points (95% CI [-0.00489, +0.00099]). The learned probe chose
1,387 switches with strictly positive gain and 109 with wholly unknown
labels, compared with 3,213 and zero for the oracle. The fixed zero-score
abstention diagnostic also does not establish a gain (+0.00065 points,
95% CI [-0.00556, +0.00979]). The oracle result measures this one candidate
and this switch budget, not the earlier rowwise amplitude sweep.

I would stop tuning this fixed anchor correction. It is difficult to
identify the right agent from the observed prefix, and even perfect
selection would recover little at this budget. A stronger next comparison
would replace the forecast candidates themselves and hold the selection
budget and data split fixed.

## Reproduction and limits

Run from the repository root with the private fitting assets available:

```sh
.venv-pytorch/bin/python scripts/run_m3w_ols_anchor_retrieval.py train
.venv-pytorch/bin/python scripts/run_m3w_ols_anchor_retrieval.py score
.venv-pytorch/bin/python scripts/run_m3w_ols_anchor_retrieval.py replay
```

The targeted regression suite passed eight tests. Frozen scores and row
IDs were checked before evaluation. Scoring and replay produced identical
[analysis JSON](analysis.json), SHA-256
`1b0653fb16879ba456b7f1ee6a1a51a6e1aacee116164e2efcd2830ee80efde0`.
The weights and per-row arrays remain local.

This is an eight-observation, twelve-prediction experiment at native
annotation steps in image coordinates. It does not establish time or metric
calibration, true 3D, a foundation model, or submission readiness. No
development or external outcomes were used. The original 2% rule and
DroneCrowd 15-source calibration / 30-source independent confirmation split
are unchanged; confirmation remains sealed. No deployment, Stage5C or SMC.
