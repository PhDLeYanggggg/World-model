# One Versus Two Edits, and Independent Versus Joint Choice

The previous one-edit result left a concrete question: can one edit
supply the easy reference required by the recording risk check while
a second edit provides useful forecast gain? I kept the original
OpenDD development cohort, CV and OLS4, the finite candidate bank,
eight observed and twelve requested steps, and the original 2% risk
and missing-label rules.

[M2I](https://openaccess.thecvf.com/content/CVPR2022/html/Sun_M2I_From_Factored_Marginal_Trajectory_Prediction_to_Interactive_Prediction_CVPR_2022_paper.html)
models interactions by combining marginal and conditional predictions.
Here the path bank is frozen; the matched comparison isolates which
two baseline paths to replace under the same intervention budget.
This is not a benchmark comparison against M2I or a test of joint
distribution modeling.

## Exact Capacity on the Credit-Starved Recordings

I enumerate legal one- and two-edit selections over the frozen
prefix-selected paths, including different windows when available.
Each edit uses an eligible row and one of the three registered
amplitudes. Two edits cannot target the same row; a query must retain
at least one unedited row. The oracle lifts only the original
chronological token constraint and checks strict conservative gain
over both baselines. A witness for each reported positive category is
rescored with the original evaluator.

| Hindsight condition | Views out of 105 |
| --- | ---: |
| At most one edit | 30 |
| At most two edits | 36 |
| Two in one query | 6 |
| Two across different queries | 30 |
| Two with an easy row and a hard row | 12 |
| Two with neither constituent qualifying alone | 12 |

These categories may overlap. Only six seed views (two distinct
recordings) are added by the second edit; none of these six has
an easy-plus-hard witness under this bank and the training-derived
hard cut. The earlier arbitrary one-edit geometry ceiling of 57
is higher than this two-edit finite-bank ceiling. Thus this test
does not establish that more edits are a substitute for better
candidate trajectories.

## Causal Comparison at the Same Budget

At the first eligible query with at least three agents and two legal
recipients, every policy makes exactly two nonzero edits. The policy
does not know whether the recording is short. All three policies share
the same candidate value head, which predicts both conservative gain
increments and CV-relative risk moments. Each fold fits only on the
other six locations; donor labels use easy cuts excluding both the
held and donor locations. None of the three scores is calibrated.

- **Independent:** sum the two individually scored actions.
- **Joint:** apply the two-reference minimum and risk penalties after
  summing predicted moments, with a soft easy-coverage bonus.
- **Joint plus geometry:** also penalize decreases in predicted
  minimum pair separation versus CV. This is an image or coordinate
  geometry proxy, not a physical collision or safety claim.

| Policy | Edits | Risk passes | Undefined easy | Risk pass and beat CV and OLS4 | Direct conservative gain vs independent (pp; 95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent | 744 | 64 | 432 | 48/606 | reference |
| Joint | 744 | 57 | 435 | 44/606 | +0.0045 [-0.0109, +0.0277] |
| Joint plus geometry | 744 | 57 | 434 | 44/606 | +0.0042 [-0.0111, +0.0274] |

This is a matched-count causal probe, not a replacement for the
streaming policy. There are 606 views of 202 recordings. Exactly 372
views trigger, including 12 of the 105 credit-starved views; each
policy passes just two joint checks in that short subset. The
conservative paired gain averages three seeds before a 3,000-draw
location bootstrap. Row-pooled ADE/FDE is 0.5560/1.2270 to four
decimals for all arms in dataset-local units, with at most 4.876%
worst known easy net degradation across recording views for each arm.
The geometry version does not improve the held-query relative path
diagnostic: conservative pooled gain versus independent is -0.100 pp
with five supported locations (joint without the penalty: -0.077 pp).
The diagnostic uses all within-query pairs and debits unobserved pair
steps. Its 18,702 observed pair-step seed views are not independent.

No policy passes the original full-recording gate. In particular the
result does not show that joint selection improves forecasting at a
matched intervention count, or that the penalty improves consistency.
The much larger across-query than same-query capacity suggests one
next check: test a causal schedule with the same total two-edit budget,
without committing to a learned joint advantage. Retain the old
operating policy and leave the 30 DroneCrowd groups sealed.

## Reproduce

```sh
# Fit only if the 21 registered checkpoints are absent.
.venv-pytorch/bin/python scripts/replay_m3w_two_edit.py --fit
# Score-only; refuses any unfinished checkpoint.
.venv-pytorch/bin/python scripts/replay_m3w_two_edit.py
```

21 fresh native arm64 Torch fits, 8,400 updates, four compute threads,
no data workers, checkpoint and heartbeat every 100 updates.
The first run completed every fit and every cell readout, then
stopped on an empty short-roster cell at aggregation. The repair
changes only that cell's representation and reuses the identical
checkpoints. A second zero-update replay reproduced the exact
aggregate. Twenty-four targeted tests passed. No metric/seconds
claim, new deployment, CREATE submission, Stage5C, SMC or independent
confirmation.

[Registration](registration.json) | [Results](analysis.json) |
[Checkpoint receipts](head_freeze.json) | [Readout repair](readout_repair_v2.json).
