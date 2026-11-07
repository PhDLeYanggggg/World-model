# Which Forecasts To Change At Fixed Count And Amplitude

**Decision:** Do not replace the fixed-quarter development control.
Dense cost supervision reduced one risk count relative to gain-only
ranking, but did not produce a better policy than the original roster.

## Matched Experiment

The previous atomic exchange used a rare binary rescue target: about
0.015% positives. Instead, this experiment learns dense gain, positive
harm and reference mass for each legal candidate at quarter amplitude.
For each query, the learned arms can change agents and families but
must keep the original intervention count. The gain-only and gain-plus-
harm arms share the fitted network, history descriptors and forecasts.
Subtracting estimated all/easy excess risk from predicted gain is the
only difference in their rankings. The additive objective is solved
exactly by maximizing over families per agent and taking the top agents;
this is not an interaction model or a safety theorem.

Across 14 source groups, 34 exposed recordings and three forecast
seeds, the 42 fresh fitting folds each use seven fitting and six donor
sources, excluding the scored source from both. The model trains on
causal features; target labels are used only for supervision, donor
selection and scoring. All learned arms retain exactly 6,952 interventions
(1.846%). Rules and support requirements are unchanged.

## Results

| Arm | Original Passes / 102 | Total Risk >2% | Signed Easy >2% | Undefined Easy | Replaced Agents |
|---|---:|---:|---:|---:|---:|
| Fixed quarter, original roster | 15 | 39 | 0 | 35 | 0 |
| Learned gain only | 7 | 64 | 2 | 33 | 6,824 |
| Learned gain and harm | 7 | 56 | 2 | 35 | 6,766 |
| Donor-calibrated | 14 | 38 | 0 | 36 | 18 |
| Hindsight additive (diagnostic) | 6 | 0 | 0 | 96 | 6,902 |

| Paired Contrast | Observed-Label Gain (pp) | Completion Lower (pp) | Source Bootstrap 95% CI For Lower (pp) |
|---|---:|---:|---:|
| Gain and harm vs gain only (primary) | -0.01872 | -0.28547 | [-0.52885, -0.10279] |
| Gain only vs fixed quarter | +0.17344 | -0.35805 | [-0.83541, -0.04505] |
| Gain and harm vs fixed quarter | +0.15472 | -0.19471 | [-0.56199, +0.07197] |
| Donor-calibrated vs fixed quarter | +0.01047 | +0.01005 | [0, +0.02700] |
| Hindsight additive vs fixed quarter | +1.23221 | +1.22953 | [+0.68113, +1.86072] |

**The negative primary contrast is for the completion lower bound, not a
demonstration that actual unobserved errors are worse.** The direct
pair's completion upper point is +0.24802 pp. Bounds use the actual
disagreement between the two paths for wholly unknown rows, so they
cannot be reconstructed by subtracting two bounds against CV.

Both learned rankings change most of the original selections, and
unobserved outcomes can reverse the apparently favorable observed-label
comparison. The donor screen reverts to the original roster in 40 of
42 folds. Hindsight shows that the candidate set permits utility gain
at the same count, but its unconstrained score selects too few
easy-reference cases to assess risk in 96 views. Zero counted total-risk
violations is not an all-recording pass. This separates utility ranking
from preserving reference support; it does not prove either is learnable
on independent scenes. The next useful model comparison should limit
the number of replacements and account for retaining easy-reference
support rather than reranking every selected agent.

## Paper Position And Reproduction

Regression deferral already learns to allocate examples to different
predictors ([Mao et al., ICML 2024](https://proceedings.mlr.press/v235/mao24d.html)).
Joint forecast consistency is also an established problem
([Ngiam et al., ICLR 2022](https://openreview.net/pdf?id=Wm3EA5OlHsG)).
The specific question here is whether baseline-relative gain and harm
can improve decisions at matched intervention count under partial
outcomes. These results are a negative ablation, not a new calibrated
guarantee. Donor selection does not satisfy the assumptions of
[conformal risk control](https://arxiv.org/abs/2208.02814).

```sh
.venv-pytorch/bin/python scripts/run_m3w_quarter_allocation.py fit
.venv-pytorch/bin/python scripts/run_m3w_quarter_allocation.py score
.venv-pytorch/bin/python scripts/run_m3w_quarter_allocation.py replay
```

Registered in `4921731e`. Native arm64 PyTorch used four compute
threads, no loader subprocesses, 440 updates per fold, checkpointed
every 55 updates. Fitting took 130 s. There were 17,884,001 fitting
candidate occurrences across repeated folds, not independent examples.
The original quarter score matches exactly. Fresh score and replay
bind the same [analysis](analysis.json); forecast banks are verified
cache. Thirty-six targeted tests passed. The CREATE read-only query
timed out; no jobs were submitted or restarted.

Eight observed steps and twelve predicted steps are in image
coordinates. No independent confirmation, metric or seconds-level
result, deployment, true 3D, foundation-model or submission-readiness
claim follows. DroneCrowd outcomes remain sealed; Stage5C and SMC stay off.
