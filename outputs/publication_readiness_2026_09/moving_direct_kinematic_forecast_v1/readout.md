# A Larger Feasible Candidate Bank, Not A Deployable Policy

I tested a direct kinematic model after the PCT loss changes failed to improve
conservative utility. Instead of adding an unconstrained correction to CV,
the model learns to combine the seven observed velocity increments and
integrates the resulting velocity at each of the 12 requested steps. A matched
peer variant lets nearby tracks inform the weights and speed. The question is
whether adaptive history weighting beats fixed smoothing, and whether peer
context helps after this change. The strong CV comparison follows the
[motion baseline study by Scholler et al.](https://arxiv.org/abs/1903.07933);
[EqMotion](https://arxiv.org/abs/2303.10876) is the existing learned comparator.
This is not a reproduction of their headline benchmark results.

## Forecast Accuracy And Conservative Utility

Points are paired, query-balanced, source-macro normalized ADE gains in
percentage points. Brackets are 95% intervals from 3,000 source resamples
after averaging three trained seeds. The A set is already exposed. The
completion lower bounds wholly unknown rows by paired path disagreement;
partially labelled rows retain their observed-step loss.

| New model vs comparator | Known-label gain (pp) | Paired completion lower (pp) |
|---|---:|---:|
| Self vs fitting-selected OLS4 | +17.10 [6.41, 28.76] | +0.75 [-14.16, 14.45] |
| Peer vs fitting-selected OLS4 | +18.65 [8.29, 29.52] | +0.71 [-11.44, 11.97] |
| Self vs CV | +14.85 [5.17, 24.89] | -5.69 [-33.51, 14.10] |
| Peer vs CV | +16.37 [6.62, 26.04] | -5.31 [-29.54, 11.62] |
| Self vs EqMotion (OLS4 normalization) | -3.37 [-6.05, -0.84] | -19.62 [-33.59, -9.01] |
| Peer vs EqMotion (OLS4 normalization) | -1.83 [-4.55, 0.13] | -17.45 [-31.39, -6.69] |
| Peer vs self (CV normalization) | +1.52 [-0.82, 3.85] | -7.71 [-12.51, -3.12] |

Adaptive history weights improve accuracy on known labels, but do not
establish conservative superiority over OLS4 or CV. Peer information does
not establish a gain over the matched self arm. Its additional disagreement
cost makes the direct completion contrast worse. This is evidence about
this peer encoder, not an assertion that interactions are irrelevant.

## Candidate Feasibility And Original Risk

The previous bank reproduces exactly at 101/102 feasible views. Adding
either kinematic model alone raises this to **102/102 feasible and positive**
for the free-count, future-label-aware oracle. In the formerly infeasible
view (recording 10, seed 43, source 053), the self-expanded oracle selects
38 of 39 paths, with zero selected easy positive harm and -3.58% signed easy
net degradation. The selected alternative is genuinely different from CV;
there is no identity certificate, support exemption or removed recording.

This resolves **availability on exposed A**, not learnable selection. Both
new raw forecasters still pass **0/102** views: each violates the easy
completion budget in 102, and the all completion budget in 99. Signed easy
net exceeds 2% in 75 self and 81 peer views. EqMotion and OLS4 are also 0/102
when used raw under this same CV-relative rule. Worst recording p95 known ADE
is 323.77 (self), 326.01 (peer), 360.14 (EqMotion), 335.11 (OLS4) image pixels;
the corresponding worst recording mean is 213.86, 213.60, 214.03 and 209.71.
The scalar record table retains partial support, available endpoint FDE,
easy/hard scores, intervention rates and every risk denominator outcome.
No oracle row choices are exported and neither variant is deployed.

## Next Experiment

Use the self-only candidate as the more parsimonious new action. Build its
source-out-of-fold forecasts for fitting a multi-action gain/harm allocator,
and compare agent, scene and joint allocation on the same paths and count.
The oracle's 102/102 result says this experiment is now worth attempting; it
does not tell us whether the relevant easy-support choices can be predicted
from the past. Independent DroneCrowd scientific roles are still unassigned.
No independent inference or labels were used. The original 2% contract,
8/12 native image-coordinate task, Stage5C and SMC status are unchanged.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_kinematic_forecast.py prefix
.venv-pytorch/bin/python scripts/run_m3w_kinematic_forecast.py train
.venv-pytorch/bin/python scripts/run_m3w_kinematic_forecast.py infer
.venv-pytorch/bin/python scripts/run_m3w_kinematic_forecast.py score
.venv-pytorch/bin/python scripts/run_m3w_kinematic_forecast.py replay
```

Protocol commit `fc53afb0`. Actual native CPU4/interop1/workers0 Torch fits:
six 9,768-parameter models, 69,746 fitting rows, 14,400 updates including the
20-step prefix, and exact paired draw/sampler/RNG equality. The remaining
14,380 updates complete in 22.81 s. All six paths for each of 125,540 A rows
are frozen before scoring (753,240 forecasts, 3.41 s). Original and expanded
bank scores replay exactly (17.87/17.98 s; all processes exit 0). CREATE
read-only SSH stopped at host key verification; no remote job or status
claim. The benchmark remains image-coordinate, not metric or seconds-level.

[Protocol](protocol.md) | [Scalar results](analysis.json) | [Score replay](replay_execution.json)
