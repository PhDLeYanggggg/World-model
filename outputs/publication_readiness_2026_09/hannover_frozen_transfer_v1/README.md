# Frozen Transfer to Hannover Shared Space

I ran the frozen OpenDD models on the three trajectory tables of the
[Hannover release](https://data.uni-hannover.de/en/dataset/trajectories-in-a-shared-space)
for the first time. The previous intake checked structural support only.
No model or rule was selected on Hannover. This is an out-of-source
diagnostic from one provisional acquisition group, not a new official
test split or a substitute for the sealed DroneCrowd confirmation.

The archive documents 0.5 s per step and 21.1856604 pixels per publisher
unit. The ratio is used as an input conversion; ADE and FDE are reported
in the original image pixels. No independent geometry calibration or
time resampling was performed, and the native time grid differs from
OpenDD. The 8-observation/12-prediction cohort uses no future visibility
to choose targets and retains missing labels. Only pedestrian positions
are used. The package also contains a shared background and user-info
files, which are not read for prediction or redistributed. Repeated
numeric IDs are treated as clip-local. No exact timestamp-position track
duplicate was found across clips; shared people cannot be ruled out.

| Clip | Queries | Targets | All 12 known | No future labels |
|---|---:|---:|---:|---:|
| D1 | 98 | 678 | 560 | 12 |
| D2 | 47 | 408 | 332 | 4 |
| D3 | 17 | 85 | 68 | 1 |
| Total | 162 | 1,171 | 960 | 17 |

## Fixed policy comparison

Each of three original training seeds uses its frozen seven-member
ensemble. The learned choice heads use only OpenDD training. Predicted
arrays are hash-stamped before scoring; the label-aware ceiling is
calculated separately. Half and quarter are fractions of the realized
peak of the original frozen 5% CV blend. All matched candidates share
recipients and peak radii. CV and OLS4 are unrestricted reference paths.

| Forecast | Conservative vs CV (pp) | Direct vs matched hindcast (pp) | Available ADE / FDE (px) | Original risk passes |
|---|---:|---:|---:|---:|
| CV | 0 | -0.9517 | 19.279 / 38.023 | Not estimable (no edits) |
| OLS4 | +6.3984 | +5.9661 | 15.514 / 29.806 | 0/9 vs CV |
| Half: matched hindcast | +0.3630 | 0 | 19.153 / 37.788 | 0/9 |
| Half: tangent | +0.3627 | -0.3272 | 19.145 / 37.749 | 0/9 |
| Half: learned full choice | +0.3744 | -0.2680 | 19.140 / 37.771 | 0/9 |
| Half: six-path utility oracle | +0.9578 | +0.3909 | 19.042 / 37.638 | 0/9 |

The original 5% blend gains +0.7023 pp versus CV and passes 0/9.
Quarter radius still passes 0/9. The means weight each clip equally
after averaging seeds; there are not nine independent observations.
OLS4 is the best tested aggregate accuracy reference, not the best
everywhere: the conservative gain over CV is +22.195 pp in D1,
+0.929 pp in D2 and -3.929 pp in D3. OLS4 is not certified under the
original risk comparison to CV. No multi-source interval is claimed.

## Separating path choice from intervention capacity

The maximum possible gain on each observed point in a CV-centered ball
of radius $r$ is at most $\min(r,\mathrm{CV\ error})$ by the reverse
triangle inequality. Summing this over all observed points and allowing
zero penalty on missing points gives an optimistic upper bound, even
for paths outside the six-candidate bank. It is **+2.7188 pp at half**
and **+5.4077 pp at the original radius**, below OLS4's aggregate
conservative +6.3984 pp. D1 alone has this gap in all three seed views.
These are bounds on the fixed CV-centered intervention budgets, not
claims about models with unrestricted radii or another reference.

At even **quarter radius**, the original easy-risk charge from wholly
unknown edited rows alone is **2.65%-4.56%** in all nine views, before
any known-row harm. Therefore changing path direction at the same
recipients and radii cannot meet the 2% rule on this roster. Selecting
different recipients or changing radii is a different policy and would
still need the same approved risk test. Do not reinterpret this as a
passed risk gate. Deployment, Stage5C, SMC and the confirmation seal
remain unchanged.

```sh
.venv-pytorch/bin/python scripts/run_m3w_hannover_frozen_transfer.py register
.venv-pytorch/bin/python scripts/run_m3w_hannover_frozen_transfer.py run
.venv-pytorch/bin/python scripts/run_m3w_hannover_frozen_transfer.py replay
.venv-pytorch/bin/python scripts/audit_m3w_hannover_radius_capacity.py
```

[Frozen registration](registration.json) | [All scores](analysis.json) |
[Geometric and risk ceilings](capacity.json).
