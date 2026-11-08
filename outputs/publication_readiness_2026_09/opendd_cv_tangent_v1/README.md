# CV-tangent training at a matched edit radius

**Result: a forecast-shape gain, not a deployable policy.** Training
against the local CV error slope improves the full-grid conservative
score on the existing OpenDD development roster at the same edit
recipients and radii. It does not preserve the original recording
risk requirement.

## Matched method

The predictor proposes a bounded shape change from the frozen OpenDD
incumbent. Both new arms use identical networks, initializations,
samplers and loss components except for the directional term. The
ordinary arm minimizes the actual finite change in observed ADE,
positive harm and an auxiliary acceptance loss. The treatment also
optimizes the one-sided derivative of ADE in the proposed direction
at CV. For nonzero residual it is the normalized residual dot product;
at exact zero it is the edit norm. It lower-bounds the finite change,
so a favorable derivative does not guarantee a safe finite edit.
Coordinates and masks of future labels occur only in the loss and scorer.

The fixed departure is one half of the frozen incumbent peak per row.
Training uses pair-excluded producers, with neither the fitting donor
nor the held location in each producer's fit. At inference the producer
excludes the held location. The older learned shape is reproduced from
its checked checkpoint and rescaled to the same half radius. Comparison
uses the prior OpenDD full-grid lower utility (per-missing-step debit),
and the original 2% recording checks are unchanged. This is not a
comparison of the absolute scores from different estimands.

| Policy | Edits | Conservative gain vs CV (pp) | Risk passes / 606 | Undefined easy views |
|---|---:|---:|---:|---:|
| Incumbent half | 30,075 | +0.219 | 547 | 30 |
| Earlier learned shape at half radius | 30,075 | +0.255 | 542 | 30 |
| Finite loss | 30,075 | +0.280 | 530 | 30 |
| CV-tangent loss | 30,075 | +0.306 | 522 | 30 |
| Tangent with fixed 0.6 guard | 25,109 | +0.291 | 508 | 41 |

Direct paired contrasts use 3,000 location bootstrap draws after
averaging the three actual seeds. Tangent minus finite is **+0.0228 pp**
[+0.0066, +0.0458]. Tangent minus the earlier shape is **+0.0396 pp**
[+0.0198, +0.0574]. Worst defined signed easy degradation increases
from 11.54% for the earlier shape to 26.26% for tangent. The
relative-vector error diagnostic improves by -0.000219 release units
[-0.000334, -0.000117] versus finite; this is not physical safety.

The fixed probability guard copies CV exactly when it rejects. It was
not tuned on held locations or calibrated as a risk guarantee.
At the same guarded recipients and radii, the earlier shape passes
531/606, so the guard does not rescue the safety shortfall.

## Keep support while reallocating amplitude

A separate no-refit diagnostic uses one smaller budget: half of the
training radius. It preserves every edit and the exact sum of peak
departures within each query. Per-row weights shift around the uniform
allocation using the radius-weighted mean confidence in that query;
neither test labels nor other queries enter the allocation.

The adaptive tangent gains **+0.0011 pp** [+0.0005, +0.0017] over
uniform tangent with the same query budget. Both pass **560/606**;
uniform earlier shape passes 564 and uniform incumbent passes 566.
All five reduced-budget arms have 30 undefined easy views, each
with no easy reference anywhere in its recording roster. Budget
redistribution cannot fix that absent support under the unchanged rule.
No deployment or independent confirmation is warranted.

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_cv_tangent.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_cv_tangent.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_opendd_tangent_budget.py
```

42 actual Torch fits (seven held locations, two objectives, three
seeds, 600 updates), native arm64, four threads and zero workers.
Fresh run 22.27 s; no-refit replay 1.92 s. Fifty-one targeted tests
passed. The 606 recording-seed views are repeated development
measurements, not independent scenes. No new data role, physical
unit claim, Stage5C, SMC, enrollment, or submission.

[Training and paired outcomes](analysis.json) |
[Budget-matched diagnostic](budget.json) |
[Working manuscript v50](../evidence_manuscript_v50/README.md).
