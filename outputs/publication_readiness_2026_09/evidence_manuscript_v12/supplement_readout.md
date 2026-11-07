# Decision Supplement And Source Influence

**New calculation:** delete each of the 14 development source groups in
turn, keeping all three forecast seeds together for retained sources.
The forecasts, decisions, data roles, primary statistics and 2% risk
rule are unchanged. The inputs are verified aggregate statistics, not
fresh forecasts. The original point estimates and 3,000-draw paired
intervals replay exactly for the 15 checked contrast/metric cells.

| Relative-path ADE comparison | Positive observed groups | Observed omission range, pixels | Full-grid lower omission range, pixels |
|---|---:|---:|---:|
| Joint vs. independent | 14/14 | [+0.1833, +0.2462] | [-1.0410, -0.2223] |
| Joint vs. shared family | 13/14 | [+0.1946, +0.2797] | [-1.3510, -0.5466] |
| Joint vs. additive | 8/14 | [-0.0002, +0.0172] | [-0.1725, -0.0798] |
| Joint vs. CV | 9/14 | [-0.5041, -0.2248] | [-2.3566, -1.3456] |
| Joint vs. OLS4 | 1/14 | [-4.1801, -3.4764] | [-16.5115, -12.6374] |

The observed joint-minus-additive effect also changes sign across
seed means, from -0.0285 to +0.0559 pixels. Its conservative lower
mean is negative for every seed and every source omission. The
observed joint advantage over independent selection is more
consistent, but cannot establish a full-grid improvement. This is
an influence check, not a new confidence interval, a physical
collision metric, or a rule for discarding unfavorable sources.

The three-page anonymous supplement documents the exact unweighted
selected-row risk ratios, the distinct query-balanced accuracy
estimand, the forecast and graph controls, the outcome support
counts, and the precommitted independent protocol. The references
distinguish [regression deferral](https://proceedings.mlr.press/v235/mao24d.html),
[subgroup selective regression](https://proceedings.mlr.press/v162/shah22a.html),
and [expected conformal risk](https://openreview.net/pdf?id=33XGfHLtZg)
from the original empirical recording rule. No claim of novel
deferral or conformal safety follows.

## Reproduction

Use the existing verified [development aggregate ZIP](../review_replay_v1/README.md).
No raw trajectories or external calibration labels are needed.

~~~sh
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v12.py
cd data/stage_cvpr2027_experiments/evidence_manuscript_v12/supplement
tectonic -X compile supplement.tex
~~~

The builder checks the aggregate ZIP, the underlying replay receipts
and the official style. It emits the exact local influence numbers
and the values read by LaTeX. Influence JSON SHA-256:
df4207328f0f39266b89bf71f8f5928dbfcb89fb7dba868a8c2d1e1225010f33.
The PDF was reviewed page by page. Rebuilding in an empty directory
produced identical page rasters, generated values and influence JSON.
Seventeen focused tests passed. The supplement does not add independent evidence or
promote a model; Stage5C and SMC remain off. Native annotation steps
and image pixels do not establish metric geometry or physical safety.
