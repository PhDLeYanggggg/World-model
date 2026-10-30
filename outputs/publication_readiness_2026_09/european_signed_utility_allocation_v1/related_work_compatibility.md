# Optimization control,not a new calibration guarantee

## Material Passport

Read-only literature/method positioning while the fixed full experiment runs.
No paper is claimed reproduced here. This changes no model,threshold or result.

## Decision calibration

Zhao etal.'s NeurIPS2021 formulation uses multiclass predicted distributions;
their loss is a function of label and action,and the decision rule depends on
features through those predictions. Their discussion explicitly distinguishes
population averages from individual assurances and excludes direct feature-
dependent decision rules from that formulation. Our source-dependent costs,
continuous futures and joint allocation do not automatically inherit it.
[Primary paper,Sections2-4](https://proceedings.neurips.cc/paper/2021/file/bbc92a647199b832ec90d7cf57074e9e-Paper.pdf).

In particular,seven forecast options per agent are not seven joint actions:
with abstention andNagents there can be up to8^N combinations before constraints.
Neither signed-utility enlargement nor lower fitting MSE implements that
recalibration algorithm. We claim no finite-sample decision-calibration bound.

## Regression expert deferral

Mao,Mohri and Zhong's ICML2024 paper studies regression routing with multiple
experts in single-stage and pretrained-predictor two-stage settings,including
surrogate consistency results. Generic routing between motion and neural
forecasts is therefore not a novel contribution.
[Primary proceedings and abstract](https://proceedings.mlr.press/v235/mao24d.html).
The full PDF fetch was unavailable in this pass;no detailed theorem transfer
or faithful implementation is claimed. A compatible strong control remains
missing,not silently treated as outperformed.

## Newer nonlinear calibration

Tang etal.'s2025 preprint considers nonlinear losses. Its abstract distinguishes
deterministic best-response limitations from a smooth best-response relaxation,
and gives conditions involving bounded-norm RKHS approximations. That does not
certify this deterministic integer decoder or missing-outcome risk ratios.
[Primary abstract](https://arxiv.org/abs/2504.15615).
Only its abstract was inspected here;it is not an implemented comparison or
a claimed impossibility result for M3W.

## What the present experiment can contribute

The [counterexample](mechanism_note.md) shows why individually nonpositive
utility need not preclude a positive feasible combination. This is an ordinary
discrete-optimization distinction,not a novel theorem. The SOURCE experiment
tests whether this distinction matters with fixed estimated risks at equal
intervention counts. Even a positive fitting effect would still need original
recording risk,independent evaluation,strong decision-method controls and useful
neural dynamics evidence before supporting a main paper claim.

No metric/seconds,human gold,true3D,foundation,independent success or deployment
claim. Stage5C execution and SMC remain disabled.
