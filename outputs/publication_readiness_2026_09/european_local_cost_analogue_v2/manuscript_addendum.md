# Exploratory Method Note: Cost Alignment Is Not Conditional Risk Reliability

We held the policy-selected query cohort and recording-excluded nearest-history
reference sets fixed, replacing uniform harm/envelope averaging with a bounded
local quadratic fit to the evaluated signed-risk objective. The fit accounts
for causal envelope scale, source-normalized equal-source loss and frozen
reference-prediction offsets. Known reference outcomes supervise fitting;
query future outcomes are used only for evaluation. Uniform, weighted-harm,
global-mean and within-recording shuffled-target controls are fixed in advance.

Across 72 exposed-TRAIN views, local signed-cost fitting reduced easy-risk MSE
from 0.002366 to 0.000442 and all-risk MSE from 0.101728 to 0.089637. However,
easy MSE remained 1.43 times the original head's 0.000309, with zero of 18
context/head-seed paired wins against both comparators. The registered screen
failed. These are risk-estimation metrics, not forecast ADE/FDE improvements.

The weighted-harm ablation explains almost all of the easy-MSE reduction in
this contrast. Reference offsets add only 0.87% while introducing 9,019 outputs
with predicted easy harm greater than total harm. Formal neighbor count also
overstates effective support: a single donor can carry 99.9935% of the local
quadratic weight. Improved average cost error therefore does not establish
coherent conditional risk or a safe intervention policy. Neither the global
mean nor any posthoc feasibility projection replaces the registered primary.

This is a conventional estimator diagnostic, not a standalone method novelty
claim. Upstream training/preprocessing exposure, detector-silver labels and
dependent repeated windows preclude an independent generalization claim.
No new neural model, policy utility test, held/independent calibration or
confirmatory interval is reported. The result motivates support-aware coupled
risk learning but does not demonstrate that follow-up. The full M3W paper still
requires reliable intervention, independent evidence and stronger comparisons.

Protocol: obs8/pred12 rawstride12, pixel/local detector-silver; 2% positive
selected harm/reference. No metric/seconds/human-gold/physical-safety/true3D/
foundation claim. Stage5C/SMC disabled; not submission ready.
