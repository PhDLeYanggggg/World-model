# Auxiliary Membership Cost Model Card

## Material Passport

Registered source-development cost-head experiment, not a forecasting-policy
or world-dynamics upgrade. Read completion_checks, compute_receipt and gates
for actual completion and outcomes; the registration budget is not a result.
Native Torch fitting is fresh_run; inherited source forecasts, preprocessing
lineage and original reference costs are cached_verified. Independent
selection/calibration/confirmation are not_run and remain closed.

## Model

383 causal features ->64 GELU ->4 direct nested moments, plus a separate
64->1 easy-membership logit. Both cost_only and membership_aux contain24,901
parameters. The control's auxiliary coefficient is zero; the intervention
adds one binary cross-entropy loss to the shared encoder. The inherited
four-moment cost loss uses fitting-only RMS scales. No class probability is
multiplied into a cost prediction. No future-derived input at inference.

Membership E uses the inherited positive-CV-error25th percentile from three
fitting localities. The fourth locality is excluded from model fitting and
all learned preprocessing. Known zero-harm rows remain supervised; unknown
labels are excluded. Six source assignments and seeds17/29/43 are retained.
Training uses2,000updates/batch256/lr.0003/AdamW and exact-resume checkpoints.

For cost comparisons, replace only predicted H/H_E and preserve original
reference D/D_E arrays. A disagreement envelope bounds predicted harm and
cost nesting; it does not guarantee physical safety or realized policy risk.
Cost/BCE losses are separate diagnostics; total objectives differ by arm.

## Limits

Only a component gate can pass here, not a deployment gate. Four localities
per assignment and repeated exposed source roles limit statistical evidence.
No full/motion modality contribution can be inferred from their different
disagreement populations alone. Weights and row caches remain private.
Pixel/annotation-step, detector-derived supervision: no metric/seconds,
human-gold, true3D, foundation or submission-ready claim. Stage5C/SMC off.
