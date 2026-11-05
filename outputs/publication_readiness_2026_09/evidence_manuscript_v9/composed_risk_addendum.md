# Jointly Learning A Conservative Intervention Decision

This is an exploratory addendum, not a successful primary result or a section
already incorporated into the compiled manuscript. Original scientific rules,
the unsupported24-group primary and all earlier failures remain unchanged.

## Hypothesis And Fixed Method

Separately optimized risk-regression and decision heads need not minimize the
cost of their conjunction. We test an added cost-weighted logistic surrogate
on the per-channel maximum of their normalized outputs. The original risk MSE
and decision objectives remain active, including conservative squared-error
supervision for missing labels. Only observed labels contribute to the new
term. Its coefficient is fixed at0.5 before fitting and subsequent readout.

Six models compare independent and composed losses across three own-fitting
contexts with identical64,260-parameter separate encoders,492 causal features,
warm initializations, query draws and2,000 updates. All three fresh independent
controls exactly reproduce the preceding models. A complete treatment refit
and interrupted/resumed fits agree exactly. The trajectory producer is frozen;
this is intervention-risk learning, not new trajectory dynamics or JEPA.

Inference retains both heads, both all/easy channels and original support checks.
The maximum has a sign interpretation, not calibrated probability or certified
risk-bound semantics. The added objective does not directly optimize the full
selected-population risk ratio. No threshold is selected on held outcomes.

## Results

All three treatment contexts pass the unchanged internal fitting screen.
Compared with independent training, fitting all-decision cost falls30.94%,14.68%
and9.89%; easy-decision cost falls13.64%,9.46% and1.19%. Risk MSE increases by
1.03%,0.59% and0.16% relative to these matched controls, although it falls from
warm initialization. Better fitting decisions therefore have a small regression
tradeoff and do not establish calibration.

On seven already exposed development SOURCE units /17 recording views, known
ADE gain over the floor changes from0.7454% to0.7585%. The matched contrast is
only+0.01313pp, with missing-outcome completion bounds[+0.00766,+0.01860]pp.
These are not sampling confidence intervals or independent significance tests.
The treatment loses2.8420pp against the original constant-control model.

Worst defined known easy-net degradation changes from8.1165% to8.0378%, still
above2%. Known all-risk violations remain10/17 and easy-risk violations12/17;
full original risk/support passes0/17. The risk-only ablation recovers more
utility but worsens easy-net degradation to40.38%. No model is promoted.

## Interpretation And Limitations

The controlled experiment supports a narrow conclusion: joint sign training
repairs part of the fitting mismatch. It does not establish safe cross-source
generalization. Remaining discrepancies include all-action versus selected-action
training, the full all/easy conjunction, tail risk and reference support. These
are hypotheses for further discriminating tests, not settled causal explanations.

Only one producer seed and repeatedly exposed development roles are used.
Partial known ADE must not be presented as complete-roster12-step accuracy.
Independent source-level confirmation, main-method multi-seed uncertainty and
matched strong baselines are still incomplete. Dual heads and cost-sensitive
deferral are established ideas; this result alone is not a novelty claim.

Data/schema/initial checkpoints are cached_verified. Six fits, readout and
replay are fresh computations, not new data independence. No reserved/H labels
are opened. No identity certification or original-rule change occurs. Native
obs8/pred12 image coordinates do not justify metric, seconds, true3D, foundation
or human-gold claims. Stage5C and SMC remain disabled. Submission-ready is false.

[Full experiment](../moving_composed_risk_v1/README.md) |
[Prospective protocol](../moving_composed_risk_v1/protocol.md) |
[Failure analysis](../moving_composed_risk_v1/result_interpretation.md)
