# Hurdle Risk Heads: Fixed Source-Development Experiment

## Material Passport

Registered before new head training or readout. The preceding geometric-cost
experiment released useful predictions but underestimated selected harm. Its
six new positive neural-versus-damping ADE intervals all failed observed safety.
This refit tests loss decomposition, not a new trajectory predictor or a relaxed
scientific risk tolerance. All twelve localities are opened development data.

## Fitting-Only Support Diagnosis

The fresh support audit covers 144 fitting-locality/event/candidate/seed slices.
None lacks positive event-harm labels. For a hypothetical 256-row sample from
one locality, easy-event positive-harm support ranges from 7.50 to 29.47 rows
for neural forecasts, and 4.04 to 24.24 for damping. Actual training draws are
balanced across four localities, not exclusively from one locality. All-event
harm probabilities range 17.79-44.02% for neural and 11.06-37.39% for damping.
Thus absence of positive labels is not the explanation; zero-heavy easy-event
targets can still give poor occurrence/severity identification under MSE.
This is a hypothesis, not a verified root cause. No held outcomes were read by
this support audit. Existing opened-source outcomes remain development evidence.

## Parameterization and Controlled Objectives

Let H_E be positive harm times the fixed all/easy event indicator. D is the
maximum causal candidate-versus-CV rollout separation over twelve steps, with
no target or future mask used. J=1[H_E>0] is a training label, not an input.
Three raw outputs represent reference-event mass, joint harm probability and
conditional normalized severity:

```text
Bhat = softplus(zB)
phat = sigmoid(zP)               targets P(H_E > 0 | causal inputs)
qhat = sigmoid(zQ)               targets E[H_E / D | H_E > 0, causal inputs]
Hhat = D * phat * qhat
```

J includes the event; phat is not P(harm | easy). D=0 forces Hhat=0 but must
not force Bhat=0. Unknown labels are excluded, not treated as zero. Conditional
severity receives no loss on an empty positive batch. Future targets are used
only for fitting labels and subsequent diagnostics.

Two new arms share the same 355 inputs, train-only scalers, initialization,
width64 three-output network, draws, AdamW and 2,000 updates per head:

1. `product_mse`: original mean squared error on Bhat and Hhat only.
2. `hurdle`: the same moment loss plus binary cross-entropy on J plus MSE of
   qhat on positive rows. Both extra losses have fixed weight1; no class weights,
   positive oversampling, hyperparameter search or threshold selection.

Both have 22,979 parameters, 65 more than the preceding two-output heads. The
product-MSE arm controls for that architectural/initialization change. Fitting
probability and severity priors initialize both arms identically. Their product
need not reproduce the original marginal constant because D varies with input.
Keep the original utility heads fixed throughout. The reference-mass output
remains a direct moment target; this does not yet factor all event support.

## Matrix and Evaluation

72 new risk heads =3folds x3seeds x2candidates x2events x2objectives;
144,000 updates. Reuse verified OOF/final forecasters and old/geometric heads.
Four fitting localities and eight complete-producer-chain-excluded localities
per fit. All72 fits must finish and replay before any new outcome readout.

144 views retain `old`, `envelope`, `product_mse`, `hurdle`, each with original
utility. The first two arms must exactly reproduce72 previous views. Report108
contrasts against old risk,72 neural-versus-matched-damping contrasts,36 paired
hurdle-versus-product-MSE and36 hurdle-versus-envelope contrasts. Retain all
negative/undefined results. No outcome selects a deployed winner.

Keep the predicted 2% risk rule fixed. Report all/easy/hard ADE, FDE with its
actual label support, worst-locality easy degradation, zero-CV harm, switch
rate, opportunity accounting, selected/population moment reliability, binary
Brier/ECE and positive-severity error. Probability calibration statistics are
diagnostics, not fitted calibrators or guarantees. Physical safety is unproven.
Use3,000 paired locality-bootstrap draws and three seeds. Views share twelve
localities; intervals are conditional and not multiplicity adjusted.

## Runtime and Prior Art

Native arm64 Torch, CPU4/inter-op1/workers0; pilot100 updates then resume within
2,000 total, checkpoints every200, recorded PID/heartbeat/loss and exact resume.
All72 new samplers must match old draws; replay first4,096 excluded-index rows
per head, then separately recompute full metric accounting. Preserve local
private caches/checkpoints, do not publish them. Use CREATE only if justified
by measured resource needs; never train on login nodes or touch simulation jobs.

The two-part occurrence/severity idea is established methodology, not claimed
novelty. The introduction and model section of [Yiu and Tom (2017)](https://arxiv.org/html/1703.09147v1)
describe binary and positive continuous components for semicontinuous outcomes.
This experiment adapts the decomposition to a causal forecast-relative harm
moment; it does not reproduce their longitudinal stochastic-process likelihood
or inherit a risk-control guarantee. Any paper contribution requires evidence
about baseline-relative intervention, not renaming a standard two-part model.

Released detector tracks, image pixels, obs8/pred12 rawstride12. Not legacyt50,
seconds, metric, human gold, true3D or foundation. No independent confirmation,
new calibration, deployment promotion, Stage5C execution or SMC.
