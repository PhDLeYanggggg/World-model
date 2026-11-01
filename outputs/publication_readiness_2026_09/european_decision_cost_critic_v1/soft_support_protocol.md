# Post-Hoc Training-Support Diagnostic

Declared after observing positive expected training-record constraints,during the
registered18-head fitting phase and before reading this experiment's policy outcomes.
This is explanatory SOURCE analysis,not an added primary metric,gate or model-selection rule.
No original config,loss,temperature,training budget,decoder or risk requirement changes.

The head bounds normalized B+H by C=max(envelope,original B+H)/past_scale.
Thus net-gain logits satisfy -C/T<=z<=C/T while fallback logit is exactly0.
This imposes a nonzero minimum soft intervention probability whenever an action
is available. Calculate its exact row bound and the best possible expected risk
for each training recording under this relaxed logit box.
For each of the three label/bound constraints,minimize expected row cost over all
logit-box vertices using fractional-root bisection,then sum by recording and subtract
the original easy-reference constant. The box ignores shared-network and coupled
moment restrictions,so its optimum is a lower bound on attainable training risk.
Positive record lower bounds prove this smooth surrogate cannot satisfy those
training constraints,even before considering model capacity or optimization.
Nonpositive bounds prove neither attainability nor final-policy safety.
Labels and unknown-envelope bounds enter this offline diagnostic only;they are
never features. Test128 box vertices on synthetic cases independently of the root
implementation. Publish every context and record support count,not a chosen failure.
No independent/test data,no new training,no physical safety or calibration theorem.
