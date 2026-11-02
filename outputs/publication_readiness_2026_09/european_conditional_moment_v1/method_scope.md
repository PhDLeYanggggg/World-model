# Conditional Cost Factorization: Scope

This experiment reuses the project's earlier easy-hurdle idea. Its old negative
results remain negative; this is not a newly invented hurdle method. The new
application is a coherent full seven-candidate, five-moment bank with shared easy
occurrence and conditional reference levels, complete causal query coverage and
frozen current-bank TDEF ranking.

Let E denote the future easy-event label, used only in fitting/evaluation. Causal
inputs predict q=P(E=1|X) and conditional B/H/R magnitudes. The composed outputs
are B=(1-q)B0+qB1, H=(1-q)H0+qH1, R=(1-q)R0+qR1, ER=qR1 and EH=qH1.
Reference levels and q are shared across candidate actions. B/H are bounded by
causal path disagreement and B is also capped by the conditional reference.
These are statistical cost factors, not future trajectory branches.

Both matched arms have the same architecture, initial fitting priors, causal
encoder, draws and complete budgets. The identified arm adds occurrence BCE
and class-balanced conditional-cost regression. Unlike the preceding tied
boundary loss, occurrence confidence is not a sigmoid of a mean-harm ratio.
This changes a training objective, not the original pointwise risk contract.

For q>0, EH-0.02ER=q(H1-0.02R1). Predicting occurrence more accurately therefore
does not, by itself, change the pointwise easy-risk sign or certify a safe
intervention. Lower BCE/Brier, latent separation or decoder coherence cannot
substitute for downstream utility and the full recording-level safety check.

Unknown-label rows retain their causal context and NaN outcomes. The conditional
objective targets the observed fitting-label distribution; it neither identifies
unobserved outcomes nor establishes missing-at-random or population safety.
Both arms start fresh output heads from equal fitting priors, not an exactly
prediction-preserving continuation of the old critic. Capacity and initialization
are matched within this comparison only.

All outcomes are exposed SOURCE fitting/resubstitution on the original roster.
Three cost seeds share forecaster43. There is no new trajectory training,
independent calibration/test access, deployment, Stage5C execution or SMC.
Detector-silver local-image/raw-frame results are not metric, seconds, true3D,
foundation or human-gold evidence. Whether this factorization helps is an
experimental question, not an implication of the algebra.
