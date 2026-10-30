# Methods Draft:Identified Costs And Budget-Score Supervision

This is an English methods component,not a submission-ready paper or a
claim that the proposed repair works.Results must come from the frozen
complete readout and separate replay.No result is supplied by this derivation.

## Task And Causal Decision Interface

For a query i,let x_i contain the supplied past target/neighbor states,
past-only detector descriptors and static/predicted scene descriptors.
Observe8annotation steps and request12future steps at stride12raw frames.
Let f_i be the frozen causal floor and p_ik a legal candidate forecast.
The candidate can be a physical motion rule or a frozen neural producer.
Candidate rollout descriptors are functions of x_i and predicted paths,
not future labels.The causal envelope is

```
D_ik = max_t ||p_ik(t) - f_i(t)||_2.
```

Source coordinates are supplied detector-silver bbox-midpoint pixels.
Raw frame spacing is not a verified conversion to elapsed seconds.

## Supervised Costs Under Supplied Label Support

Let M_it indicate a supplied future label.For rows with at least one label,
the observed-point cost of a forecast p is

```
E_i(p) = sum_t M_it ||p(t) - y_i(t)||_2 / sum_t M_it.
R_i = E_i(f_i).
B_ik = max(R_i - E_i(p_ik), 0).
H_ik = max(E_i(p_ik) - R_i, 0).
```

The fixed easy event is0<E_i(CV)<=c_easy,using the original context's
fitting-defined cutoff;it is not an inference feature.The remaining targets
are R_easy=I_easy R and H_easy=I_easy H.Each legal action has five targets
(B,H,R,R_easy,H_easy).A row with no label has no observed cost,not zero cost.

Partial rows retain their observed-point objective.This is not the identified
complete12step ADE.The completion sensitivity analysis keeps these observed
partial-row costs fixed and bounds wholly unknown rows by causal forecast
disagreement;it does not recover the missing steps within a partial row.
Final-step FDE requires its actual requested endpoint label.These limits
are part of the estimand,not repaired by fitting a confidence head.

Labels and label masks enter supervision/evaluation only.The original
past-eligible sampling groups include unknown-outcome rows;the loss mask,
not the model input,excludes their unobserved costs.No future endpoint,
central velocity,remaining-track length or test-endpoint goal is supplied
to the cost head or decoder.

## Identified Moment Model And Original Objective

Training moments and scores are divided by the original positive past-derived
cost scale c_i;inference reverses this scaling.The following loss formulas
use those normalized quantities.The prediction-time risk ratio is unchanged
by the common positive scale.No scale is estimated from new evaluation labels.

The fixed30,405parameter head predicts nonnegative B,H,R,R_easy,H_easy
from462causal features and the candidate identity.Its parameterization
keeps B+H within the causal envelope,B<=R,R_easy<=R and H_easy<=H.
The learned reference can vary by action even though its supervised target
is the query's common floor cost.The earlier shared-reference ablation is
a different architectural factor;it is not introduced here.

Let rho_j be the unchanged physical-fitting RMS and A_i the legal observed
supervision actions.The original objective is normalized moment MSE with
weight4for underpredicted H or H_easy,and weight1otherwise,averaged over
the five moments and A_i.The ranking term centers each action's error in
predicted signed net benefit B-H across A_i and penalizes its squared
deviation,using sqrt(rho_B^2+rho_H^2) as scale.Its coefficient is0.5.
This is continuous supervised cost estimation,not oracle-class learning.

## Coupling The Deployed Risk Numerator And Denominator

The unchanged deployment inequalities motivate two signed scores:

```
q_all = H - alpha R,
q_easy = H_easy - alpha R_easy,
alpha = 0.02.
```

Reference overprediction can make a score too low even if the harm prediction
is correct.The auxiliary loss therefore acts on score residuals rather
than the harm component alone.For each channel use

```
s = sqrt(rho_H^2 + (alpha rho_R)^2),
e = (q_predicted - q_observed) / s,
L_score = mean_i mean_{k in A_i} mean_{all,easy} w(e) e^2.
```

The symmetric arm has w=1.The conservative arm has w=4when e<0,otherwise1.
Both retain the original moment/ranking loss and add L_score with weight1.
All constants are fixed before policy scoring.The quadrature is numerical
scaling,not an independence assumption,estimated standard error or confidence
bound.The combined multitask objective is not a proof that predicted moments
are calibrated or that the decision is safe.

## Inference,Comparisons And Reproducibility

The existing causal decoder admits an action only when predicted net benefit
is positive,both reference predictions are positive,and both predicted
harm/reference budgets pass.It selects the largest admitted predicted net
benefit;otherwise it returns the exact floor.No target-dependent inference
filter,post-readout threshold or new rollout action is introduced.

The matched matrix keeps six contexts,three head seeds,three banks and three
objectives:162models with2,000updates each.All54original arms must exactly
reproduce prior full scientific states.The first conservative new-bank head
also receives a separate same-seed full refit.Checkpoints preserve optimizer,
sampler,Torch RNG and actual loss traces.

Primary is conservative new-bank minus original new-bank;the symmetric
objective and original old bank are fixed controls.Utility comparisons
use each actual selected path,not just a common action index.The denominator
is the original known floor-reference mass.Count matching is a utility
control only;empty comparisons remain undefined.Primary uncertainty uses
3,000locality resamples after context/head-seed averaging,not independent
window resampling.All required recording risks retain the original2%
all/easy contract and missing-outcome sensitivity treatment.

SOURCE is development-exposed.This experiment does not open independent
selection,calibration or confirmation;an exact separate inference replay
tests numerical reproducibility,not scientific independence.No scene-joint
decision training,Stage5C execution orSMC is performed.
