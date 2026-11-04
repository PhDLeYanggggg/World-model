# Why Aggregate Excess Is Different From Row Prediction Loss

This is an elementary loss identity and conditional sufficient condition,
not a new calibration theorem or a claim that this experiment is safe.
Use the unchanged released-label benchmark cost functional, not true physical
safety or unobserved full12step costs.

For a fixed selected multiset S, let g_i = H_i - tau R_i with tau=.02.
The easy analogue is g_i^easy = EH_i - tau ER_i. Unknown whole-row labels
use the previously specified conservative Hupper/Rlower contributions;
partial labels remain partial. The original requirement also needs positive
supporting reference mass, so a zero denominator does not become a safe ratio.

Let e_i = predicted_g_i - target_g_i. For a fixed positive causal normalizer A,

    (sum_S e_i / A)^2
      = sum_S e_i^2 / A^2 + 2 sum_{i<j in S} e_i e_j / A^2.

Separate row MSE has no such cross-error terms. Consequently a change can
improve row MSE and worsen aggregate mass error, as the fixed-set diagnostic
demonstrates descriptively. The code applies this identity to each sampled
recording multiset, with A=sum(selected causal scales)*fitting RMS units, then
averages active records within sources. Random multiset membership, duplication
and a random normalizer mean this is not an unbiased full-video squared-loss
estimator. No pointwise or per-record guarantee follows from minimizing it.

A separate, valid aggregate residual bound u(S) would imply

    target_excess(S) <= predicted_excess(S) + u(S).

If the right side were nonpositive and actual reference support positive,
the corresponding ratio-free risk inequality would hold for that set. This
is only a conditional implication. We do not possess such a calibrated u(S)
from this training loss. Marginal mean predictions, same-Cteacher masks and
source bootstrap intervals are not substitutes for selection-aware independent
calibration. The easy-support requirement remains separate.

The fixed teacher set S0 used in fitting may differ from the new controller's
set S_theta at evaluation. Teacher masks are trained on C, not honest OOF
validation labels. Thus there is both selection transport and finite-support
uncertainty. These are limitations, not removed by the unchanged network's
causal inference schema. All Hpolicy criteria remain those registered before
training; no loss weight or threshold is chosen from their results.

The general prediction-versus-decision issue is established literature; see
related_method_position.md. This study implements neither SPO+ nor a new
distribution-free risk guarantee. Original24group primary remains unavailable,
and all23supported groups are exposed development. Stage5C and SMC remain off.
