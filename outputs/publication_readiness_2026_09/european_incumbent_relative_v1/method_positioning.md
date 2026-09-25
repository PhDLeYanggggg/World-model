# Method Positioning And Limits

Literature checked2026-09-25. This note does not alter the registered experiment.

## Prior Work

[Laroche, Trichelair and Tachet des Combes, ICML2019](https://proceedings.mlr.press/v97/laroche19a/laroche19a.pdf)
study baseline-preserving improvement in batch reinforcement learning. Section2
defines constraints that retain baseline action probabilities for poorly sampled
state-action pairs; Theorem2 depends on the specified MDP and uncertainty setup.
Preserving an incumbent when evidence is weak is therefore not a new general
principle introduced by M3W. Our supervised forecast-cost experiment does not
meet those RL theorem assumptions and inherits no SPIBB safety guarantee.

[Mao, Mohri and Zhong, ICML2024](https://proceedings.mlr.press/v235/mao24d.html)
study regression with multiple-expert deferral, including a fixed predictor
followed by a learned deferral rule, bounded regression losses and label-dependent
costs. Their official abstract and bibliographic record were checked; the linked
PDF fetch failed in this session, so no full-proof verification is claimed.
Learning which prediction to use is not by itself a new contribution either.

## What This Test Can Establish

Both learned arms observe the same causal features, original action bit and
candidate/floor forecasts. The floor-reference target asks whether neural is
better than the motion floor. The incumbent-reference target asks whether the
alternate action improves the original decision. When the original selects the
floor these targets coincide; when it selects neural, benefit and harm swap and
the risk-reference cost changes. The fallback semantics must change with that
reference. Consequently this tests target-plus-policy parameterization, not a
loss-only causal effect or a fundamentally different predictor class.

For a fixed causal input x and deterministic incumbent action a0(x), let the two
conditional trajectory costs be c_floor(x) and c_neural(x). Subtracting
c_a0(x) from both costs does not change their optimal unconstrained ordering.
Thus an ideal estimator gains no new information merely by changing reference.
Any empirical difference here comes from finite-sample fitting, the bounded
cost parameterization and the reference-dependent gate/fallback. In particular,
2% of incumbent cost is not2% of floor cost where the incumbent chose neural.
These experiments cannot attribute an effect exclusively to the regression
loss, or claim that changing reference alone creates new dynamics knowledge.

Directional rules are declared in advance. Add-only can preserve incumbent
neural decisions but can still introduce harmful new ones. Remove-only can undo
harmful incumbent decisions but can also discard useful ones. Neither has a
sample-wise error guarantee; their risks require empirical accounting.

Any useful result here is an engineering/method-development observation. To
support a paper contribution, I still need strong published cost/deferral
comparisons, joint-agent controls at matched intervention or risk, independent
source calibration/confirmation and evidence that the scene structure matters.
The same development sources have already been opened repeatedly; no number of
new bootstrap draws converts them into independent confirmation.

No certified risk, physical safety, new dynamics training, metric/seconds,
human-gold, true3D, foundation, deployment or submission-readiness claim. Stage5C
and SMC remain off.
