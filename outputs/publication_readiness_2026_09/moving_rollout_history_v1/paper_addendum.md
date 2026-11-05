# Rollout-History Relations: Small Development Gain, Safety Still Failed

We add32 causal relations between observed motion, a frozen baseline forecast
and each candidate forecast to a neural intervention-risk model. Features
include along/across-heading deviations, acceleration alignment, speed ratios
and candidate-baseline discrepancies. Two separate risk/decision encoders each
receive a zero-initialized32to64 branch. The matched control masks the new
features; both models have68,356parameters and identical initial predictions.
The original joint objective, complete-query draws, source roles and2,000-update
budget remain fixed. This is a feature-bias comparison, not a new forecasting
architecture or a pure effective-capacity control.

Across three existing contexts, all six fits and one full treatment refit
complete. Full-fitting risk MSE improves8.74%,10.48%,1.98%, while joint decision
cost changes are mixed. On seven already exposed development sources, the
known-label ADE gain over the physical floor is0.9813% versus0.8948% for the
matched control. The+0.0865pp contrast has missing-label completion bounds
[+0.0451,+0.1279]pp, not a sampling CI. Hard gain is1.3032% versus1.1698%.

The policy fails the original requirements: worst easy-net degradation is
8.6603%, and no recording view satisfies the full risk/support contract(0/17).
Only two sources improve, four worsen, and one is unchanged. Source078 accounts
for most positive utility. Removing the decision head raises known gain to
3.2682% but worsens maximum easy degradation to25.7262%. Neither arm is deployed.

These results support a limited representation effect, not reliable safe
intervention. All model weights, fitting metrics and development predictions
replay exactly; this is reproducibility, not independent confirmation. One
producer seed, repeated development exposure, missing outcomes and unavailable
original24-group primary remain limitations. The original2% risk/easy/support
rules are unchanged. No identity exemption or new reserved test evaluation.

[Method positioning and primary references](method_positioning.md) distinguish
this experiment from EqMotion's equivariant motion model and Learn then Test's
calibration guarantee. No geometric-feature or theoretical novelty is claimed.
This addendum is separate from the unchanged compiled main PDF. It does not
make the paper submission-ready. Stage5C/SMC remain off; coordinates are native
obs8/pred12 image coordinates, not metric/seconds-level world dynamics.
