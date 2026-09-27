# Evidence Addendum: Unit-Consistent Bounded Correction

## Motivation and Matched Design

When a normalized trajectory network predicts a fraction of a causal motion
budget, restoring coordinate scale before a nonlinear output squash creates
unit sensitivity. We compare the previous bounded predictor
`B + R*q(S*f(x/S))` with `B + R*q(f(x/S))`, retaining its architecture, initial
parameters, data, loss, sampler, optimizer and training budget. The correction
budget itself is unchanged. This is an implementation repair, not a novel
risk-control method. Scale equivariance is conditional on the inherited input
clamp being inactive; general rotation or domain invariance is not claimed.

Nine native-Torch fits use three producer folds and three seeds, each with
4,000 updates. Their forecasts are committed before comparative scoring.
The task observes8 and predicts12 requested positions at raw-frame stride12.
Each producer trains on four source localities and excludes eight; the twelve
localities have already been opened for development. Three thousand paired
locality bootstrap resamples provide exploratory uncertainty, not independent
confirmation. Overlapping windows and producer contexts are not independent units.

## Results

The unit-consistent candidate reduces ADE by4.72% relative to the matched
grouped neural control (95% locality interval2.53--7.89%) and FDE by6.49%
(3.38--11.10%). All three seed intervals favor the candidate. Relative to CV,
all-ADE improves8.46% (1.91--12.98%) and hard-ADE13.36% (8.89--16.60%).

The positive-easy subset, however, remains11.17% worse than CV, with an interval
for degradation of3.14--21.26%. Four reference-exact queries become worse in
each of six dependent views. One locality loses ADE against the neural control;
another exceeds4% easy degradation against it. Thus the candidate passes only
the exploratory matched-forecaster screen, not deployment safety.

On identical unlabeled observed-input prefixes, the candidate passes36/36
uniform rescaling checks and the control0/36. Among729 recorded training
gradients per arm, clipping is triggered3 times versus729. These are mechanism
diagnostics. They do not distinguish the causal effect of equivariance from
changed gradient scaling and effective correction amplitude.

## Implication for the Research Question

This experiment strengthens the candidate forecast bank, but it also reinforces
the need to learn when neural intervention is worthwhile. Average trajectory
improvement does not resolve relative harm, rare reference-exact events or
scene-joint consistency. The contribution question remains matched cost-aware
intervention with independent scene-level calibration, not the use of a
Transformer or a bounded residual alone. This contrast does not establish
image/goal/JEPA contributions, calibrated safety or foundation-model ability.

Labels are released detector-derived image trajectories, not human gold.
Coordinates and horizons are not verified meters or seconds. No independent
calibration/confirmation readout, deployment, Stage5C or SMC is performed.
This addendum supplements the working evidence manuscript; it is not a complete
submission-ready paper or an independently confirmed main result.
