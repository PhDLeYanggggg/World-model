# What This Representation Repair Can Establish

## Material Passport

- Type: code-derived methodological interpretation, not a new empirical result.
- Scope: development-exposed source SDD; frozen obs8/pred12, stride12, annotation pixels.
- Status: written while the registered 36-fit matrix is running, before its readout.
- Inputs: the registered feature implementation and its two preceding controls.
- No new data role, predictor, risk tolerance, policy threshold or external readout.

## Information and Units Are Different Questions

Let `s` be past-only trajectory scale, `D` the causal candidate-to-CV forecast
distance and `c` the frozen training-native cutoff defining the easy target. The
preceding dimensionless head retained normalized shape and `log1p(D/s)` but
discarded `s`. Two samples can have identical normalized shape and `D/s`, yet
different easy labels under a fixed native cutoff. The synthetic witness in the
code demonstrates that possibility; it does not prove the empirical errors have
only this cause.

The present head retains the same first 354 features and supplies `log(s/c)` and
`log1p(D/c)`. Together these recover the scale ratios and `D/s`. No future error
is needed at inference. The future error remains a training/evaluation target.
All folds use the already frozen target/budget cutoff, not the separate cutoff
stored for sampler stratification.

A coordinate-unit relabel must transform `s`, `D` and `c` together. A physical
motion-amplitude change at fixed units must not change `c`. Conflating these
operations was the conceptual risk in discarding all amplitude information.
Neither operation licenses transporting an SDD pixel cutoff into an external
dataset without a justified calibration protocol.

## Not Extra Expressive Capacity

Relative to the native 356-column head, these two columns are an invertible
reparameterization at fixed positive `c`. In exact arithmetic, subtracting
`log(c)` from the scale column cancels under centering. The transformation of
`log1p(D)` is monotone but nonlinear; random-threshold trees can therefore fit
differently even though no information has been added relative to that native
control. Float32 rounding is another possible difference.

Relative to the 355-column dimensionless head, one scalar degree of freedom is
restored. Both column count and numerical representation change. All arms use
118 candidate columns per tree split, but that does not imply identical sampled
column identities or identical tree partitions. This is a representation-level
comparison, not an isolated proof that amplitude alone caused every change.

## Interpretation Rules

1. Report all three rules and all three forecast actions, not a post-readout winner.
2. Compare utility and held-source six-target errors jointly with easy degradation,
   zero-CV harms, unknown-label interventions and incomplete-future bounds.
3. Preserve the old 30 control reductions exactly. A failed matched-count control
   stays failed; a shared numerical risk budget is not shared intervention rate.
4. Report all registered paired contrasts. Three seeds and 3000 paired resamples
   of four already exposed physical sites do not create independent confirmation
   or multiplicity-adjusted significance.
5. Exact metadata unit probes cover only the frozen feature/head calculation.
   They do not establish end-to-end external invariance, calibrated risk control
   or a physical-safety guarantee.

The existing [method-positioning note](../risk_subsidy_v1/literature_and_claim_limits.md)
distinguishes learned risk budgets from formal conformal calibration. No new
formal guarantee, novel neural dynamics result, independent external improvement,
deployment change or submission-readiness claim follows from this experiment.
DroneCrowd confirmation remains closed; IMPTC remains quarantined. Stage5C and
SMC remain off.
