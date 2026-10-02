# Conditional Harm Failure Taxonomy

## Established Within the Frozen Model

1. **Unsafe expansion dominates added actions.**22,805 additions versus5,827
   removals;99.45% of additions were previously blocked by easy risk. This is a
   deterministic response to changed cost predictions under fixed thresholds.
2. **Average added easy risk is harm-underestimation dominated.** The observed
   known-label added ratio is7.1334%; the harm-error term contributes+6.2414pp,
   while reference inflation contributes-0.1926pp. Unknown outcomes are excluded
   from this observed ratio and remain in the separate conservative bound.
3. **Both harm channels matter.** Harm-only yields15 known-label violations;
   easy-harm-only yields7, versus original4. Removing harm reduces the complete
   quality model's20 known-label violations to3 but does not restore support.
4. **Projection couples predictions.** After benefit/harm envelope scaling,
   easy harm is capped by total harm. It can fall when only total harm changes.
   The preprojection comparison proves action coupling, not that every failure
   was caused by clipping. No retrained ablation was performed here.

## Added Versus Retained

| Cohort | Occurrences | Unknown | Defined easy-risk groups / localities | Known easy risk (%) | Predicted excess (pp) | Harm error (pp) | Reference error (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Added | 22805 | 299 | 62/12 | 7.1334 | -0.9154 | +6.2414 | -0.1926 |
| Removed | 5827 | 74 | 33/9 | 2.0853 | +0.4125 | -0.5281 | +0.2009 |
| Retained | 89628 | 844 | 41/11 | 0.8365 | -1.1403 | -0.4194 | +0.3962 |

These descriptive means first average defined head views within locality, then
localities. Different cohorts have different support; they cannot be subtracted
as randomized treatment effects or treated as independent populations. Unknown
outcomes do not become observed harm. `cohort_decomposition.json` preserves each
locality and the averaging rule.

## Concrete Known-Label Failure

For `single2_seed43_controller0_dimensionless_fit_eu-locality-110`,head29,
quality adds one known occurrence. Actual harm is0.460929 and reference0.352385.
Predicted total harm is0.732814, but predicted reference47.653942. Its all-risk
failure is reference-inflation dominated even though total harm is overpredicted.

For the same row, predicted easy harm is0 and easy reference0.090701. Actual
easy harm is0.460929 and easy reference0.352385. Its easy-risk failure is instead
harm-underestimation dominated. A single row therefore exhibits different failure
mechanisms for the two constraints. A global denominator rescale cannot be
assumed to repair both, and a zero predicted easy harm is not a safety certificate.

## Not Established

No proof that annotation noise, raw-history length, clipping alone or model size
causes the residual failure. No proof that deleting a harm component produces a
deployable policy. The only/drop interventions change an already fitted model;
they are not retraining evidence. No independent generalization or calibration.

The useful past-quality information should be retained for a targeted learning
test. The next hypothesis concerns the conditional harm output and its loss,
not another global cutoff. It must still handle unknown-outcome support and pass
the original risk screen; positive parameterization alone cannot guarantee that.
