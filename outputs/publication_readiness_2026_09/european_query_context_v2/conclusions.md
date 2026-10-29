# Same-Query Context Does Not Improve Safe Intervention

## Result

The corrected comparison is complete: 36 real Torch risk-head fits, 72,000
updates and 72 frozen TRAIN-internal held-recording views. All eight inherited
controls reproduce exactly. Adding a leave-self-out mean of contemporaneous
agent features does not advance the method. Deployment is unchanged.

Compared with the equal-parameter self-only head, peer context changes
matched-count paired completion-lower utility by **-0.239274 percentage points**,
nominal 95% locality-bootstrap CI **[-0.352816, -0.143938]**. Full-policy change is
**-0.980376 pp [-1.630511, -0.496614]**. All three head seeds have negative matched
contrasts. No peer view has complete risk support; 44/72 violate the known
selected positive-easy-harm criterion, and 12/72 have undefined easy risk.

These are cost-selection utility measures, not ADE/FDE changes, historical
raw-frame t50 results or a population-safety guarantee. The original 2% positive
easy harm/reference criterion is unchanged; it is not net easy ADE degradation.

## Complete Comparison

Peer minus comparator; positive utility is better. Intervals use 3,000 draws
over 12 already exposed original-TRAIN localities, averaging repeated context
and head-seed views within locality first. They are nominal exploratory
intervals, not independent or adaptive-search-adjusted confirmation.

| Comparator | Full utility, pp [95% CI] | Matched-count utility, pp [95% CI] |
| --- | --- | --- |
| local | -2.0515 [-4.1667, -0.2365] | -0.0809 [-0.4304, +0.2832] |
| anchor | -2.1287 [-3.3282, -0.9772] | +0.2184 [-0.2437, +0.8611] |
| old_risk | -0.2022 [-1.6333, +1.0867] | +0.0965 [-0.0106, +0.2249] |
| utility_continue | -2.8917 [-4.0340, -1.7610] | -0.1059 [-0.3640, +0.1552] |
| joint_risk | -0.6112 [-1.3806, +0.1562] | +0.0269 [-0.1091, +0.1575] |
| isolated_risk | -0.0279 [-1.3381, +1.3297] | +0.1701 [-0.0529, +0.4615] |
| full_range | -1.3936 [-2.0473, -0.8069] | -0.2398 [-0.4074, -0.0753] |
| rational | -0.5830 [-0.9068, -0.2620] | -0.1585 [-0.2499, -0.0771] |
| self | -0.9804 [-1.6305, -0.4966] | -0.2393 [-0.3528, -0.1439] |

The prespecified screen fails all 18 positive-CI conditions and the complete
support condition. A few positive point estimates against older controls do
not satisfy it and do not override the matched self-only negative result.

## Risk and Coverage

| Model | Selected occurrences | Unknown selected | Known easy violations /72 | Undefined easy /72 | Complete support /72 |
| --- | ---: | ---: | ---: | ---: | ---: |
| rational | 27,371 | 699 | 34 | 19 | 0 |
| self | 32,621 | 847 | 44 | 14 | 0 |
| peer | 24,501 | 728 | 44 | 12 | 0 |

All three have three empty policies. Counts are repeated context/seed
occurrences, not independent people or observations. Fewer interventions do
not establish safety: known easy violations remain, and unknown-label bounds
cannot be replaced with zero error. Peer has 58 defined completion-upper easy
risk violations, compared with self 55 and rational 51.

## Provenance

- `fresh_run`: corrected 36-head training, 72 ten-arm readouts, bootstrap and
  frozen-output decomposition. PID60147 training exited 0 in 381.689s, peak
  resident memory 2.952GB. PID61062 evaluation exited 0 in 95.521s.
- `cached_verified`: source packets, frozen upstream predictors, preprocessing,
  source exclusions and all eight inherited comparator predictions/policies.
- `not_run`: independent roles, outer development, new trajectory training,
  unseen-domain transfer, deployment and formal submission. Their prerequisites
  are not met. Three head seeds are not three newly trained forecasters.
- The first v1 run mistakenly froze its copied encoder. All 36 fits and 72,000
  updates are preserved separately, with no held readout. The repair was
  registered before v2 fitting and before any new held-out outcome analysis.

The experiment concerns obs8/pred12 at raw stride12 with detector-silver,
image-local data. It is not metric, seconds-level, human gold, physical safety,
true 3D or foundation-model evidence. Stage5C and SMC remain off.

[Failure analysis](failure_analysis.md), [verification](verification.md),
[machine-readable result](summary.json), [research gap](project_gap.md).
