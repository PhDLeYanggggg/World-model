# Full EqMotion Does Not Resolve the Forecasting Contribution Gap

## Completed Experiment

The registered extension is complete: **twelve new 128-tree cost forests**, twelve
cached-verified neural cost heads, and a fresh common-protocol comparison with
six protected causal motion alternatives and the full Transformer. No EqMotion
predictor or neural cost head was retrained. Their pair-excluded producers already
existed; the earlier inventory mistake is corrected rather than repeated as an
expensive new run. This is not another twelve newly trained world models.

The population remains 175,756 past-eligible pedestrian windows, 33 recordings,
four development-exposed SDD sites and three seeds. Complete future paths exist
for 143,918 windows; 29,039 have partial labels and 2,799 have none. Eligibility and
decisions do not depend on those future masks. The primary estimate is the equal
mean of site-relative available-point ADE gains over causal constant velocity.
It is neither pooled-window gain nor the historical raw-frame t+50 metric.

Each cost head excludes its outer site. The predictor generating each training
cost also excludes that row's site. Forest weights reproduce the paired neural
head's 768,000 actual draws, with zero-disagreement rows contributing no objective
mass. Input, label, preprocessing, support and checkpoint hashes are verified.
All forest fits and decisions completed before the new outer outcome readout.
No threshold, predictor or winner was selected from this comparison.

## What the Matched Results Show

| Fixed policy | ADE gain over CV, CI95 | Hard gain | Worst site/seed easy degradation |
|---|---:|---:|---:|
| EqMotion without control | 11.044% [8.225, 13.398] | 15.401% | 47.392% |
| EqMotion + neural, net-positive rule | 11.006% [8.854, 13.159] | 13.330% | 40.945% |
| EqMotion + neural, strict rule | 1.609% [0.716, 3.039] | 0.527% | 0.452% |
| EqMotion + forest, net-positive rule | 12.417% [9.753, 14.537] | 15.100% | 40.695% |
| EqMotion + forest, strict rule | 1.263% [0.429, 2.735] | 0.052% | 0.000% |

The strict rule is fixed: moving history, nonzero forecast disagreement,
predicted benefit greater than harm, and predicted harm at most 0.1 times benefit.
It removes much observed easy damage but also most of EqMotion's average/hard
gain. Net-positive expected benefit alone is not adequate easy-case protection.
Zero observed easy degradation for the strict forest is not a population guarantee.

With the same strict neural head family, EqMotion is below damping005/010/020 by
2.025, 2.135 and 2.096 percentage points. The nominal paired intervals are negative.
However, those *unmatched* damping arms breach the 2% worst-site/seed easy ceiling;
they cannot be called safely superior. Under matched forest protection, the
EqMotion-minus-damping005 contrast is -0.151 pp, CI95 [-1.721, +1.491].

Full Transformer/neural still gives 2.437% average gain with 1.067% worst-site/seed
easy degradation. EqMotion-minus-Transformer is -0.828 pp, CI95 [-2.148, +0.442].
The point estimate favors Transformer; the interval does not establish a stable
ordering. A more structured public predictor has not repaired the main claim.
The adapted EqMotion evaluation is not a reproduction of its published
best-of-many evaluation protocol.

## Equal Intervention Counts

Both arms keep the smaller strict count per fold/seed, ranked by predicted net
gain within their own strict pools. This is outcome-blind *offline* allocation,
not a query-local deployment rule or risk certificate.

| Neural-head matched comparison | EqMotion ADE gain | Comparator ADE gain | EqMotion easy degradation | Comparator easy degradation |
|---|---:|---:|---:|---:|
| Damping005 | 1.609% | 2.668% | 0.452% | 1.622% |
| Damping010 | 1.609% | 3.101% | 0.452% | 2.692% |
| Damping020 | 1.609% | 3.305% | 0.452% | 2.992% |

Unlike the earlier Transformer-count comparison, the smaller EqMotion count
brings damping005 below the empirical easy ceiling. It has greater mean utility,
but the paired EqMotion-minus-damping005 interval remains inconclusive:
-1.059 pp, CI95 [-2.361, +0.408]. This further weakens an indispensable neural
forecasting claim without proving simple-motion dominance. Counts and selected
population matter; fewer switches are not automatically safer.

All fourteen strict and fourteen matched contrasts, including null and adverse
results, are in [paired_controls.csv](paired_controls.csv). Acceleration/forest's
matched zero-switch result is fallback, not a predictive success.

## What Can Be Claimed

**A supported development observation:** learned gain/harm routing has value.
For the same full EqMotion forecasts, the neural head exceeds its matched forest
by 0.346 pp ADE gain, nominal CI95 [0.217, 0.489], and by 0.475 pp on the common hard
subset. This complements, rather than replaces, the earlier Transformer cost-head
comparison. The neural head has a somewhat worse easy tradeoff than the forest.

**Not established:** neural forecasting is necessary, risk is independently
calibrated, joint decisions have meaningful utility, or scene/image/goal inputs
contribute. No new world-model deployment or submission-readiness claim follows.

**A previous failure remains a failure:** the earlier registered EqMotion
cost-refit primary comparison against transferred frozen heads was negative.
This new comparison neither changes that primary endpoint nor retroactively
turns it into success. The shared easy/hard cutoffs here are those of the
protected-motion study; its hard/easy numbers must not be substituted for the
earlier report's differently defined subsets.

## Verification, Loss and Limits

- Fresh-process inference/aggregate replay is exact across twelve outer views.
- Separately implemented arithmetic verifies 24 fit/reuse budgets, 384 policy
  choices, 168 matched pairs and 4,512 scene reductions. An additional check
  reconstructs all 87 paired ADE/hard/easy contrasts and their bootstrap CIs.
- The scoped suite has 73 passing tests. The full legacy repository suite was
  not rerun. These are same-agent independent implementations, not independent
  research-team reproduction or independently confirmed results.
- [Fitting losses](training_losses.md) and full 16-tree traces are retained.
  Eleven of twelve fits have lower final than initial recorded fitting MSE;
  one does not. Budget completion is not convergence or generalization.
- New fitting-loop time totals 317.433 seconds, excluding loading, between-fit
  overhead and wall-clock interruptions. Native arm64 CPU4/interop1/workers0,
  atomic checkpoints, no sample reduction. All required processes are terminal.
- Origin: `fresh_run` forests and comparative evaluation; `cached_verified`
  predictors and neural heads. New external readout, calibration, confirmation
  and deployment are `not_run`, not zero-valued successes.
- CIs use 3,000 physical-site bootstrap draws. Only four independent source-site
  units exist; three seeds and overlapping windows do not create more sites.
  Nominal multiple comparisons are exploratory, not multiplicity-adjusted claims.
- Observed-point ADE does not impute unknown futures. Complete cases, zero-CV
  harm, tails and per-site full-grid missing-label gain bounds remain in analysis.
- Histories are past-indexed offline annotations; interpolation provenance is
  unresolved. Access exclusion alone does not prove real-time sensor causality.
- Eight observed/twelve predicted native annotation steps, annotation pixels.
  No verified seconds, metric scale, true 3D, foundation, physical safety or
  human-gold label claim. Stage5C and SMC remain off.

## Next Research Action

Do not repeat these predictor fits or search the strict threshold on the exposed
outer outcomes. The next method work should test **predictor-agnostic conditional
gain/harm control** against protected simple motion, with an explicit policy and
independent calibration role, rather than assume that a larger predictor is the
missing contribution. Reuse these now-complete source controls.

DUT is already predictively exposed; its two locations do not provide broad
independent risk support. DroneCrowd remains reserved and closed. Independent
site support, annotation provenance, and weak joint/multimodal contribution are
still research gaps. CREATE's latest read-only refresh failed authentication;
no queue or remote-job status can be inferred from that failure. Local research
continues. **Not yet submission-ready; ultimate research goal unmet.**
