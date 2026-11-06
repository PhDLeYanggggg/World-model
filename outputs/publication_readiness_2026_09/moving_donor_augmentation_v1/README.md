# Does Source-Held Donor Augmentation Improve Risk Learning?

I compared the same risk architecture, original120 causal inputs and fixed
4,096-update budget with own-B donors, producer-A crossfit donors, and a50/50
mixture. The eight query-source folds and original decision threshold stay fixed.

**Information screen: 3/3 contexts.**
Verdict: `donor_augmentation_information_supported_not_deployable`. No deployment change or full-policy safety claim.

## Matched Final Readout

| Context | Arm | Risk MSE | Decision cost | False-safe cost | Cost reduction vs own-only % |
|---|---|---:|---:|---:|---:|
| critic0 | A_only | 1.374481 | 0.003345 | 1.192e-04 | +18.503663 |
| critic0 | augmented | 1.275031 | 0.003254 | 2.414e-04 | +20.710862 |
| critic0 | continued_neural | 3.981874 | 0.003575 | 1.639e-03 | +12.882824 |
| critic0 | own_only | 3.575238 | 0.004104 | 1.824e-03 | +0.000000 |
| critic0 | prior_neural | 1.427056 | 0.003727 | 7.612e-04 | +9.194980 |
| critic0 | shared64 | 1.112450 | 0.003424 | 0.000e+00 | +16.570310 |
| critic1 | A_only | 2.962186 | 0.001152 | 3.852e-04 | +33.105874 |
| critic1 | augmented | 3.054147 | 0.001166 | 4.076e-04 | +32.267086 |
| critic1 | continued_neural | 3.701376 | 0.001777 | 9.770e-04 | -3.237066 |
| critic1 | own_only | 3.667601 | 0.001722 | 9.009e-04 | +0.000000 |
| critic1 | prior_neural | 3.520696 | 0.001737 | 8.050e-04 | -0.900991 |
| critic1 | shared64 | 2.931579 | 0.001052 | 4.298e-07 | +38.902234 |
| critic2 | A_only | 1.451257 | 0.005993 | 1.394e-03 | -5.331895 |
| critic2 | augmented | 1.705709 | 0.005081 | 4.330e-04 | +10.692731 |
| critic2 | continued_neural | 1.719334 | 0.005436 | 1.289e-03 | +4.453619 |
| critic2 | own_only | 1.938243 | 0.005689 | 1.203e-03 | +0.000000 |
| critic2 | prior_neural | 1.331881 | 0.005534 | 1.287e-03 | +2.733869 |
| critic2 | shared64 | 2.000643 | 0.005198 | 0.000e+00 | +8.642633 |

Positive reductions are better. Costs are source-macro known-proposal
signed decision costs, not trajectory ADE/FDE or full-policy harm ratios.
Every arm uses the same original own-B evaluation normalization per source.
Cached forest/prior neural/continued neural controls are separate from the
fresh matched own-only control; training samplers and budgets differ from
the older models. No threshold is fitted to these outcomes.

**The stricter forest comparison does not pass.** Augmented false-safe
cost exceeds shared64 in all three contexts, even where total decision
cost is lower. A better point-risk predictor is not a safe deployment rule.

## Training And Source Coverage

All24models complete4,096updates:98,304primaryupdates. All24are fully refit
before final readout, another98,304updates. The recovery pilot contributes
512additionalupdates and is not reused in the primary fits. One seed only.
The network has9,890parameters. Sampling is source/query-balanced with
replacement, not32epochs; actual distinct-row visits are retained.

| Parent | Source components | Known legal candidates | Unknown legal candidates | Identical-to-floor candidates |
|---|---:|---:|---:|---:|
| A0 | 6 | 319728 | 4008 | 0 |
| A1 | 8 | 416568 | 12936 | 0 |

The full donor treatment includes six fixed candidate amplitudes, unlike
the already-selected own-B candidates. Crossfit teachers also have smaller
source training sets. This is a practical donor-bank treatment, not an
isolated causal estimate of source count. No donor source is excluded
after observing its forecasting error. Unknown futures are not called safe.

## Evidence Boundaries

- fresh_run: new A candidate features/labels,24riskfits,24full refits and final own-B readout.
- cached_verified: six OOF forecast banks, own-B features/labels and old controls.
- not_run: new outer C/H evaluation, independent confirmation or a deployable policy.
- Original2% all/easy harm, signed easy-net and reference/support rules remain.
- Empty references and empty utility controls remain NA. No identity certificate.
- Original24-group main estimand remains NA; old failures remain failures.
- Native8/12image-coordinate silver labels, not verified metric/seconds/true3D/foundation.
- Annotation-generation temporal causality remains unverified. Stage5C/SMC off.

[Per-source results](source_results.csv) | [Loss trace](loss_trace.csv)
| [Failure analysis](failure_analysis.md) | [Reproduction](operation_zh.md)
