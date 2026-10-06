# Failure Analysis and Next Causal Check

Source: `cached_verified` complete fresh readout and original training code.
No new fitting, threshold search or independent-role access in this analysis.

## What the Experiment Established

The 216 fits and fixed seven-arm readout completed. The registered advancement
screen failed. Temporal supervision improves all-row signed-score MSE relative
to row-mean supervision by -0.024422, nominal 95% CI [-0.054556, -0.001405], but
its full paired-completion utility is lower by -0.353836 percentage points of
full known reference cost, CI [-0.545292, -0.186078]. The same-count difference
also favors row-mean: -0.231955, CI [-0.372209, -0.112760]. This is not FDE.

Against the no-auxiliary neural head, temporal has lower full paired utility by
-1.026746, CI [-1.557310, -0.574154], and lower same-count utility by -0.567678,
CI [-0.939022, -0.286494]. Its MSE difference against that control is uncertain.

Temporal does improve paired utility over several old controls, including the
original forest (+1.390725 full, CI [0.566962, 2.193647]). That isolated positive
contrast does not excuse failed stronger controls, prediction error or safety.
No new arm is promoted.

## Safety Failure Is Not Just Missing Outcomes

These counts are source/head-seed views, not independent scenes. Known risk is
observed selected easy positive harm divided by observed selected easy reference
cost. Upper risk also accounts for unknown outcomes. Neither is a physical
safety probability. The unchanged budget is 2%.

| Arm | Defined known risk /72 | Known violations | Median known risk | Worst known risk | Extra upper violations while known risk passes |
|---|---:|---:|---:|---:|---:|
| Original | 43 | 4 | 0.601964% | 2.825284% | 3 |
| Additive | 61 | 20 | 0.844794% | 130.802899% | 22 |
| Poisson | 51 | 2 | 0.509856% | 2.825284% | 9 |
| Cost | 48 | 4 | 0.566794% | 2.825284% | 3 |
| No auxiliary | 72 | 60 | 7.382428% | 52.948098% | 12 |
| Row-mean | 72 | 61 | 7.251524% | 56.900595% | 11 |
| Temporal | 72 | 60 | 7.283471% | 62.822896% | 12 |

All 72 temporal full-policy views exceed the completion-upper risk budget; 60
already exceed it on known outcomes. Unknown-label conservatism explains the
remaining 12, not the main failure. Large upper ratios reflect their defined
harm/reference construction and potentially small denominators, not an agent
collision probability. They must not be described as raw trajectory degradation.

The temporal policy selects 83168 repeated occurrences, fewer than the original
forest's 95455, yet has worse safety. This is not simply a larger global switch
count. Same-count controls also fail, implicating which cases are selected.

## Utility Loss Is Not Only an Unknown-Outcome Penalty

Equal-locality means below decompose the full paired lower difference into the
known utility difference minus the unknown XOR-envelope penalty. These means
are descriptive decomposition, not additional independently tested hypotheses.

| Temporal minus comparator | Known utility difference | Unknown disagreement penalty | Paired lower difference |
|---|---:|---:|---:|
| Original | +1.852256 | 0.461531 | +1.390725 |
| No auxiliary | -0.604758 | 0.421988 | -1.026746 |
| Row-mean | -0.124703 | 0.229134 | -0.353836 |

Temporal already loses observed utility to both matched neural controls. It
would be incorrect to explain the negative result solely by unknown labels.

## Failure Taxonomy

1. **Observed decision mismatch.** Better global quadratic prediction error
   versus row-mean does not improve the selected policy, even at the same count.
2. **Observed false-safe outcomes.** Known easy harm exceeds the budget in 60/72
   temporal views. This is a primary failure, before completion uncertainty.
3. **Observed auxiliary tradeoff.** The temporal objective alters a shared
   encoder and improves one regression contrast while degrading neural-control
   utility. This supports a task tradeoff, not a proof of its precise mechanism.
4. **Undefined common-cohort evidence.** Twenty-nine original-selected views
   lack known selected support; their required common-cohort MSE contrast is
   undefined. They remain in the audit and are not replaced with zeros or dropped.
5. **Not established:** which predicted moment causes false-safe selection,
   whether the decoder or loss is responsible, or whether additional features
   would repair it. Those require frozen-prediction component diagnostics.

The existing primary objective is a symmetric query-weighted quadratic loss on
five moments and three signed scores. The auxiliary targets do not directly
optimize the loss of a mistaken safe switch. This is a plausible mismatch to
investigate, not causal proof that changing the loss will succeed.

## Next Experiment Boundary

The next priority is a TRAIN-only diagnostic of predicted easy harm and reference
cost on false-safe decisions, grouped by recording/query and causal support.
Reuse the frozen 216 checkpoints and source packets. Separate harm underprediction,
reference overprediction, near-zero support and uncertainty before registering
one bounded decision-aware repair. Compare against the unchanged no-auxiliary
head, not only older forests. Reuse the original 2% risk definition and fixed
development readout; do not tune this completed run or drop unsupported groups.

An asymmetric false-safe objective is a candidate repair only after that
diagnostic. More epochs or model size alone is not supported by this result.
Independent calibration and confirmation remain closed. No deployment, transfer,
Stage5C or SMC advancement is justified.
