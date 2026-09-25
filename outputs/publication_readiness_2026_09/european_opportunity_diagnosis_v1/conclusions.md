# Opportunity Diagnosis: Utility Rejection Dominates, Not Absent Headroom

## Material Passport

Completed source-only diagnostic, freshly calculated from cached_verified
forecasts, scores and decisions. No new model fitting or reserved-data access.
The plan and implementation were committed before attribution readout in
`146dbe2f`. This follows, rather than replaces, the failed protected-neural
comparison. All 48 fixed policies and both producer controls remain reported.

## What the Decomposition Shows

The neural candidate is not devoid of potentially useful predictions. Its
hindsight CV/candidate oracle gain is 17.16--17.36%, versus 8.73% for fixed
damping. This is unavailable-future opportunity, not a learned model result
or proof that the gains are inferable from past inputs.

For the previously emphasized easy-event neural-risk policies without the
source-support guard:

| Seed/candidate | Oracle gain (%) | Gross benefit captured (%) | Capture share of oracle | Lost at utility (pp) | Lost at risk (pp) | Switched harm (pp) | Actual net gain (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 17 neural | 17.3638 | 0.2505 | 1.44% | 10.2023 | 6.9110 | 0.0124 | 0.2382 |
| 29 neural | 17.2431 | 0.1794 | 1.04% | 10.5390 | 6.5248 | 0.0131 | 0.1663 |
| 43 neural | 17.1564 | 0.4769 | 2.78% | 10.1689 | 6.5106 | 0.0458 | 0.4310 |
| 17 damping | 8.7335 | 2.0362 | 23.31% | 2.6959 | 4.0014 | 0.2254 | 1.8108 |
| 29 damping | 8.7335 | 2.2133 | 25.34% | 2.5760 | 3.9442 | 0.2745 | 1.9388 |
| 43 damping | 8.7335 | 2.1363 | 24.46% | 2.8689 | 3.7283 | 0.2388 | 1.8975 |

Here pp means percentage points with the same per-locality CV error denominator,
averaged equally over localities. Captured gross benefit minus switched harm is
actual net gain. Missed benefit plus switched harm is oracle regret. These
identities hold for each locality and subset, not just the headline means.
Damping forecasts are deterministic; their identical oracle across seeds is not
three independently trained trajectory models.

The utility stage alone rejects 58.76%, 61.12% and 59.27% of the neural oracle
opportunity. Remaining misses mainly occur at the risk rule. Under the fixed
source-support guard, 5.35--5.48 pp are attributed first to support abstention,
7.62--8.15 pp to utility, and 3.48--3.67 pp to risk. These are ordered gate
attributions, not additive causal effects of experimentally removing gates.
Releasing these rows would also release harmful switches, so this is not
authorization to relax the 2% risk limit.

## Producer Comparison

On exactly the same held source rows, the eight-locality producer loses to the
predeclared lower-index four-locality control by 1.8432%, 2.3447%, 1.4266%, with
conditional 95% intervals [-3.4003%, -0.4170%], [-3.6632%, -1.0750%],
[-2.7919%, -0.2070%]. Against the other four-locality control, gains are
0.5253%, 0.0879%, 0.1277%, with every interval crossing zero.

Both controls are retained; the lower-index control was not selected by score.
All compared producers exclude the held localities. This shows consequential
producer variation, but it does not establish that training on more data causes
harm: source composition, fitted preprocessing/internal baseline and fixed
update budget also differ. It also does not establish how much of utility-head
failure is caused by the four-to-eight producer change.

## A Specific Objective Mismatch to Test Next

The current utility head estimates gain and harm, then subtracts them. However,
its training loss penalizes harm underestimation four times more heavily. That
harm output is not generally an expected-harm estimate. The same conservative
utility is then passed through a separate conservative event-risk head.

An analytic two-outcome regression test makes the distinction explicit: targets
(gain=1.5, harm=0) and (gain=0, harm=1), equally weighted, have mean utility +0.25.
Symmetric squared error favors gain=0.75/harm=0.50. The asymmetric objective
instead has stationary harm=0.80, changing predicted utility to -0.05. This
verifies objective semantics, not that this toy distribution describes the
real data or that a new trained head will improve safety/accuracy.

The next single-factor experiment should change only utility training from
underharm4 to symmetric MSE, for both neural and damping candidates. Freeze the
forecast bank, risk heads, folds, source draws, training budget, support guards
and 2% limit. Compare all fixed pointwise views before considering joint-policy
expansion. No model or threshold should be selected on these diagnostic outcomes.
The utility-loss repair is not_run in this diagnostic package.

## Verification and Limits

All 48 frozen pointwise decisions reconstruct exactly. Their full-population
ADE gains reproduce numerically. All 144 all/easy/hard ledgers and 1,728 locality
accounting blocks conserve rows, benefit and harm. Full diagnostic recomputation
is exact. All 318,969 targets remain, including 7,047 without future labels.
Oracle labels never enter causal reason assignment. Future poisoning tests
change attribution but leave the causal reasons unchanged.

The 3,000-locality bootstrap remains conditional source-development evidence.
This study does not fix prior safety failures, establish joint interaction
benefit, validate independent calibration, or change deployment. Not metric,
seconds-level, physical safety, true 3D, foundation or submission-ready evidence.
Stage5C and SMC are off. Full tables and localities: [tables.md](tables.md),
[summary_metrics.json](summary_metrics.json). Commands: [operation_zh.md](operation_zh.md).
