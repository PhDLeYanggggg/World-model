# Matched Continuous-Cost Readout: No Learned Neural Increment

Fresh run; cached-verified source forecasts and old binary control. All 54
endpoints were frozen in `905a4ac3` before the first readout. Three head seeds
are not three newly trained forecaster seeds. Independent roles remain closed.

## Primary Development Contrast

On previously exposed whole-recording SOURCE TRAIN inner-held subsets:

| Policy | Known-label ADE gain over fixed floor (%) | Supported recording/head views |
|---|---:|---:|
| Newly learned physical-only | +0.595285 | 30/186 |
| Newly learned physical+neural | +0.553787 | 20/186 |
| Newly learned neural-only | +0.331112 | 32/186 |
| Frozen historical binary control | +3.598939 | 3/186 |

Physical+neural minus physical-only is **-0.041498 percentage points** on known
labels, conditional 3,000-locality-bootstrap CI **[-0.124505, +0.025491]**.
With unknown outcomes retained, the paired completion-lower contrast is
**-0.810703 pp**, CI **[-2.385747, +0.005279]**; the completion-upper contrast
is +0.727708 pp. The data do not establish a positive neural increment.
Same-count lower utility is -0.464531 pp, CI [-1.378516, +0.011413]. It is
utility-only; empty or unknown risk is never credited as a safety pass.

The physical+neural rule chooses 25,478 repeated row occurrences, including
340 without future labels. It has known all-risk violations in 55/186 and
known easy-risk violations in 86/186 recording/head views; easy risk is
undefined in another 22. The 186 views repeat source recordings across contexts
and three head seeds, not 186 independent recordings.

Fixed 4x predicted-harm conservatism improves support to 102/186, still not
the full roster. It is a preregistered sensitivity, not a promoted threshold.
No seed or threshold is selected from this readout. Deployment is unchanged.

## What Failed and What Is Not Yet Established

The problem is not simply that the learner never switches: on fitting rows it
uses the neural action in 18,546 of 54,296 interventions. Yet its fitting
recording support is only 77/444. Fitting failures include 303 easy-harm-upper
violations, 187 all-harm-upper violations and 36 undefined-reference views.
These reasons overlap; they are not disjoint recording counts.

Predicted pointwise 2% conditions therefore fail to deliver the corresponding
observed recording constraints even on fitting data. Mean continuous-cost
regression and local thresholding have not solved the deployed decision
objective. Some additional failures involve unknown-outcome envelopes; removing
unknown rows at inference would be leakage, not a repair.

A specific bounded next experiment is to freeze these cost heads and train a
small causal decision correction with exact whole-record gain/harm constraints,
including positive real easy-reference support. Compare all three banks with
matched settings. This tests decision-objective mismatch; it does not yet prove
irreducible uncertainty, feature insufficiency or generalization failure as the
unique cause. Its protocol must be frozen before training and its own endpoints
frozen before the next readout. The current failed results stay immutable.

## Execution

Pilot PID53293 exits0: 120 real updates, 7.299032s, exact interrupted resume
for all arms. Primary PID53708 exits0: 54 fits, 108,000 updates, 210.034352s,
peak RSS10,093,199,360bytes. All head initializations and sampling streams match
within each context/seed triple. No unknown or inner-held loss row is sampled.

Readout PID54467 and exact replay PID54975 both exit0. All 144 groups and the
summary reproduce exactly. Fifteen focused tests pass; the complete historical
suite is not rerun. Data/weights remain outside Git. No Stage5C/SMC, metric,
seconds-level, true3D, foundation, deployment or submission-ready claim.
