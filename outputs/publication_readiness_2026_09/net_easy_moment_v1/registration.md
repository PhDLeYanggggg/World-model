# Net Easy-Risk Target Experiment

Registered before fitting the new moments or reading their outcomes. This is a
development experiment on four already exposed SDD sites, not independent
calibration, confirmation, deployment selection or a safety certificate.

## Hypothesis

Positive easy harm ignores improvements on other easy targets. Net easy cost
matches the existing net-degradation evaluation more closely, but is a weaker
accounting constraint and may conceal harmed individuals. A better empirical
mean is not proof of individual safety or novel interaction modeling.

## Fixed Design

- Obs8/pred12 native annotation steps, SDD stride12, annotation pixels. No metric
  or seconds claim. No change to endpoint support or primary equal-site ADE gain.
- Three frozen actions: damping005, Transformer, EqMotion. Four excluded physical
  sites and three seeds17/29/43. Thirty-six 128-tree ExtraTrees heads, same source
  draws, features, preprocessing, complete-support labels and forest budget.
- Targets: easy-positive-harm/disagreement, easy-benefit/disagreement,
  easy-CV-error/source-cutoff, easy probability. The first minus second estimates
  signed risk. Labels are used only for fitting and the later frozen readout.
- Keep the old source-predicted net-gain objective and past-only eligible mask.
  No inferred future endpoint, unknown-label filtering or calibration at inference.
- New positive-point vs net-point and positive-population vs net-population use
  the SAME new forest. Include old positive policies because changing multioutput
  targets can also change forest partitions.
- Population allowance is rho=.02 times predicted easy-CV denominator over all
  forecastable targets in ONE recording/frame. No transfer across queries/sites.
  Exact-count net-matched-positive uses the new positive-population count.
- Signed risk may be negative: never remove individually over-budget agents before
  optimization. Exact MILP with original-unit primal/dual/risk checks; failure closes
  to CV. Report solver failures and count failures, never call a fallback an optimum.
- No geometry term in this target isolation experiment. Previous negligible joint
  effect remains a negative result, not erased by an accuracy improvement here.
- Old net/strict/positive-point/positive-population are cached-verified controls.
  Save all nine policies before outcome readout. No test threshold or arm selection.
- Report all arms, seeds, sites, hard/positive-easy/zero-CV/complete subsets,
  missing labels, tails, coverage, partial-label gain bounds and paired 3000
  physical-site bootstrap intervals. Four sites give limited exploratory uncertainty.

## Falsification and Operations

Net vs positive has useful development evidence only if accuracy increases without
positive-easy net degradation above2%; zero-CV harm and worst-site failures remain
separate veto information. Compare old strict too, not only an overly conservative
control. These are diagnostic comparisons, not policy promotion gates.

First run a16-tree native-arm64 CPU4/workers0 pilot, then resume128 for all36 heads.
No outcome-based early stopping or replacement seed. Checkpoint every16 trees and
256queries; log PID and heartbeat. Existing equivalent checkpoints occupy667MB;
50GiB free was observed before this run. No confirmation source is opened.

DroneCrowd remains closed confirmation data; DUT remains exposed diagnostic;
HT21/CroHD remains quarantined. Stage5C, SMC and deployment remain false.
