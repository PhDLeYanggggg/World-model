# Project Gap and Next Controlled Step

## Current Status

The research goal remains active. This round completed 144 genuine Torch
risk-head fits and locality-held diagnosis, not a new neural trajectory model
or deployable policy. Reserved calibration/confirmation remain closed.
Historical Stage35/37 and Stage43/44 test-selected or exposed scores remain
exploratory; this diagnosis does not restore their independence.

## What This Round Adds

The uncertainty is now narrower than simply poor generalization. Full all-row
harm-event AUROC of about 0.792 coexists with conditional AUROC about 0.486
on positive forecast disagreement. Yet learned scores capture costly tails
better than a disagreement control in several source assignments. Magnitude
is still unstable and frequently underestimated. Therefore neither 'no useful
ranking' nor 'only a global calibration scale is wrong' fits all the evidence.

## Next Falsifiable Repair

Keep the original mean model, forecast pair and primary risk budget fixed.
Test whether allocating supervision explicitly to positive-disagreement
support improves both within-support cost ordering and harm mass estimation,
relative to an equal-budget mean control. The change must be distinct from
the already tested unconditional hurdle/ranked-hurdle and oversampling arms.
Before training, compare their implemented objectives and document exactly
which examples and losses would differ. If there is no distinct difference,
do not repeat the experiment under a new name.

Use a train-only, locality-excluded design for selecting any nuisance settings;
do not pick thresholds on C or on the 144 held outcomes just read. Keep
original mean and simple causal controls. Check predicted harm on the same
fixed actions as well as model-chosen actions, and match intervention counts.
Test magnitude effects separately from ranking changes. Report support, tail
concentration and failures for all source assignments and seeds.

Predeclare failure if a better all-row score disappears on the actionable
support, if apparent safety merely removes nearly all intervention, or if
accuracy gains again exceed the unchanged harm budget. A fitted quantile or
one-sided estimate is not automatically an independent risk guarantee.

## Still Missing for the Main Claim

- Reliable gain/harm estimates where decisions actually differ.
- Better accuracy than strong simple controls at a matched observed risk.
- Independent locality-level calibration and frozen confirmation.
- Matched public strong forecasting baselines and interaction evidence.
- A main-results package with defensible novelty, limits and reproducibility.

Existing source experiments and negative results can support method development
and limitations, not a claim of submission readiness. No claim of metric time,
meters, human-gold labels, physical safety, true3D or foundation training is
allowed. Stage5C and SMC remain off. Normal implementation/audits/safe Git sync
continue under the user's delegation; formal submission remains separate.
