# Cost-Shape Data and Model Card

## Material Passport

Scope: source-development expected-cost estimation. This is a fitted scalar
readout on frozen neural risk scores,not a new neural world-dynamics model.
The previous checkpoint weights,causal forecasts and feature schemas remain
unchanged. The result does not change the deployed trajectory policy.

## Data and Units

Observation8 and prediction12 native annotation steps,detector image pixels.
No verified seconds,metric scale,homography or human-gold annotation claim.
Three model seeds,six overlapping producer/controller assignments and four
outer localities per assignment yield144 dependent source views. No window
count is presented as an independent sample count.

These sources have been used in earlier development. The current fitting
excludes outer outcomes,but that does not restore pristine-test status.
Independent selection,reserved calibration and confirmation remain unopened.

## Inputs and Targets

Inference accepts four frozen cost estimates and a nonnegative causal
forecast-disagreement envelope. It uses H_all/envelope and H_easy/H_all;
zero denominators map to zero. No inference argument accepts future labels,
future endpoint,true goal,central velocity or test endpoint goal clusters.
Targets are harm-cost labels used only inside the fitting or evaluation role.
Known fitting rows with positive envelope are weighted equally by locality;
unknown labels are excluded,not converted to negative or zero labels.

All feature,preprocessing,label-cut and forecast producers retain the parent's
nested exclusion lineage. The entire forecast-producer locality roster stays
disjoint from the controller roster. Row and target hashes are rechecked.

## Model and Fitting

Fixed five-knot monotone functions on[0,1],two harm components in sequence.
Each component enumerates63 possible positive-increment supports. A convex
first-order certificate and constraint checks accompany each solution.
Ten stored knot ordinates per readout,with monotonicity/bounds and optional
mean equalities reducing the free dimensions. No hyperparameter sweep.

shape_L2 and shape_mass share fractions and knots. Their comparison isolates
the additional mean constraint,but the sequential easy caps are different.
Neither is a joint optimization of the two losses. The old origin-scaled
readout is not exactly nested in this fixed-knot parameterization.

## Limits

An empirical fitting mean constraint is not conditional calibration or a
finite-sample transport guarantee. A fitted risk is not a physical safety
certificate. Rare events,limited independent localities,historical source
exposure and label-cut changes between inner producers remain limitations.
Full/motion forecast families have different event populations and cannot
stand in for a matched feature ablation. No true3D or foundation claim.
Stage5C and SMC are off;no generative rollout or new deployment is authorized
by this experiment's numerical checks.
