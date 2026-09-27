# Fitting-Only OOF Magnitude Study: Data and Model Card

## Intended Use

Mechanism experiment on expected intervention costs, not a deployed trajectory
predictor. Investigate whether better easy-harm ranking from a true cap-event
auxiliary has recoverable magnitude information under the same simple readout
as cost-only and shuffled auxiliary controls. No policy, threshold or forecast
model is changed. This card does not imply the scientific gate passed.

## Data Roles and Units

Cached, checksum-verified European source-development controller data and causal
forecast features. Observation8/prediction12 native annotation steps; detector
image coordinates. Previously exposed source-held localities remain development,
not newly independent tests. Independent selection, reserved calibration and
confirmation roles are unopened. No future endpoint, target-derived input,
central velocity or test-derived goal prototype is introduced.

Each source assignment has four controller localities, disjoint from its
forecast-producer roster. For each outer held locality, three fitting localities
provide inner OOF scores. Three seeds17/29/43 and six producer/controller
assignments produce144 outer views across full and motion-only families.
Views and overlapping windows are dependent, not independent sample counts.
Full and motion-only also differ in forecast/event populations; their comparison
is not a matched input-feature ablation.

## Models and Exclusion

383 causal cost features;GELU64 shared encoder;four nested cost moments;optional
cap-event auxiliary. Fixed2000-update AdamW training with original balanced
fitting-locality sampling. No selected checkpoint, horizon or hyperparameter.
The three-site outer models and432 two-site cost-only inner models are frozen
cached controls. Fresh single-site references supply the two-site auxiliary
training labels without touching either held locality.

The deepest reference has one fitting locality, the inner auxiliary has two,
and the frozen outer head has three. All learned preprocessing, cost scale,
easy thresholds, auxiliary references and label definitions exclude the held
locality at their own level. This changes training population size and easy-cut
definition across levels; it does not establish distributional exchangeability.

The final readout has only two nonnegative origin slopes, fit from inner OOF
predictions with equal supported positive-envelope mass per fitting locality.
Slopes are clipped to[0,8], followed by nested causal-envelope projection.
Denominator moments stay frozen. No probability multiplication, stacked raw
future label, intercept, threshold search or scene-specific deployment rule.

## Evidence and Limitations

Primary is expected easy-harm MSE on positive causal envelopes. It is neither
trajectory ADE/FDE nor easy degradation. Guards cover top10 harm capture,
coverage-log error and all-envelope H_all MSE. Four-locality bootstrap with
three seeds averaged inside locality is descriptive source-development evidence.
Sparse event support, producer training-size shift, target-cut drift, only four
held localities per assignment and prior development exposure limit inference.
Origin regression fits its unprojected objective, not a globally optimized
clipped objective. No formal conformal or physical-safety guarantee is claimed.

Minimum single-site valid cost rows:1225. Some references have zero positive
easy-harm rows. Numerical support is not a power claim; rare-positive strata
remain visible in fitting receipts. Unknown costs are never sampled as zeros.

No true3D,metric,seconds-level,human-gold or foundation claim. Stage5C and SMC
remain off. Artifacts/checkpoints are local; public lightweight reports alone
do not constitute retraining reproducibility without their recorded inputs.
