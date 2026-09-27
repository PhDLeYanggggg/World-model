# Temporal Probe Data and Model Card

## Role and Population
This is a source-development information screen, not an independent evaluation.
Previously admitted European top-down source recordings retain their original
producer/controller assignments. Each view excludes its outer locality and
scores one of the three fitting localities after fitting on the other two.
Localities rotate between these roles across views; they are not globally new
or untouched data. No independent selection, reserved calibration or
confirmation outcomes are opened. Historical contaminated stages are not
restored to independent status by this experiment.

The source adapter uses eight observed and twelve requested future samples,
with raw annotation frame-ID increments of 12. The observation/target interval
is query-84 through query+144. This is not a seconds or metric claim. Coordinates
are detector-box image pixels. Annotation provenance is not human gold.
Missing future labels remain missing; support is not fabricated. The past-only
index is unchanged and does not require future survival for inclusion.

## Inputs and Targets
Only causal risk predictions, forecast-disagreement envelope, eight past
positions, observed box width and up to eight selected past neighbor histories
enter inference. Neighborhood support is restricted to current-visible agents
with complete observed history; it is not full crowd coverage. No future
endpoint, central velocity, test goals or held-data standardization is used.

The four input dimensions are 7, 21, 75 and 143. Motion features use a last
nonzero ego direction and max(history path length, observed width) as scale.
Stationary histories use the image-axis direction; no rotation-invariance claim
is made for those cases. Neighbor features do not affect the history-only
normalization. Time fields must match synchronous past observations.

Targets are pixel-ADE-derived expected easy-harm costs under the fitting-only
easy definition. They are supervision/evaluation labels, never inference
features. The single-locality nuisance heads and two-locality scoring heads
have different fitting sizes and potentially different easy cuts. Explicit
lineage exclusion removes direct scoring-site exposure but does not remove
this statistical transport problem.

## Model and Limits
One fixed regularized linear residual estimator is fitted for each input arm.
Its normalization and target RMS use only supported positive-envelope fitting
rows, equally weighted by locality. The readout retains the frozen all-harm
cap and other risk moments. This bound is an output constraint, not a physical
or conformal safety guarantee. The causal score cap can itself be inaccurate.

Cached-verified Torch models produce nuisance predictions. The 1,728 new fits
are ridge probes with no gradient updates, not neural world-model training.
Capacity differs across input arms; a positive context result alone would not
isolate temporal order from capacity or prove a neural dynamics contribution.
A negative result would reject this fixed probe, not prove that history has
no information for any possible model.

Three training seeds and three excluded-outer contexts are averaged inside
scoring locality before a four-locality paired bootstrap. Six role assignments
overlap. These are exploratory, unadjusted intervals with few independent
localities. Event-track counts and harm-mass concentration remain descriptive;
neither disjoint time intervals nor many agent-query windows establish
independent statistical replication.

Deployment is unchanged. No new trajectory gain, easy-preservation guarantee,
joint-scene safety, true3D or foundation-model result follows from this screen.
Stage5C and SMC remain disabled.
