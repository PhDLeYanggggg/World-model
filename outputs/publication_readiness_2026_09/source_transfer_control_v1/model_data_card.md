# Source Modality Control: Model and Data Card

## Intended Use

Exploratory diagnosis of optimization and visual-input utility in stationary-history
trajectory forecasting. Not a deployment release, a new Stage37 claim, an
action-conditioned world simulator, or proof of cross-dataset generalization.
The main approved8-observation/12-prediction protocol is not replaced.

## Predictors

The inherited `SourceDynamics` head has44,864 parameters. It combines a small
past-image/coverage CNN with supplied past geometry and predicts12 two-dimensional
offsets. This specific control does not train a new Transformer or JEPA model.
The output is bounded in an observed local frame and restored with past-derived
radius/rotation. Unsupported contexts return the stationary baseline exactly.
The bound is a numerical/context constraint, not a physical collision guarantee.

All candidate models use the same480-dimensional input schema and network size.
The coverage-only arm zeroes RGB but retains geometry, frame rotation and image
coverage. Mask/RGB parent fits already differ in learned weights; they share
initialization scheme, data and sampled rows, not identical trained parameters.
Constant/cosine continuations within each modality share the same trained parent.

Three seeds17/29/43, uniform sampled source rows, batch64, AdamW, clipnorm5,
ADE loss with a training-only scale. Each of six newmask branches continues
from2000 to10000updates. ConstantLR0.0003; cosineLR ends0.000003. Six matching
RGB endpoints were trained in the previous registered experiment and are reused
after hashverification. No new architecture, feature, threshold or seed search.

All18 parent/final states are evaluated. The secondary classifier guard keeps
its previous0.9threshold and same-modality/source-excluded probability producer.
Those probabilities predict annotated future change, not verified human intent
or calibrated forecast gain/harm. Guarded modality comparisons change both
trajectory output and probability producer, so only the uncontrolled contrast
isolates the paired predictor-modality comparison.

## Data and Provenance

Only originalSDDtrain40 is admitted. The complete stationary-history source
cohort contains22,374queries across5physicalsites. The training complement here
uses15,430queries,545scopedagents,29recordings and4sites. The excludedbookstore
diagnostic contains6,944queries,181scopedagents,7recordings and1physicalsite.
Bookstore was evaluated in earlier project experiments; it is not independent
confirmation. The historical Stage37/43/44 lineage limitations are not cleared
by this source experiment.

The observed8points and future12targets use stride12source frames. Future targets
are read only for losses, complete-label cohort construction and evaluation.
Observed features, image crops, coverage and restoration frames are supplied
past annotations, including generated/interpolated annotations. This is not a
strict sensor-as-of measurement study. Missing future labels are not negatives.
This source population is not a substitute for the complete main benchmark.

Training-derived normalization, optimizer scale and hard cutoff are fixed before
held scoring. No held endpoints construct goals, no future endpoint enters a
predictor, and no held-label statistic selects a threshold. Main development,
calibration and confirmation roles remain closed during this diagnostic.

## Units and Safety

Report parent-normalized coordinates and native annotationpixel errors separately.
Source horizons are +144rawframes for12futuresteps, not audited seconds. No meter,
true3D or foundation claim is justified. Zero-target cases have zeroCV error;
their relative degradation isundefined. A small absolute error does not by itself
pass a2%relative-easy safety criterion. Bounding offsets is not sufficient safety.

All2000 bootstrap draws resample the7recordings within this one exploredsite,
after averaging three seed errors perquery. The interval is conditional and
does not establish independence among recordings, new physicalscene coverage,
or a valid confirmatory decision after historical exploratory reuse.

## Release and Reproduction

Code, configurations, aggregate metrics and original statisticalSVG are public.
Rawdata, imagecaches, weights and perquery predictions remain private local
artifacts, excluded from Git. Asset hashes allow exact validation where the
underlying data are lawfully available; they do not make a code-only clone a
complete runnable release. See [commands](reproducibility.md) and
[Chinese operations](operation_zh.md). Results and gates are separate from this
prospective modeldescription. No Stage5C execution or SMC is authorized.
