# Crossed Fitting-Regime Data and Model Card

## Purpose

This experiment tests why a frozen expected-harm magnitude readout transports
poorly. It crosses the controller's fitting-row regime and train-derived easy
definition. It does not train a new trajectory forecaster or deployment policy.
Cost-only mechanisms are not proof of the original auxiliary task's failure.

## Data Passport

Checksum-verified European source-development causal forecast features;
obs8/pred12 native annotation steps in detector pixels. Previously exposed
source-held localities remain development data. Independent selection,
reserved calibration and confirmation stay closed. No metric, seconds-level,
human-gold, true3D, foundation or physical-safety claim is supported.

Six producer/controller assignments, three seeds17/29/43, two model families
and four outer localities give144 views. Each view uses all three choices
of omitted fitting locality. The432 replicas are dependent, not432 new
scenes. Full and motion-only have different forecasting/event populations,
so their difference is not a matched feature ablation.

## Producers and Exclusions

All four cells exclude the outer locality from gradient rows, preprocessing
and easy-cut estimation. Their producer sites are also disjoint from the
forecast training roster. In two_cut3, the third site supplies label-definition
statistics but no gradient rows. That third site is therefore not held out
from the entire learning pipeline. This cell may predict the outer locality
only, never provide inner-OOF scores for the third site.

No future endpoint, future goal label, central velocity or test-derived goal
is introduced as an inference feature. Future forecast outcomes define
supervised costs and are used only in their registered fitting/evaluation role.
Unknown-label rows are excluded from gradient sampling, not assigned zero cost.

## Fixed Models

383 causal inputs, a width64 GELU encoder and four nested cost moments;
24901 parameters. AdamW,2000 updates,batch256,learning rate0.0003,
weight decay0.0001,gradient clip5. No selected checkpoint or hyperparameter.
Two crossed cells per replica require864 fresh heads. Native controls reuse
432 two-site and144 three-site models. One additional native bridge is freshly
retrained and must reproduce every parameter exactly.

The row-regime factor includes preprocessing and sampling composition, not
pure numerical sample count. The cut factor includes supervised easy moments,
easy-dependent initialization and RMS loss scaling. Within a row regime,
inputs, known support, draws, fixed diagnostic batch, cost scale and non-easy
loss scales must match across cuts.

The parent cost-only magnitude readout is applied unchanged to all four cells.
Its two origin slopes were fitted on inner-OOF predictions, not current held
outcomes. Raw identities are retained. No denominator is copied from another
cell. This is fixed-readout transport, not optimal per-cell recalibration.

## Evaluation and Limits

Primary: expected easy-harm MSE on positive causal envelopes against the
unchanged outer three-site definition. Guards: top10 harm capture,
coverage-log error and all-envelope H_all MSE. These are not ADE/FDE,
easy degradation, conformal coverage or physical-safety measurements.
Changing evaluation to the inner two-site definition is a separated diagnostic,
not a replacement endpoint. No result is selected using that diagnostic.

All three replica contrasts average within locality/seed, then three seeds
average inside locality.3000 paired resamples of four localities give
descriptive unadjusted intervals. Six overlapping assignments are not six
independent datasets. Missing support is retained, never silently dropped.

Minimum supported fitting rows:6372;minimum positive easy-harm rows:5.
Numerical support is not statistical power. Sparse positives, prior development
exposure, few independent localities and changed target definitions limit
inference. No study outcome changes deployment. Stage5C and SMC remain off.
