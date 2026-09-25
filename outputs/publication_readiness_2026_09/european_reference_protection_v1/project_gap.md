# Project Gap and Next Controlled Step

## Current Status
The research goal remains active. This round completed real training,
source-held development readout and reproducible negative evidence. It did
not establish a safer deployable neural policy or a submission-ready world
model. Historical test-selected results remain exploratory.

## What Has Been Ruled Out as a Sufficient Repair
The recent registered experiments separately tested a selected-group mean
penalty, importance-corrected rare-harm sampling, matched extra training and
reference-cost protection. None establishes the required transported harm
control. These results do not imply that all possible objectives or models
fail, but repeating the same interventions without a new falsifiable reason
is not useful.

The new information is specific: protecting reference moments yields better
easy-harm MSE on B in 17/18 full comparisons against equal-budget continuation,
yet only 3/18 improve on C. Even on B, selected easy-harm mass becomes more
underestimated than at the original checkpoint. Both selected-population
estimation and transport remain unresolved; merely freezing the denominator
or increasing updates is insufficient.

## Immediate Next Action
Keep forecasts and decision tolerances fixed. Diagnose the residual tail on
the training source before specifying another fit: use B-only score bins and
leave-one-locality-out predictions to distinguish poor ranking of harmful
events from underestimation of their magnitude. Report event frequency,
severity, selected-mass coverage and the influence of the largest errors.
Apply frozen B-defined bins to historical source-C development only for
transport diagnosis, never to choose thresholds or favorable checkpoints.

That diagnosis will determine one registered next intervention: repair the
harm ordering if harmful events are not ranked, or test a selection-aware
one-sided uncertainty estimate if ordering transfers but magnitude does not.
Retain original mean/ridge controls and matched-budget/count controls. Do not
describe a point prediction or a fitted quantile as a risk guarantee. Do not
silently clip high errors, relax the 2% budget, use future support in gating or
open the reserved calibration/confirmation localities to rescue development.

## Risks and Stop Conditions
The largest risks are rare-event support, repeated development exposure,
within-video window dependence and an unbounded error-ratio target. A B-only
cross-locality check can reveal limitations; it cannot create independent
confirmation or guarantee transport to a new locality. Failure to retain any
useful intervention under the fixed budget is a negative result, not a reason
to relabel the same data or change the accepted risk tolerance.

## Before a Main Paper Claim
Still required: accuracy beyond strong simple controls while satisfying the
observed harm constraints; a defensible independent calibration argument;
frozen independent confirmation; matched public strong baselines; interaction
consistency evidence; complete reproducible main results and paper materials.
Architecture composition or a large number of passed software tests does not
substitute for these scientific requirements.

Routine code, experiments, audits and safe Git updates proceed under the
user's delegation. Formal submission remains separate. Only code, configs,
light aggregates and reports are synchronized, not raw data or private model
caches. This is image-pixel, annotation-step, detector-derived research, not
metric/seconds, human gold, physical safety, true3D or foundation success.
Stage5C and SMC remain off.
