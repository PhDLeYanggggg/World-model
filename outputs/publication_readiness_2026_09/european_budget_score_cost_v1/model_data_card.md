# Budget-Score Cost Learning: Model And Data Card

## Intended Use And Scope

Controlled research on the gap between continuous cost prediction and
baseline-relative intervention decisions.Not a deployed trajectory model,
an independent-generalization result or a calibrated risk guarantee.
The completed outcome belongs in[readout.md](readout.md);a lower fitting
objective alone cannot promote deployment or submission readiness.

## Models And One Changed Factor

Six original contexts,three head seeds17/29/43,three frozen forecast banks
and three training objectives define162complete cost-head fits.Each head
retains30,405parameters,462causal fields,width64,batch128,learning rate0.001,
AdamW,gradient clip5and2,000updates.The54original arms must reproduce
the preceding matched-bank experiment's full scientific states exactly.
The first full upper_score/new_neural model also has an independent2,000
step refit.A nine-cell real recovery pilot precedes full training.

The original cost head estimates benefit,positive harm,reference error,easy
reference error and easy positive harm.The two experimental objectives
retain that identified regression plus ranking and add residual supervision
on H-0.02R and H_easy-0.02R_easy.Symmetric and4xunderprediction-weighted
versions separate coupling from directional conservatism.Auxiliary weight1
and physical-fitting RMS normalization are fixed before scores.They are
not confidence levels or risk-calibration parameters.

No model architecture,parameter count,input feature,forecast producer,
normalizer,causal sampler,physical baseline or inference threshold changes.
The three banks are physical_only,old_neural,new_neural;the latter two use
the already frozen actual grouped/partial-neighbor dimensionless producer
and its detector-history version,with producer seed43fixed.The six physical
actions and the original causal CV/damping0.97floor are unchanged.

## Data And Roles

The original SOURCE controller-fitting cohort contains149,510unique queries
from74recordings and12historically explored localities.Some future labels
are partial;3,163queries have no requested future labels.The897,060row views
per objective are not independent queries and must not be multiplied by
three banks to claim a larger dataset.All2,691,180cross-objective row views
are repeated experiment views,not that many independent observations.

Past-only inputs are380motion/rollout fields,18static-region/rollout fields
and64supplied-detector-history features.No future endpoint,remaining-track
length,central velocity,test endpoint goal or future target latent enters
inference.Future labels/masks determine only fitting supervision and offline
evaluation.Unknown rows stay in the original past-only sampler and receive
zero observed-cost loss.No missing outcome is imputed or declared safe.

The exact original input/label identities,preprocessing,producer exclusions
and frozen bank hashes are checked before execution.SOURCE has historical
development exposure;this is fitting resubstitution,not whole-pipeline OOF.
Controller-held outcomes,independent selection/calibration/confirmation and
DroneCrowd outcomes remain closed.No role is restored by renaming a file.

## Evaluation And Reproducibility

Keep all216context/source/seed/objective groups and the full recording
roster.Primary is upper_score/new_neural minus original/new_neural;
symmetric_score and original/old_neural remain controls.The unchanged2%
full-policy all/easy recording rule retains missing-outcome bounds.Empty
same-count utility comparisons and unsupported final-step FDE stay
not_estimable.Same-count utility is not the primary safety test.

Three head seeds and3,000locality resamples provide conditional development
uncertainty,not independent confirmation.All original predictions/actions/
full metrics must replay,then the complete new readout is rerun in a separate
process.Actual training traces and native-coordinate tails are retained.
Finite trajectories and smoothness proxies are not physical certification.

Raw data,large caches and weights stay private;public aggregate archives
preserve numerical JSON bytes.A public checkout alone does not recreate
licensed raw data or all prerequisite checkpoints.No new CREATE job is
required for a run that fits the measured local arm64 CPU environment.

## Restrictions

SOURCE supplied detector-silver bbox midpoint pixels,obs8/pred12/stride12raw
frames.Not historical SDDt50,human gold,verified live sensor-as-of,metric,
seconds,true3D or foundation evidence.No new scene-joint decision training,
JEPA rollout,Stage5C execution orSMC in this experiment.
