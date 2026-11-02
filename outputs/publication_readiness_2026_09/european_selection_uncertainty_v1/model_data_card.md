# Model And Data Card: Seed-Spread Cost Critic

## Intended Use

SOURCE-only development of a continuous intervention-cost critic. It estimates
benefit,positive harm,reference,easy reference and easy positive harm for seven
fixed forecast candidates. This experiment changes an input covariate,not
the trajectory forecaster,physical baseline family or deployment. A completed
training process is not evidence of a safe or submission-ready world model.

## Inputs And Model

Original444past-only geometry/rollout/detector fields and contemporaneous target
context. New16fields describe spread among three frozen same-fold actual seeds
17/29/43. Predicted paths are legal inputs;observed future,target mask,actual
endpoint,central velocity and test goal construction are prohibited.

A16->64biasless zero projection augments the original context encoder. All
disabled/real/noise arms have30,277parameters and share original initial weights,
RNG,mean_absolute loss,query draws,budgets and output coherence. Disabled uses
the exact original forward;its base optimizer/weights/traces/predictions must
reproduce the old control. No ensemble-mean candidate or new scene/image/goal
input is introduced. Same-query pooling is not pairwise conflict optimization.

## Data And Ancestry

EuropeanSquares detector-derived silver,local-image pixels,obs8/pred12,
rawstride12. SOURCE controller whitelist149,510rows/74recordings/12localities.
Each forecaster fits four sites;each cost head fits the excluded eight. All
three ensemble members share those four producer sites and exclude all cost
rows' scenes. SOURCE rows by fold91,562/145,422/62,036are overlapping views.

Normalization and noise catalogue fit SOURCE covariates only,without label
selection. Noise samples a fixed balanced8x512SOURCE catalogue by hashed row ID;
it is a learned random prior,not current-query evidence or selection-derived data.
Each original query receives at least four visits,without favorable row filtering.

Selection includes all38,102targets/28recordings/six already exposed localities:
37,254known outcomes,15,820partial,21,434complete12and848wholly unknown.
No future-completeness filter changes membership. Selection labels never fit
weights,thresholds or normalizers. Risk calibration/confirmation/DroneCrowd stay
closed. SOURCE fitting resubstitution is not validation or external confirmation.

## Evaluation And Uncertainty

Pre-registered real-minus-disabled paired completion-lower primary and real-
minus-noise negative control. Actual seeds17/29/43 and3000locality bootstraps;
overlapping windows and repeated recording views are dependent. Known secondary
and all arms accompany the primary. Original every-record2%all/easy positive-
harm/reference and easy-net requirements remain;empty reference is not estimable.
Available-label ADE does not identify missing future steps;endpoint FDE is diagnostic.

Seed spread is not a calibrated predictive distribution. Three same-data point
models may share errors,miss irreducible future ambiguity or fail under scene
shift. Lower mean cost loss,high marginal AUROC and low spread are not safety
certificates. Only frozen full-policy performance can advance this candidate.

## Runtime And Access

Local arm64Torch,CPU4/inter-op1/workers0.100step atomic checkpoints,heartbeat,
same-contract recovery and12hper-process boundary. Private data,features,
forecasts,weights and PNGpreviews remain outside Git. Public scalar reports and
derived scientificSVGs do not include third-party raw media. See actual phase
receipts for completion;this card is not a declaration of a passed scientific gate.

2.5D/pseudo-3D trajectory-state research,not metric,seconds,human gold,true3D,
foundation or standalone safe world dynamics. Stage5C/SMC remain off.
