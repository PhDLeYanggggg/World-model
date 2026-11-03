# Model And Data Card: Bound-Aware Cost Control

## Material Passport

Fresh real Torch cost-head training, held-SOURCE prediction and separate replay;
cached-verified frozen EqMotion forecasts and causal SOURCE packets. Completed
experiment, unsupported registered primary, failed complete risk contract.

## Model

- 36 width-64, 29,253-parameter same-query-context cost heads: 18 matched jobs
  times known-only/bound-aware targets. Seeds 17, 29 and 43 are actual fits.
- 444 past-only geometry, rollout and detector fields; seven fixed actions.
  Neural action 6 alone can intervene in the primary comparison.
- Outputs B, H, R, ER and EH. Observed targets use native masked ADE; wholly
  unknown bound-aware rows use conservative causal disagreement contributions.
  Unknown zero known-reference mass is not an observed zero-error trajectory.
- The fixed predicted guard supplies a bounded amplitude or exact fallback.
  No learned threshold, epsilon intervention, seed selection or new joint solver.
- 116,160 primary optimizer updates; 4,000 full-refit and 1,200 recovery-pilot
  updates are counted separately. Atomic model/optimizer/sampler/RNG checkpoints.

## Data And Roles

149,510 controller-whitelist windows, 74 recordings, twelve European SOURCE
localities. Each forecaster fits four sites, its critic fits four other sites,
and held evaluation uses the remaining four; both critic-group choices rotate.
All preprocessing and floor/cutoff selection respect those fitting roles.
The held locality is excluded from both models. SOURCE is nevertheless exposed
development, not an untouched final test. No calibration private arrays,
reserved final confirmation or DroneCrowd outcomes are used in this experiment.

Inputs contain no future endpoint, central velocity, eventual track length,
future-valid mask, test normalization or test endpoint goals. Future labels and
availability supervise targets only. Detector/tracker silver is not human gold.
Observation/prediction is 8/12 at raw stride12 in image-local pixels; no physical
time, homography, metric scale or depth is established by this experiment.

## Evaluation And Intended Use

Direct paired completion-lower gain is -1.7031 pp, CI [-5.7081, +1.1166]; known
paired gain is +0.3774 pp, CI [-1.4814, +1.9475]. The primary does not pass.
Bound-aware full-risk support is 108/444 views versus control 28/444, but worst
signed easy-net degradation is 43.6929%, far above the unchanged 2% requirement.
Intervals resample twelve localities 3,000 times after excluded-view/seed
averaging, not overlapping target windows as independent samples.

All 36 cost/amplitude predictions and 72 metric groups replay exactly. Native
arm64 Torch CPU4/inter-op1/workers0: fitting 378.708 s, peak 9.24 GB; separate
verification 124.657 s including 76.350 s inference. This local workload needs
no additional CREATE job. No MPS success or new HPC job is claimed.

Use these models for SOURCE diagnosis only. They are not a deployable upgrade,
population-risk certificate or world-dynamics contribution. No new final model
is promoted. Full cold-start anonymous reproduction and independent method
confirmation remain gaps. Stage5C execution and SMC remain disabled.
