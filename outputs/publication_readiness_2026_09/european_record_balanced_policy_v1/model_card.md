# Record-Balanced Neural Intervention Policy

## Intended Use

Research-only TRAIN fitting comparison. This policy chooses whether to use a
fixed neural forecast or a frozen causal reference. It is not a newly trained
trajectory generator, a generative latent rollout or an independently validated
deployment. Deployment remains unchanged.

## Model and Training

- 385 causal inputs; width32 SiLU hidden layer; scalar intervention logit;
  12,385 parameters per policy; threshold remains logit>0.
- Six fixed forecast/controller contexts, three policy seeds17/29/43,
  18 fits and2,000updates per fit; no seed/checkpoint selection.
- Cost-sensitive logistic loss with unchanged source-balanced benefit weights.
  Recording constraint weights change to1/J; coefficient multiplier q_j/w_j.
- AdamWlr.001, weight_decay.0001, gradientclip5, exact hard-record dual update,
  dual learning rate1, cap100, temperature1to.25; original2%budgets.
- Single-process native arm64 Torch,CPU4/interop1,workers0. Atomic checkpoints
  every100updates, heartbeat every50, process lock and explicit resume.

All final checkpoint hashes are in `training_freeze.json` and `fits/`, committed
as7481dfd5before scoring. Weights remain in the ignored local experiment directory;
public manifests are not substitutes for the private weights or authorized data.

## Data and Provenance

European source-TRAIN packets are streamed read-only from CREATE and verified
against packet size, hash and frozen TRAIN manifests. The existing split and
causal feature schema are unchanged. Historical inner-held recordings are
exposed development data, not independent confirmation; no new inner-held score
is computed. Independent selection/calibration/confirmation remain unopened.

The model cannot consume future targets, oracle decisions or central velocity
at inference. Training/evaluation labels are separated from causal inputs.
Record IDs are constraint groups, not inference metadata. This run creates no
new goals or test-derived normalizers.

The observation task is8past/12requested future steps at stride12rawframes in
image coordinates. No seconds, metric scale, homography, true3D or foundation
claim follows from this experiment.

## Measured Outcomes and Limits

Fitting record-budget coverage131/444versus84/444for the parent; known all-harm
violations289versus346. Eligible positive benefit retention80.9251%versus81.5516%.
Full utility changes[-0.135714,-0.133642]pp; matched utility changes
[+0.020396,+0.021290]pp. These are missing-label completion bounds, not CIs or
FDE gains. All72readouts and frozen controls replay exactly.

All444net easy-degradation checks are within2%, yet required all/easy positive
harm and support checks fail.74unique recordings are repeated across contexts
and policy seeds. No independent forecast-seed variance or generalization is
established. A known fixed-candidate support obstruction remains.

## Deployment Decision

**Do not deploy.** The changed objective provides partial fitting improvement,
not complete-policy safety. Preserve previous deployment unchanged and report
the old failures as failures. Neither Stage5C nor SMC is enabled.
