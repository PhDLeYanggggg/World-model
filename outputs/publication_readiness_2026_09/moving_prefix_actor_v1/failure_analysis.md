# Actual Model Repair: Early Gain Without Whole-Forecast Success

## Decision

The fixed component progression screen fails. Keep all original rules and
existing deployment unchanged. Do not open a new own-B/C/H test or refit a
controller with old labels pretending they came from the new predictor.

This is seven actual full neural fits,not a threshold sweep:one4000update
old-loss refit and six4000update prefix-balanced trajectory producers.
The latter take7792.307787seconds as one healthy CPU4 process. Full control
state/common trace and the100+100vs200recovery test are exact. Update accounting
must not double-count the reused200update pilot prefix:
the six final models contain24000updates; the control4000; the separate direct
recovery run adds200. Thus28200actual updates including that duplicate pilot.

## What Changed And What Did Not

Only the loss changes:half available12ADE plus half mean supported1/3/6prefix
ADE. Inputs,author EqMotion core,fixed K1 head,seed17,initialization,source-only
normalizers,sampler,draw counts and final4000step selection remain matched.
There is no ordinary residual head,no added future input,no hard-angle veto,
no changed amplitude ramp,no independent test and no deployment change.

## The Hypothesis Is Only Partly Supported

| Prefix | Equal-source paired known gain difference vs old actor | Positive sources |
|---|---:|---:|
| 1 | +12.4391pp | 12/14 |
| 3 | +2.9155pp | 12/14 |
| 6 | +0.6775pp | 7/14 |
| 12 | +0.0976pp | 6/14 |

These differences normalize by each source's fixed query-balanced causal-CV
error mass. They are not pooled pixel errors or independently replicated
scene estimates. Early supervision really changes forecasts in the intended
direction,but the advantage mostly vanishes across the full requested path.
It does not establish improved whole-forecast dynamics or a safe policy.

Producer-fold whole12known differences are+1.3003pp and-0.8044pp. This is not
a stable cross-source repair. Source051 improves11.1957pp,but087 loses4.5909,
082 loses4.3104 and067 loses3.3818. Keeping only the improving sources would
misrepresent the registered whole-population result.

## Unknown Support Is A Separate Problem,Not The Sole Excuse

Retain125540rows:122716have some labels;90965have all12;31751are partial;
2824have no future label. The known paired contrast is only+0.0976pp even
before accounting for wholly unknown rows. The paired completion-lower is
-9.4738pp with exploratory source-bootstrap interval[-17.8089,-3.0825]pp.

This interval describes the predeclared conservative lower statistic,not an
estimate proving that actual unknown future errors are negative by9.47pp.
The unknown-row path-disagreement allowance is large. Source087 alone moves
from known-4.5909pp to lower-56.4978pp with194wholly unknown rows; source052
has18unknown rows and lower-20.3617pp. Low known reference mass can amplify
these ratios. Unknown labels are not silently assigned zero error or dropped.

The failure cannot be blamed entirely on conservative bounds:only6/14known
source contrasts improve,and one producer-fold mean is negative.

## Easy Cases Are Still Harmed By Direct Neural Forecasts

This producer-A cohort has34recordings and no empty known-easy reference at
the whole12diagnostic. Do not import the missing-reference count from the
different previously exposed own-B cohort.

All34new raw-actor recording views exceed2%known-easy positive harm;the old
raw actor exceeds2%on33. Worst signed easy degradation falls504.06%to421.08%
but is still unacceptable. Twenty-eight new raw-actor recording views exceed
2%all-known positive harm,versus29old. These are unbounded error ratios,not
probabilities or physical collision rates,and raw-actor diagnostics are not
the complete selected-policy certificate. There is no safe deployment pass.

## What This Rules Out And What Remains Unproved

- Existing direction information alone is not the missing feature:it was
  reconstructed from raw120on124816comparable rows before this trial.
- This actual prefix-loss repair is not enough. It differs from earlier
  prefix-critic and frozen-ramp negatives,but still fails its whole outcome.
- It does not prove that every loss or architecture must fail. Only one fixed
  mixture and producer seed were tested; no unregistered sweep is claimed.
- It does not establish camera motion,appearance or latent quality as the
  causal explanation. Those remain hypotheses without admitted-input evidence.
- It does not support a new independent generalization or publication-ready
  main-method claim. Source-fold bootstrap is exploratory and training roles
  overlap; overlapping windows are not independent data.

## Next Highest-Value Model/Data Test

Do not relabel the same cosine,retune a test threshold or call prefix1a new
primary success. First test whether a genuinely new past-only observation
is measurable on fitting roles:for example,background-motion estimates from
past image pairs,with explicit image-to-annotation alignment and a prospective
input-admission version. Crowd mean displacement is not verified camera motion.
No images have been newly admitted or used predictively by this trial.

Any such input must first have a reproducible source-only measurement and
matched model comparison. Only a positive whole-forecast component would
justify re-learning gain/harm/reference/support against its actual new OOF
rollouts. Independent C/H and original safety gates remain closed until then.
Keeping the current metric and support rules is part of that decision.

Native observation8/prediction12,annotation stride1,image coordinates only.
No seconds,meters,true3D,foundation,human-gold or physical-safety assertion.
Stage5C execution and SMC remain disabled. The research goal is not achieved.
