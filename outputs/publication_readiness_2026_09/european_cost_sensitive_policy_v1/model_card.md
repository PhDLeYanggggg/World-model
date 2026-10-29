# Cost-Sensitive Intervention Policy

## Intended Use

Research-only binary intervention policy for frozen European-source forecasts.
It is a fitting-objective comparison, not a new dynamics predictor or a
deployable model. The complete recording safety requirement fails.

## Model and Training

- Inputs: 380 verified causal features plus five frozen predicted moments.
- Architecture: 385 inputs, width32 SiLU, scalar logit, 12,385 parameters.
- The logit is anchored to the previous causal policy; intervention uses the
  fixed logit>0 rule and the existing causal eligibility mask.
- Supervised TRAIN targets: benefit, harm, reference, easy-reference and
  easy-harm. Unknown outcomes retain their fixed completion-worst-case envelope.
  Outcome labels, oracle actions and dual target coefficients are not inputs.
- Cost-sensitive logistic primal loss; exact hard-decision record dual updates.
- Eighteen fits: six forecast/controller contexts by head seeds17,29,43.
- Each final model uses2,000updates; no checkpoint/threshold/seed selection.
- Native arm64 Python3.11.1, PyTorch2.12.0, CPU4/interop1, workers0.
- Real pilot verifies exact10+10 resumed versus20 direct updates. Checkpoints
  every100updates, heartbeat every50, process locking and explicit resume.

## Data and Evaluation

74 source-qualified fitting recordings,148record/context views,72 aggregate
source/context/head views and444 repeated record/head views. No new inner-held
evaluation. Independent selection, calibration and confirmation remain unopened.
Underlying source archives also contain historical held rows; the verified
loader filters fitting rows before preparation or scoring. Physical archive
reading is not described as an unopened archive.

No new feature construction was performed. Causality, split and preprocessing
lineage are cached_verified, not newly demonstrated independent evaluation.
No future endpoint, central velocity or test endpoint goal was added.

## Results and Limitations

TRAIN benefit retention81.552%, versus25.998% exact-gradient. Matched TRAIN
utility completion bound[+1.824042,+1.824777]pp against that control. Record
safety84/444, versus178/444. No deployment promotion. No independent CI or test
lift is claimed. Positive-harm/reference constraints are error accounting, not
physical-safety probabilities; net easy error preservation is not sufficient.

This remains a2.5D trajectory research scaffold. No verified metric, effective
seconds, true3D or foundation claim. No Stage5C or SMC. Large private weights,
raw data and caches are excluded from GitHub; public hashes alone do not make
the real-data experiment reproducible without the authorized private assets.

See [results](conclusions.md), [registered protocol](protocol.md) and
[operation guide](operation_zh.md). Research deployment stays unchanged.
