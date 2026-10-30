# Model and data card

- Component: seven-action continuous cost head, not a trajectory generator.
- Architecture: native Torch MLP with static scene and causal detector-history
  branches; width64,30405parameters,462causal fields per candidate.
- Inputs: the frozen motion/scene/rollout descriptors plus64past detector
  history fields. No future position, target latent, future support flag,
  central velocity or test-derived goal enters inference.
- Outputs: benefit, positive harm, reference error, easy-reference mass and
  easy positive harm for each candidate. Benefit/harm obey causal envelopes;
  easy moments are nested. These are learned moments, not calibrated bounds.
- Training change: original plus upper-score loss, with unit current-chosen
  supervised false-admission penalty. Labels supervise loss only. The discrete
  winning action is detached; decoder and2%budget are unchanged.
- Matrix: six SOURCE contexts, seeds17/29/43, three losses,54real complete
  heads. Each uses2000updates,batch128,AdamW0.001,gradient clip5,checkpoint100,
  heartbeat50. All36control full states and one full new refit are exact.
- Data: released EuropeanSquares detector-silver, pixel/raw-frame, eight
  observed and twelve requested steps at raw stride12. This is not the
  supplemental raw-frame t+50 task and is not converted to seconds or metres.
- Data role: SOURCE fitting/resubstitution only;149510unique queries,
  146347with at least one observed future point and3163wholly unknown,
  74recordings,12localities. Overlapping windows/context/seed views are not
  independent samples. Final-step availability differs from observed ADE.
- Upstream trajectory banks and fitting preprocessing remain frozen and
  hash-verified. This study trains no new trajectory producer. The entire
  pipeline is not certified whole-pipeline OOF.
- Unknown outcomes remain in the past-only draw pool, with zero observed-cost
  loss. Completion sensitivity covers wholly unknown rows, not the missing
  positions inside partially observed ADE labels.
- Evaluation:72groups, all144original/upper control prediction/action/full
  score comparisons exact; a separate complete replay is exact.3000locality
  bootstrap draws after context/seed averaging; exploratory conditional CIs.
- Result: primary repair and complete-policy safety fail;46/444supported
  recording views,308known easy violations,42undefined easy views. Do not
  deploy. A small same-head actual-neural-use contrast remains positive but
  is secondary and not independent confirmation.
- Runtime: native arm64 CPU4/interop1,workers0. Full primary training378.32s
  and10.89GB peak RSS. No MPS or remote GPU run is claimed.
- Release: code, config, aggregate metrics, actual training traces and reports.
  Inputs, raw detections, caches, weights, images and third-party assets stay
  private/ignored. A fresh public cold-start rebuild has not been verified.
- Restrictions: no true3D, metric, seconds-level, human-gold, foundation,
  independent generalization, submission-ready or verified physical-safety
  claim. No deployment change, Stage5C execution or SMC.
