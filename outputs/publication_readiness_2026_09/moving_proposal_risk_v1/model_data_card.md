# Model And Data Card

- Model: 108 proposal-conditioned context cost heads, 28,229 parameters each.
- Targets: benefit, harm, reference error, easy-reference error and easy harm.
- Input: unchanged causal 428-column candidate descriptors; shared48 slots zero.
- Output: nonnegative expected cost moments for the seven frozen amplitudes.
- Training: real native arm64 Torch, CPU4/interop1/workers0, 2,000 updates/head.
- Query sampling and initialization match the cached own-part all-action control.
- Frozen proposer: previously trained public cost-sensitive router on disjoint A.
- Cost fitting B and readout V use the original disjoint inner-C partitions.
- Preprocessing and target RMS/bias: cached_verified, own-B fitting data only.
- Causal action mask: fixed before B outcome access, unknown labels not zero harm.
- Data: already exposed MovingDroneCrowd++ SOURCE, 195,286 past-eligible rows in
  the registered pack; 64 admitted recordings, 58 support the current readout.
- Statistics: 23 source units, 276 repeated source/context/orientation views and
  696 repeated recording views; these are not 696 independent scenes.
- Coordinate/time: native observation8/prediction12 annotation order, image pixel
  coordinates. No verified seconds, metric scale, true3D or foundation claim.
- Future labels: loss/evaluation only; no endpoint, future-validity or oracle
  label used to form inference proposals. No new test endpoint goals.
- Availability: partial-label ADE is not full12 ADE; endpoint FDE is diagnostic.
- Evidence: exact optimizer/resume/refit and inference replay; no independent H
  confirmation. New H/DUT/reserved DroneCrowd directions remain closed.
- Limitations: selected-tail miscalibration, missing support, source shift,
  repeated development exposure and no new trajectory or scene-image learning.
- Status: not deployable; no change to the existing deployed policy.
- Stage5C and SMC: disabled. No generative rollout or correction training.
- Distribution: local checkpoints and data only; Git contains aggregate evidence,
  code/configs and hashes, not raw/derived arrays, images or weights.
