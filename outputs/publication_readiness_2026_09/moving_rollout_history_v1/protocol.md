# Causal Rollout-History Relations: Matched Neural Training

The preceding goal turn made progress: exact model diagnostics rejected
gradient projection and fixed neighbor-order averaging as supported repairs.
Those results remain negative. This run tests one new causal feature basis,
not new evaluation rules, a threshold sweep or a claimed new architecture.

## Model Intervention

The original380 active fields contain complete history, neighbor masks and
floor/candidate trajectories plus disagreement summaries. Kinematic64 adds
separate speed, turn and path summaries. Add32 explicit relations: candidate
and floor velocity changes along/across the latest observed heading, endpoint
departures from causal constant velocity, heading sine/cosine agreement,
initial acceleration agreement and speed ratio; candidate-minus-floor
forward/lateral/norm differences at forecast indices1,3,6,12. These forecast
indices refer to model outputs, never future ground-truth inputs. All features
are computed from existing causal fields; no new geometry or label is invented.

Latest zero velocity gives zero directional features, with zero-denominator
ratios defined as zero; existing history speed/stop features remain available.
The basis is rotation/translation invariant within the existing image-coordinate
representation, not metric calibration or a claim of full coordinate invariance.
Neighbor order is unchanged. Original48 shared-motion inputs stay masked.

Add a zero-initialized32-to64 linear branch to each separate risk/decision
encoder. Both arms have68356 parameters and identical warm-start outputs.
The control masks only the32 new normalized inputs. It has the same parameter
count but less effective learning capacity; this is not a pure capacity study.
New normalization uses all own-B causal rows, source-balanced, independent of
future availability/proposals. Existing preprocessing is unchanged.

## Fixed Training And Readout

Three existing seed17 contexts, proposer part1, fitting part0, exposed nested-C
part2. Train six models,2000updates each with the same query sampler, batch72,
AdamW1e-4, weight decay1e-4, clipping5, CPU4/inter-op1/workers0. Use the original
joint objective including unknown-label conservative bounds, not the failed
paired objective. Fixed last iterate; no validation stopping or test tuning.
All six fits freeze before any additional exposed-C predictions. Reserved
Moving20, DUT, DroneCrowd and H stay closed. Original24-group primary stays NA.

Actual pilot: both arms50+50 versus100 uninterrupted, plus initial-function
equality and a masked-control comparison. Checkpoint/heartbeat every100updates,
atomic completed-update recovery,5GiB free-disk reserve and512MiB artifact cap.
Repeat the first full relational fit. Common weights of masked models must
reproduce the previous joint controls; if they do not, diagnose before readout.
No data/cache/model weights enter GitHub. Healthy training may run up to12h.

Use the established fitting-only expansion screen: >=1% mean risk-MSE reduction
against initialization and non-increasing observed all/easy sign cost in every
context. This only permits diagnostic C readout, not safety or deployment.
If it fails, record readout not_run. Report masked-versus-relational fitting
MSE, joint cost and errors regardless. Readout compares relational-minus-masked
partial-outcome utility, hard/easy, switch rate, completion bounds and all
original risk/support checks. Completion bounds are not sampling CIs; one
producer seed and repeatedly exposed C are not independent confirmation.

The original2% all/easy positive-harm, signed easy-net, reference and support
contract remains unchanged. Empty controls/missing support remain NA. No
identity-fallback exemption. Earlier failed studies remain failed. These are
intervention-risk heads, not new trajectory dynamics or JEPA pretraining.
Native obs8/pred12 image coordinates, no metric/seconds/true3D/foundation/human
gold claim. Stage5C and SMC remain off; submission readiness is not established.
