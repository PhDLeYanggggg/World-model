# Fixed norm-parity admission repair

## Evidence-driven choice

All19complete instrumented replays preserve the previous scientific states
exactly. The18admitted trajectories activate the auxiliary on33771/36000steps,
first at22-90steps. Its median parameter-gradient norm is only0.003632%of the
anchor norm (p90:0.021178%, maximum0.185241%). Its easy-channel median is
0.001258%. Thus missing activation is not supported. The tested repair is
norm balancing, not removal of the risk gate or a threshold sweep.

## One change, fixed before scores

Retain all model, data, preprocessing, draw pools, seeds, update budgets,
candidate trajectories, five-moment/ranking/upper-score anchor losses and
current-admission auxiliary. For each train batch, calculate non-accumulating
parameter gradients of anchor A and admission auxiliary U. Use the detached
coefficient `lambda=min(1e6, ||grad A||/||grad U||)` when anchor norm is positive
and auxiliary norm exceeds1e-12; otherwise use0. Optimize `A+lambda U` with
the same AdamW and gradient clip5. Target ratio is fixed1, no weight sweep.
The numerical cap, zero-scale steps and actual achieved gradient ratios are
reported. Norm parity is pre-clipping only; it does not establish AdamW
descent, moment calibration or complete-policy safety. No inference coefficient
or threshold changes. Training labels supervise losses only.

## Matrix and exact controls

Train18fresh balanced heads (6contexts x3seeds), each2000updates, width64,
batch128,30405parameters,462causal inputs. Primary updates36000. Reuse36
frozen upper/unit heads as cached_verified matched controls; do not describe
them as newly trained. Common initialization, input/preprocessing hashes,
sampler RNG and draws must match. Run a real100+100/direct200pilot (400actual
updates) whose balancing mechanism must activate, not a20step smoke test that
precedes observed activation. One complete2000step refit must be exact.
All18endpoints freeze before the fixed72group readout. No early stopping on
scores, changing cohorts, adding loss variants or test-selected winner.

Primary: balanced minus unit-admission full-policy known and wholly-unknown
completion-lower utility. Secondary: balanced minus upper and same-balanced-
head physical-only action restriction. Recompute all predictions and require
both cached controls' prediction/action/full scores to match exactly.3000
locality bootstrap draws after context/seed averaging. Retain the complete
2% selected-reference all/easy recording risk contract, undefined support,
unknown outcomes, hard/easy slices, tails and shared-bank regret. Count-matched
controls remain utility-only; empty is not estimable. No primary substitution
from a positive secondary. These are exposed SOURCE fitting results, not
whole-pipeline OOF, validation-selected independent evidence or confirmation.

Native arm64 CPU4/interop1/workers0,checkpoint100/heartbeat50,resume,12h bound,
10GiB reserve,128MiB private checkpoint cap. The19instrumented runs completed
in185.81s at10.97GB peak RSS, supporting local execution. Measure the real
balancing pilot before full fits. No unnecessary duplicate HPC job.

Only code/config/reports/aggregate metrics/traces are public. No raw data,
feature/latent cache, weights, images or third-party assets. No deployment,
submission-ready, metric/seconds, true3D, foundation, Stage5C or SMC claim.

## Prior-work boundary

Gradient balancing is established; this is a controlled simplified norm-parity
repair, not a claim to invent gradient normalization or reproduce full GradNorm.
[GradNorm](https://proceedings.mlr.press/v80/chen18a.html) learns task weights
using gradient magnitudes and relative training rates. Our coefficient has
no learned task-weight optimizer or relative-rate target.
[SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html) already couples
selection and prediction with an auxiliary training head; we do not claim
selective learning itself as new.
[Regression with Multi-Expert Deferral](https://proceedings.mlr.press/v235/mao24d.html)
already covers regression routing over pretrained experts. None of those
results supplies a safety theorem for our selected harm/reference ratios.
