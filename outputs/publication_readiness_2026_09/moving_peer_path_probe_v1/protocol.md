# Directional Peer-Path Information Probe

## Material Passport

Prospective fixed model-side experiment on already exposed SOURCE/own-B data.
Native obs8/pred12 silver image coordinates. No independent confirmation, new
held-out readout, threshold change or revision of the user's original rules.

## Why This Is Not A Repeat

The previous neighbor24 probe tested minimum/mean/endpoint proximity and approach
summaries with linear ridge, and failed in all three contexts. The subsequent
four-cell neural experiment separated candidate identity from peer participation
but also failed its primary fitting screen. Neither establishes that temporal,
directional geometry is predictable across sources with a nonlinear model.

This experiment tests that narrower information hypothesis before another
neural branch. It does not rebrand pooling, random forests or deferral as novelty.

## Features And Controls

Start from kinematic64 + rollout/history32 + neighbor24 (120 fields). Add60:
floor and candidate nearest-peer forward/side displacement for each of12
forecast steps (48), plus candidate-minus-floor nearest radius at each step (12).
Neighbors continue using their last two observed states, aligned to the last
two ego observation times. These are proxy forecasts, not measured futures or
collision labels. Current position and the existing causal scale are retained.

Heading is the last nonzero observed ego displacement; all-stationary histories
use the image x axis. Rotation invariance is therefore claimed only where a
heading exists, not for the all-stationary fallback. Exact nearest-distance ties
average directions. Masked/unaligned peers are excluded; absent peers give zeros
with availability already present in neighbor24. No label-availability feature.

Arms:
1. summary120: verified previous summary representation.
2. aligned180: summary120 plus the ego's directional peer profile.
3. misaligned180: summary120 plus another ego's profile from the same complete
   query, using a fixed one-place ring permutation of sorted row IDs.

The permutation uses all causal rows before filtering for known labels. It
never crosses query/source boundaries. Singleton queries remain unchanged and
their count is reported. The control has equal feature count to aligned180 but
is not an exact causal intervention: donor scales and ego association change.
The120vs180 comparison is not an equal-parameter neural ablation. All cells are
reported, not only whichever is favorable.

## Fixed Fitting Design

Three original own-B contexts, eight leave-source folds. Use known proposed
outcomes only, retain counts of unknown labels without imputing them. All
preprocessing, target RMS and coefficients use only other-source fitting rows.
Weight sources equally, then queries equally, then active known rows equally.
Keep all sources including source065 (only44known proposals in the older audit).
No row-random split. The frozen forecaster lineage is already exposed development;
source-held probe fitting does not restore independent producer generalization.

Primary family: ExtraTreesRegressor,64trees,depth8,min_leaf32,max_features1.0,
no bootstrap,seed17,4threads. Fixed-order tree prediction avoids parallel
reduction drift. No hyperparameter search or source-specific model selection.
Secondary family: ridge0.01, same prior weighted normalization and unpenalized
intercept. All eight summary120 ridge coefficients/predictions must reproduce
the cached prior neighbor24 controls exactly.

48 fitting executions:3arms x2families x8folds. Eight of these are exact repeats
of prior ridge controls, not novel models. Save every completed fit atomically;
resume only missing cells, verify existing identity/inputs and predictions.
A real full64-tree pilot on the largest context must repeat exactly. Estimate
resource cost from that pilot. CPU4/inter-op1/workers0 native arm64, no resource
probing, checkpoint and heartbeat after each cell. A healthy slow fit may run;
do not downgrade the task or report it complete early. Local execution is
reasonable based on the preceding real7.3GB/779s neural study; use CREATE only
if actual resources warrant it. No concurrent jobs are assumed from old receipts.

## Outcomes And Follow-Up

Report source-macro normalized MSE and the unchanged joint sign-cost, false-safe
and false-reject cost separately by context and family, plus every source fold.
Predicted signed costs use the fixed zero boundary, not a searched threshold.

Only primary ExtraTrees can justify follow-up: in at least2/3contexts aligned180
must lower MSE by>=1% versus summary120, not increase joint or false-safe cost,
and have lower MSE and nonworse joint cost than misaligned180. Ridge is diagnostic;
it cannot be substituted posthoc if the primary fails. This is an information
screen, not the deployment safety gate or a claim that three seeds passed.

The original2% all/easy positive-harm, signed easy-net and reference/support
requirements remain unchanged. Empty support remains NA. No identity certificate.
Original24-group primary stays NA; Moving20,DUT,DroneCrowd,H stay closed. No
additional nested-C readout in this experiment. Old failures remain failures.

## Limits And Reporting

Dependent windows do not count as independent observations. Eight source folds
and one random seed are not population confidence evidence. Complete model,
coefficient, prediction and metric replay is computational verification only.
Unknown-label rows remain absent from the information probe, which cannot certify
full-policy harm. No metric/seconds/true3D/foundation/human-gold claim. No new
trajectory or JEPA training. Stage5C/SMC remain disabled. Not submission-ready.

Label actual fitting/extraction fresh_run; prior assets cached_verified;
independent evaluation/neural-branch training not_run. Commit before fitting.
Keep raw data, per-row arrays and checkpoint files private. Reports/figures and
light aggregate metrics may be committed. Use one lock for writers and replay.
