# Learned Multi-Action Relative-Cost Experiment

## Material Passport

Prospective SOURCE-development training experiment, 29 October 2026.
The preceding 148/148 feasibility result is label-aware, not learned safety.
Historical failures remain unchanged. No independent role is opened.

## Question and Fixed Contrast

Can a causal cost learner recover useful neural contribution when it can choose
real motion alternatives rather than only one neural prediction or fallback?
The primary contrast is physical+original-ADE-neural versus physical-only.
Neural-only is a matched newly trained binary-action control; the previous
frozen neural-only policy is an additional historical comparator.

Use the six frozen producer/controller contexts and original forecaster seed
43. Keep the original ADE-only neural forecasts, six motion baselines, causal
CV/damping.97 floor, input eligibility, masks, recording partitions, easy event,
and maximum-step disagreement completion envelopes. No-op copies of the floor
are not interventions. At most one real action is chosen per row.

The stronger original producer is fixed from the preceding development result,
not selected using independent outcomes. No new forecaster is trained here.

## Data and Models

Use only the 24 already hash-verified SOURCE TRAIN packets. Fit on their frozen
whole-recording fitting subsets; their old inner-held recordings are exploratory
validation, not a fresh test. Geometry is recovered by exact source, recording,
frame and original row hash. Replay original neural features and labels before
deriving other actions. Retain every required recording and unknown future row.

Each action has the existing 380-dimensional past/causal-rollout features and
a fixed seven-way action identity. A shared candidate-conditioned MLP predicts
continuous benefit, harm, reference, easy-reference and easy-harm moments.
The feature normalizer and initial moment scales use physical actions and
fitting rows only, so the physical-only arm does not consume neural labels or
features. All arms have equal parameter count, initialization, source-balanced
row draws and 2,000 updates, for six contexts x three seeds x three arms =
54 fits / 108,000 primary updates.

The fixed loss is moment regression with 4x underpredicted-harm weight plus
0.5x continuous within-row relative-cost contrast loss. It is not best-class
cross entropy. Causal envelope constraints bound predicted benefit/harm;
unknown outcomes are not imputed into the regression labels. They remain in
inference and all completion-bound evaluation.

## Policy, Roles and Safety

The primary rule selects the available real action with maximum predicted net
gain, only when predicted gain is positive and both predicted all/easy positive
harm ratios are at most 0.02. Otherwise it falls back to the unchanged floor.
Multipliers 2 and 4 on predicted harm are fixed sensitivity policies, not a
relaxed risk budget or an after-the-fact threshold search. Report all three.
Do not choose a successful seed or alter thresholds after seeing the readout.

Evaluate full policies on every required source/recording view. Actual all/easy
harm completion bounds and easy degradation must satisfy the original 2% rule.
Undefined or empty selected risk is not_estimable, not a pass. Count-matched
comparisons within each current scene query are utility-only, never a safety
gate. A fitting or inner-held success is not population safety or deployment.

The primary statistical contrast uses paired completion-lower utility over the
full known floor reference mass, counted once per row. Aggregate contexts and
three head seeds within each source locality, then use 3,000 paired locality
bootstrap draws. Also report ordinary known-label ADE/FDE, intervention and
neural-action rates, regret, per-record failures and calibration error. Do not
pool raw image-coordinate error masses across source domains.

No train/test or metric definition is changed. Selection, independent risk
calibration and confirmation roles remain closed. Any model/policy selection
using this readout is exploratory and must be frozen before later independent
roles are accessed. The experiment does not yet implement scene-joint conflict
optimization; that remains a separate required comparison.

## Execution and Stop Conditions

Native arm64 .venv-pytorch, Torch CPU4, interop1, num_workers0. First run a real
20-update pilot per arm with interrupted 10+10 versus direct20 exact replay.
Use the measured runtime/RSS to estimate all54 fits. Save atomic checkpoints
every100 updates and heartbeat every50. Resume the same run after interruption.
Preserve 10GiB disk and a256MiB compressed-checkpoint cap. Healthy jobs may run
up to12h; slow training alone is not grounds to stop or shrink the experiment.
Freeze all54 final endpoints in Git before any inner-held readout.

No external transfer, raw-frame t+50, seconds-level, metric, human-gold, true3D,
foundation, Stage5C or SMC claim follows from this SOURCE 8-observed/12-predicted
image-pixel experiment. Stage5C execution and SMC remain off.
