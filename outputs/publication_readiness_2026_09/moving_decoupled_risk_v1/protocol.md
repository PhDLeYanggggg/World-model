# Separate Magnitude And Decision Representations

I keep the original 2% all/easy risk, easy-net and support contract. No identity
certificate, threshold search, new H or reserved Moving/DUT/DroneCrowd readout.
Old failures remain failures, empty controls remain NA, and the original
24-group primary remains not estimable, not passed.

## Hypothesis And Controls

Complete-fitting optimization reduced MSE but increased at least one decision
cost in all three contexts. Signed-cost-only training previously improved its
own fitting sign error but failed held safety. I test whether learning risk
magnitude and intervention signs in separate representations helps retain both.
This is a new neural risk model, not new trajectory dynamics or JEPA.

Three models per own-B context, nine real fits:
- shared64: one width64 encoder and two heads, 32,260 parameters.
- shared128: wider shared encoder, 64,516 parameters.
- separate64: independent width64 encoders, 64,260 parameters.

The main contrast is separate64 versus shared128; separate64 versus shared64
is secondary. Wide shared has 256 more parameters, not exact capacity equality.
Keep original 492 causal features, complete-query other-agent pooling, labels,
conservative unknown-label bounds, proposal masks, query draws, preprocessing
and three own-B partitions. Initialize from the previous final constant-control
checkpoints, not the later LBFGS fits. Widening copies existing neurons, leaves
new feature neurons randomly initialized and gives them zero outgoing weights.
Both heads initially reproduce the warm function within numerical tolerance.
This is a controlled architecture/capacity comparison, not proof that dual
heads are novel. SelectiveNet and learning-to-defer already study jointly learned
prediction/rejection: https://proceedings.mlr.press/v97/geifman19a.html and
https://proceedings.mlr.press/v119/mozannar20b.html. Only their abstract pages
have been reviewed here; no theorem equivalence or novelty claim follows.

## Objective And Decisions

Risk head: original query-averaged normalized signed budget-excess MSE on all
rows. Decision head: 4*abs(y)*softplus(-sign(y)*score) on observed outcomes,
original MSE to unchanged conservative targets on missing outcomes. Average
the two objectives. The decision score is a cost-weighted logit, not calibrated
risk or probability. A channel is accepted only when BOTH outputs are <=0.
Their maximum is used only for this sign conjunction, never as calibrated
risk magnitude. Keep the original all/easy and proposal/support decision rules.

Fixed AdamW .0001, decay .0001, clip5, query batch72, 2,000 updates per fit,
at least four visits to every fitting query. No validation early stopping or
hyperparameter selection. Save the last iterate. First separate64 full fit is
repeated exactly. All nine fits freeze before any additional held readout.

## Conditional Development Readout

Reuse the preceding internal expansion screen, now on the prespecified main
separate64 arm: all three contexts must improve full-fitting risk MSE >=1%
and not worsen either combined observed sign-cost channel (allowance1e-12).
This is a training diagnostic that limits repeated development reading, NOT
a safety rule or scientific primary change. Report all nine fits regardless.
If it fails, additional held readout is not_run, not success or a safe fallback.

If it passes, evaluate every arm and its risk-only ablation on the same seven
already exposed SOURCE units /17 recording views in nested C. Include frozen
constant and affine controls, verify their cached actions and recomputed metrics.
Inference is completed before labels are read for metrics. Report per-source
known ADE, hard utility, switch, all original risk/support, worst easy-net,
matched-control NA and missing-label completion bounds. Those bounds are NOT
sampling CIs. No posthoc winning-arm/threshold selection, independent-confirmation
claim, deployment upgrade or submission-ready claim from this one-producer-seed
repeated development study.

## Runtime And Reproduction

Native arm64 .venv-pytorch, CPU4/inter-op1/workers0. Largest fitting context gets
an actual 50+50 versus100-update exact resume pilot for all three arms. Estimate
full runtime from measured pilot. Healthy local training continues up to12h;
no scheduler job is needed unless pilot resources justify it. Historical CREATE
receipts are not a current scheduler audit; no new HPC job is submitted here.
Atomic completed-step checkpoint and heartbeat every100 updates; interrupted
updates are never saved. Resume from last complete state. Disk reserve5GiB,
private cap512MiB. Record PID, runtime, memory, hashes, real updates and query
coverage. Exact full refit and fitting-metric replay, plus predictions/actions
if held readout is run. Checkpoints and row caches are private, not Git assets.

Native obs8/pred12 image coordinates, not raw-frame t50, seconds, metric, true3D,
foundation or human-gold evidence. Stage5C and SMC remain disabled.
