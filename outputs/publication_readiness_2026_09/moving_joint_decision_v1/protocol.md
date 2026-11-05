# Joint Sign Training Under The Original Rules

## Motivation And Negative Evidence

Fresh own-B diagnosis refutes the previous all-action-training explanation:
each supported row already supervises one frozen router proposal. Opposing
all/easy target signs are absent. The preceding six-fit composed experiment
improves fitting but still fails original held safety in all17 recording views.

Blocking-output attribution identifies a concrete fitting gap. Zero-target
channels alone account for0.002881/0.001674/0.013617 normalized false-rejection
cost in critic0/1/2, about39%/42%/49% of their total false-rejection costs.
Zero labels receive no signed-cost gradient, although a positive output in
that channel still vetoes the complete decision. Other nonzero blockers also
exist. This is not proof that correcting the gap will generalize safely.

Prior proposal-conditioned and joint-event-hurdle experiments already failed.
This is not a rebranding of proposal conditioning or a claim that conjunction
losses are novel. The narrow new test changes the existing two-head auxiliary
loss, not candidate forecasts, sample roles, targets supplied to the risk MSE,
or the scientific evaluation contract.

## Fixed Contrast

Six fits, three original own-B contexts, per_channel versus joint loss. Same
separate64 model (64,260 parameters),492 causal features, initialization,
preprocessing, original conservative unknown labels, proposal masks, source
query draws and producer seed. Warm starts are the same original constant
control models used previously, not the preceding completed composed outputs.
Fresh per_channel controls must exactly reproduce all three preceding composed
weights, optimizer states, draws, traces and predictions.

Keep L0=0.5*risk_MSE+0.5*decision_objective unchanged. For normalized all/easy
target y and outputs r,d, the old added term averages per-channel cost:

    old = mean_channels[4*abs(y)*softplus(-sign(y)*max(r,d))]

The new term uses the full inference sign conjunction:

    p = mean_channels[max(y,0)]
    n = mean_channels[max(-y,0)]
    joint_y = p if p>0 else -n
    score = max(r_all,r_easy,d_all,d_easy)
    new = 4*abs(joint_y)*softplus(-sign(joint_y)*score)

Both use L=L0+0.5*query_average(observed*added_term). Unknown rows add zero
to the auxiliary term and retain original loss coefficients. Positive excess
cannot be cancelled by a negative other-channel target. Exactly all-zero
targets still contribute zero. [-a,0] gives a nonzero safe-sign gradient to
the blocking zero channel. This derived supervision is loss-only, never an
inference input or a replacement scientific risk definition. Changed gradient
scale/routing is part of the intervention, not independently identified.

Deployment still requires both heads and both channels nonpositive AND the
unchanged reference/support checks. No identity certification, tolerance shift,
relaxed2% budget, unknown-label deletion, threshold search or removed head.
The logits are not calibrated probabilities or certified population bounds.

## Fitting And Evaluation

AdamW lr.0001, decay.0001, clip5, querybatch72, fixed2,000 updates and at least
four visits to each fitting query. Final checkpoint, not validation-selected.
All six fits freeze before conditional readout. First full joint fit repeats
exactly. Report every context, full MSE/channel-sign diagnostics and joint
zero/nonzero-block attribution. No branch is promoted posthoc.

Retain the same internal expansion screen: all three treatment contexts must
reduce risk MSE at least1% from their warm starts and not worsen either observed
combined sign-cost channel (1e-12 numerical allowance). Report matched joint
costs separately; they do not replace this screen. If screen fails, additional
held reading is not_run. Engineering training may complete while hypothesis fails.

If screen passes, use only the same seven already exposed SOURCE units /17
recording views, one producer seed, nested C. All predictions/actions precede
outcome access. Readout includes both arms, risk-only diagnostics, original
cached controls with fresh metric checks, sourcewise known ADE/hard utility,
switch, full original risk/support, worst easy-net, matched-control NA and
missing-label completion bounds. Completion bounds are NOT sampling CIs.
All action/metric caches replay exactly. No new H, reserved20 Moving, DUT or
DroneCrowd readout. Original24-group primary stays NA, not passed.

## Execution And Scope

Native arm64 .venv-pytorch, CPU4/inter-op1/workers0, no resource probing.
Local measured predecessor6fits took938s, peak6.78GB. This run fits the48GB
machine; no new HPC job or duplicate training. About11GiB free,5GiB reserve,
new private cap512MiB. Both arms undergo50+50 versus100-update resume/prediction
checks on the largest context. Checkpoints/heartbeat every100 completed steps;
no serialization of half-applied optimizer updates. Healthy <=12h allowed.

This is neural intervention-risk training, not new trajectory dynamics or
JEPA. Native obs8/pred12 image coordinates, not t50/seconds/metric/true3D/
foundation/human gold. No independent confirmation, deployment upgrade or
submission-ready claim is inferred from fitting. Stage5C and SMC stay off.

Run `.venv-pytorch/bin/python scripts/run_m3w_moving_joint_decision.py PHASE`
for register,pilot,train,evaluate,verify. Interrupted phases support --resume.
Code/config/protocol/registration must be committed before fitting. Only
aggregate reports and hashes go to Git; raw data, caches and weights stay local.
