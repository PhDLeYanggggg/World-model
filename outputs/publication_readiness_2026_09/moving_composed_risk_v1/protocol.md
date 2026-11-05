# Learning The Composed Risk Decision

I keep the original 2% all/easy risk, easy-net and support requirements. No
identity certificate, threshold search, new H or reserved Moving/DUT/DroneCrowd
readout. Missing support and empty controls remain NA. The original 24-group
primary remains not estimable, not passed; previous failures are unchanged.

## Prior Evidence And Priority

The previous turn completed nine real models, exact refit and error attribution:
progress, not merely a status update. Separate64 lowered fitting costs in two
contexts but raised all-decision cost 4.13% in critic0. There, false-safe cost
fell 75.82% while false-rejection cost rose 20.54%. Its decision branch alone
fit signs much better than the composed rule. This motivates a training-loss
change, not removal of a safety branch or weakening of evaluation.

Highest-priority model gap: independently trained heads need not minimize the
cost of their conjunction. Separate scientific gaps remain: independently
supported source-level confirmation, multi-seed evidence, matched published
comparisons and a justified method contribution. This diagnostic does not
redefine paper readiness around successful training.

## Fixed Contrast

Six real fits: independent versus composed loss in each of three original
own-B contexts. Both use the same separate64 architecture, 64,260 parameters,
492 causal features, preprocessing, proposal masks, observed/missing labels,
query sampler and seed. Initialize from the same constant-control warm models
as the preceding experiment, not from its fitted separate64 outputs. The fresh
independent arm must exactly reproduce all three previous separate64 models,
optimizer states, draws and predictions. This isolates the added objective.

Let r,d be the normalized risk and decision outputs, and y the unchanged
signed risk target, separately for all and easy channels. Original objective:

    L0 = 0.5 * risk_MSE + 0.5 * decision_objective

Decision objective remains cost-weighted logistic on observed labels and the
original conservative MSE on unknown labels. Treatment adds only:

    L1 = L0 + 0.5 * query_average[
        observed * 4 * abs(y) * softplus(-sign(y) * max(r,d))
    ]

Query averaging and allowed-action denominators are unchanged; unknown rows
add zero to the new term, and original unknown-label loss coefficients remain
unchanged. No target relabeling or target-derived inference feature is added.
The new term changes gradient magnitudes and clipping frequency as part of
the loss intervention; it is not a separate architecture or optimizer test.

The maximum matches the per-channel sign conjunction used by the fixed policy.
It does not directly optimize the full all/easy ratio or support requirement.
At inference, both heads and both channels still must accept, and original
support checks still apply. Logits and the maximum are not calibrated risks
or probabilities. The previously frozen trajectory producer is not retrained.

## Training And Conditional Readout

AdamW .0001, weight decay .0001, clip5, query batch72, 2,000 fixed updates per
fit, at least four visits to every fitting query. No validation early stopping,
hyperparameter search, or branch selection. Save the final iterate; freeze all
six fits before any additional held readout. The first full composed fit is
repeated exactly. Report every fit, not only a favorable context or branch.

Retain the preceding internal expansion screen on the composed arm: each of
all three contexts must improve risk MSE by at least1% and not worsen either
observed combined sign-cost channel (numerical allowance1e-12) versus warm
initialization. Also report the matched composed-minus-independent contrast.
This screen only limits additional development reading; it is not a new
scientific primary or a safety certificate. If it fails, new readout is not_run.

If it passes, evaluate both frozen arms and their risk-only diagnostics on the
same seven already exposed SOURCE units /17 recording views in nested C. Include
hash-verified cached constant/affine controls with freshly recomputed metrics.
Compute all decisions before accessing evaluation labels. Report per-source
known ADE, hard utility, switch, full original risk/support, worst easy-net,
empty matched-control NA and missing-label completion bounds. The latter are
not sampling CIs. No reserved data or independent-confirmation claim. A single
producer-seed development diagnostic is not a formal three-seed primary result.

## Runtime And Asset Checks

Local HEAD and GitHub were both c60deb43 at entry; no live M3W/pytest process
was found. Existing CREATE receipts record completed historical work, not a
current scheduler state. No HPC job is submitted: the preceding local training
took 1,059.52 seconds, peak7.92GB, so this matched six-fit experiment fits the
Apple M3 Max48GB machine. Disk has about11GiB free; reserve5GiB and cap new
private outputs at512MiB. Do not touch unrelated staged work or older caches.

Use native arm64 .venv-pytorch, CPU4/inter-op1/workers0, no resource probing.
Largest fitting context gets actual50+50 versus100-update resume checks for
both loss arms before full training. Checkpoint and heartbeat every100 completed
updates. Never serialize a half-applied optimizer update. Resume model,
optimizer, sampler and RNG exactly. Healthy runs may continue up to12h; slowness
does not justify reducing the target. Record PID, runtime, memory and hashes.
Replay full-fitting metrics and any executed held predictions/actions.

## Claims

This is neural intervention-risk training, not new JEPA or trajectory dynamics.
Native obs8/pred12 image coordinates, not raw-frame t50, metric, seconds, true3D,
foundation or human-gold evidence. Stage5C and SMC remain disabled. Multiple
heads and cost-sensitive rejection are established ideas, not a novelty claim;
see the preceding [method positioning](../moving_decoupled_risk_v1/method_positioning.md).
