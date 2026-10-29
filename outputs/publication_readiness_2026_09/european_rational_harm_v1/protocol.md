# C1 Rational Harm Mapping: One Fixed Matched Repair

## Evidence and Hypothesis

The frozen72-fit-view Jacobian census found54,254/58,707 own-selected row
occurrences with zero easy-harm logit Jacobian in the full-range model.
76.590339/79.541375 of positive easy-harm underprediction lies there; on the same
cohort the isolated-odds control has33 zero-J rows and zero such underprediction.
Hard lower clipping of easy/all ratio accounts for72.924257 (overlapping with
64.310617 at all-harm lower clipping). This supports a local optimization repair,
not a proof of sole cause, convergence or held-out lift. Final shared weights can
still change from other rows. Fit-only evidence chose this repair, not new held
or test results. The broader hypothesis remains unconfirmed.

## Single Change

Replace hard-clipped tanh increments with an anchor-exact rational mapping.
For anchor fraction q and raw logit r:

```
f(q,r) = q + (1-q)*r/(1-q+r), r >= 0
         q*q/(q-r),           r < 0
```

For0<q<1 the mapping is C1 at zero, derivative1 at zero, strictly increasing
at finite r in exact arithmetic and asymptotes to[0,1]. All-harm uses q=H/cap,
cap=envelope-B; easy-harm uses q=He/H and multiplies the changed all-harm.
This changes both geometry and tail approach, not a unique causal isolation of
clipping. Boundary anchors0/1 have unavoidable one-sided flat behavior; float32
output/gradient underflow and very small derivatives can remain. No claim of
globally nonzero parameter gradients. Float64 ratio intermediates; float32
network, optimizer and final predictions. Safe inactive denominators prevent
undefined torch.where branches. No target, label or test information in mapping.
For positive r greater than1-q, evaluate the identical expression as
1-(1-q)^2/(1-q+r) to avoid cancellation in the backward quotient derivative.
This numerical guard was fixed by extreme-anchor unit tests before registration
or any real-data training; it is not a post-outcome variant selection.

## Locked Experiment

18 continuations: identical six contexts x seeds17/29/43;2,000 updates each,
same existing shared-utility anchors, initial weights, protected B/R/Re,
12,258 trainable/25,407 total parameters, source/query draws and sampler hashes,
optimizer/lr/gradient clip5, primary/regret/dual-risk objective and2% tolerance.
No new forecast producer or extra epochs. Real initial objectives must match.
Seven controls: local, anchor, old-risk, utility continuation, joint-risk,
isolated odds and full-range hard-clipped. Last four match total4,000 head
updates including anchors; first three have2,000 and are lower-budget controls.

First run a200-update real pilot (100 direct plus50+50 resume) and verify exact
resume, predictions/protection and finite gradients. Then all18 finals must
freeze before144 fit/held views. Fresh control inference must reproduce cached
hashes/policies. No threshold, model/seed selection or new data partition.
Train/held are exposed internal original TRAIN, not independent confirmation.
Whole TRAIN packet streaming only; outer development/selection/calibration/
confirmation remain unopened. Use local arm64 CPU4/inter-op1/workers0.

## Fixed Readout and Gate

Report every control on every view: signed MSE, selected counts, known and
completion-upper easy risk, unknown/undefined counts, full support. Pairwise
full and equal-count completion-lower utility, normalized by known reference,
not ADE/FDE.3,000 locality bootstrap draws and inherited fixed seed; contexts/
head seeds averaged within12 source localities, nominal exploratory intervals.
Not independent, not multiplicity adjusted, no cross-domain success claim.

Advance requires positive lower95% limits versus ALL seven controls at both
full and matched counts AND full policy support in72/72 internal-held views.
Empty/undefined cohorts never earn a safety pass. This is a further-development
gate, not deployment certification. Preserve negative results. No need to
weaken the old gate to obtain a positive paper result.

## Persistence and Boundaries

Register and commit protocol/code/tests before real execution. Per-fit atomic
checkpoints, heartbeat/PID, explicit resume, single-writer lock,12h budget,
10GiB disk reserve,256MiB checkpoint cap. No new CREATE scheduler job; read-only
approved TRAIN streams with SHA validation. Public aggregates only; private
checkpoints/cache/history never committed. Frozen completed phases immutable.

Obs8/pred12, stride12 raw frames; detector labels, pixel/dataset-local.
No metric/seconds/physical safety/human gold/true3D/foundation claims.
Historical Stage35/37/43/44 remains exploratory. Stage5C/SMC disabled.
