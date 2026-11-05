# Within-Query Paired Decision Learning

Prospective model-loss contrast, 5 November 2026. Original 2% all/easy risk,
signed easy-net, reference/support and missing-label rules are unchanged.
No identity certification. Empty controls/support remain NA, not passed.
The original24-group primary remains NA. Reserved20 Moving, DUT, DroneCrowd
and H stay closed. No new goals, future inputs or central velocities.

## Motivation And Limits

The previous harm floor suppresses more harmful fitting proposals but loses
more useful ones and fails the matched utility comparison. A fresh own-B
ranking audit gives combined AUROC 0.9702/0.8881/0.7272. Cost-weighted ranking
is stronger, but the fixed boundary rejects 44.90/68.10/66.15% of beneficial
fitting mass. SOURCE060 has ordinary AUROC0.6389; SOURCE073 loses99.60% of
beneficial mass. These are fitting diagnostics, not independent performance.

Two-agent sum-pooling can alias actors, but only five such queries and two
opposite-label pairs appear in the largest context; none in the others.
That structural limitation is not established as the main failure mechanism.
This experiment therefore tests discriminative training, not a pooling repair.

## Fixed Arms

All three own-B contexts use the preceding separate64 neural model, 64,260
parameters, 492 causal inputs, same warm initialization, same query draw order,
preprocessing and 2,000-update budget. Last iterate, no validation selection.

1. joint: exact preceding joint objective, a freshly fitted control.
2. paired_point: joint objective plus a pointwise loss on paired useful/harmful proposals.
3. paired_rank: joint objective plus a relative-score loss on the same pairs.

Only observed supported proposals with strictly positive/negative joint targets
enter the extra loss. Zero targets and unknown outcomes do not become negative
examples. Original MSE, individual sign loss and conservative unknown-label
supervision remain unchanged. No support rule is dropped at inference.

In each complete query let p and n be harmful and beneficial proposal indices,
in their existing deterministic row order. At update s, construct max(|p|,|n|)
pairs p[k mod |p|], n[(k+s) mod |n|]. No cross-query pair, no new RNG draw.
This is cyclic pairing, not exhaustive all-pairs ranking or an independent sample.
Labels use the existing mean-channel signed target y; weight w=(y_p-y_n)/2.
Let score=max(risk_all,risk_easy,decision_all,decision_easy).

```
L_point = 4*w*(softplus(-score_p)+softplus(score_n))/2
L_rank  = 4*w*softplus((score_n-score_p)/2)
L_total = L_joint + 0.5*mean_query(mean_pair(L_extra))
```

Queries without pairs contribute zero, not a renormalized larger coefficient.
Both extra losses have identical value and score gradient at zero scores and
the same pair weights. This is a local control, not equal realized loss or
gradients throughout training. The rank term is invariant to shared score
translation; the unchanged joint/risk terms retain the zero-boundary anchor.
Scores are not calibrated probabilities or safety certificates.

## Readout And Scientific Decision

Freeze all nine fits before any additional nested-C development readout.
Keep the existing internal screen: all three paired_rank contexts improve
mean MSE by at least1% against warm initialization, and neither sign-cost
channel worsens. If it fails, no additional held readout. If it passes, read
the same seven already exposed SOURCE units /17views, with every arm frozen.
Primary contrast: paired_rank minus paired_point. Also report joint and original
constant controls, risk-only diagnostics, full unchanged risk/support and NA.
Do not choose an arm or threshold retrospectively from these outcomes.

Report fitting rank discrimination, total sign decision cost, harmful admission
and lost benefit, source breakdowns, known partial-outcome ADE, hard utility,
switch rate, easy-net and positive-harm risk separately. Missing-label completion
bounds are not sampling CIs. No independent test or deployment success is implied.

## Execution

Local arm64 PyTorch CPU4/inter-op1/workers0. Existing local assets and preceding
real fit timings justify a local experiment; no fresh CREATE job is needed.
Historical CREATE receipts are not current scheduler observations. Three
50+50-versus100 recovery pilots, nine full fits, one full treatment refit, exact
predecessor controls and prediction/metric replay. Atomic checkpoints/heartbeat
every100 updates; explicit same-binding resume, no automatic downgrade.
Keep at least5GiB disk free; private artifact cap512MiB. Raw data/weights/cache
stay private. Every code/config binding is committed before fitting.

Only native obs8/pred12 image-coordinate development claims. Not t50, metric,
elapsed seconds, true3D, foundation or human gold. No Stage5C or SMC. This
component experiment cannot by itself establish submission readiness.
