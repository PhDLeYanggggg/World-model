# Ranking-Auxiliary Failure Analysis

Result source: fresh training of36 Torch risk heads and fresh development
evaluation, using cached-verified forecasts, preprocessing and original heads.
No new trajectory predictor, threshold sweep or deployment policy was fitted.

## What the Controlled Change Shows

The ranking auxiliary is not a reliable repair. Neural/all full-policy ADE
improves over its frozen hurdle controller in7/9 point contrasts, with5 positive
conditional intervals. But at the control's per-locality intervention counts,
only1 ranking interval is positive and5 are negative. At the new counts,1 is
positive and3 are negative. The full-policy improvement cannot be labeled a
general improvement in selecting the right rows.

Against equally protected damping, all18 new neural all-ADE point contrasts
are negative;17 conditional intervals are negative. All18 hard-subset intervals
favor damping. Some easy-subset comparisons favor neural, but they do not
establish an overall or hard-case dynamics advantage.

## Failure Taxonomy

| Category | Observed Evidence | Interpretation and Limit |
|---|---|---|
| Objective does not transport to deployment ordering | Fixed-count neural/all ranking has1 positive versus5 negative intervals at control counts. | Training a pairwise objective does not guarantee better excluded-locality risk ranking. |
| Coverage can mask ordering failure | Neural/all coverage components are positive in all seeds of folds0/1, while fixed-count ranking is inconsistent. Fold2 coverage decreases. | A score-scale change is not the same as a ranking improvement. No unique causal decomposition is claimed. |
| Sparse easy-event pairs | Across all18,000 updates per candidate/event, neural/easy has167,291 valid pair draws; neural/all has2,355,302. Logged neural/easy batches contain0-20 valid pairs versus83-182 for all-event. | This construction provides far less easy-event ranking supervision. It is a plausible source of noisy optimization, not a proven sole cause. |
| Realized-ratio versus conditional-moment mismatch | Pair targets order H/(B+H) from one realized label; deployment uses predicted conditional H/B. | These are not generally equivalent estimands. Original moment losses remain, but the auxiliary may compete with them. |
| Easy preservation is not zero-error protection | New neural heads pass the positive-easy limit in18/18 views but harm zero-CV cases in12/18. | The remaining six views have no zero-CV examples. They do not demonstrate zero-error protection. |
| Auxiliary can make a strong control unsafe | New damping/easy heads exceed2% easy degradation in all three seeds of fold1; worst4.9464%. Frozen damping hurdle passed all18 views. | Large accuracy gains are not enough to promote the modified controller. |
| Source diversity and event support remain limited | Each fit uses four source localities; twelve opened localities are reused across dependent views. | Three seeds and bootstrap intervals do not supply independent confirmation or missing event support. |

## Why This Is Not an Optimization Success Claim

Average first/last logged neural/all ranking losses are0.5204/0.3580, while
neural/easy values are0.4082/0.4428. These are different stochastic minibatches,
not fixed validation losses or proof of convergence. Full traces are retained
in training_metrics.json and training_loss.svg. Training ran to its fixed budget
with finite losses/gradients, but this does not establish sufficient optimization,
calibrated risk or stronger dynamics.

There was no extra-trained initialization: the new and control heads have the
same22,979 parameters and exactly matching sampled rows. No unknown-label row
was sampled. A zero-auxiliary test reproduces the old fitting path exactly.
Thus a capacity or sampling-budget increase is not the intended treatment.

## Next Action

Do not tune the coefficient or threshold against this readout. First repair the
training-pair construction on fitting-only data: form pairs among supported
event rows within each sampled locality, rather than pairing first and discarding
most pairs afterward. Keep original minibatches, moment losses, coefficient,
parameter count and steps fixed so this is one testable change. Inspect pair
support and training-only fixed-batch diagnostics before any new readout.

Independently, zero-reference protection requires a source-supported causal
abstention mechanism; it cannot be created by memorizing the four opened
zero-CV rows, reading their future error at inference, or redefining zero error
after failure. No claim that the proposed repair will work is made here.

Released detector-track pixels,obs8/pred12 rawstride12; not t50,seconds,metric,
human gold,physical safety,true3D or foundation. Historical Stage37 is not
recertified. Reserved roles stay closed. Stage5C and SMC remain disabled.
