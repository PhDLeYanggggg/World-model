# Cross-Recording Training Experiment

## Question and Scope

Whole-record gradients improved fitting but failed held-recording safety. This
experiment asks whether an update learned on one group of TRAIN recordings can
remain useful and less harmful on another group, compared with an equally costly
same-record meta-update. It changes the TRAIN objective, not the scientific
split, inference threshold, causal features, forecast models or 2% requirement.

The existing inner-held recordings have been exposed in earlier experiments.
This entire comparison is exploratory original-TRAIN evidence, not independent
validation, whole-pipeline OOF, risk calibration or confirmation. Independent
selection, calibration and confirmation roles remain unopened. No deployment
promotion follows from an exploratory positive contrast alone.

Eight of twelve sources have only one fit recording: 020,048,082,110,112,119,124,
126. Their full source-qualified identifiers are in registration.json. A further
same-source disjoint calibration split cannot be made honestly for those sources.
Instead, TRAIN recordings rotate between two training episode sides. Every
recording remains in the objective; neither side is an independent validation
set. No exposed inner-held recording is moved into fitting. Singleton sources
cannot occur on both sides in one episode, so cross-record and some source
distribution differences are inseparable in those episodes.

## Fixed Design

36 fits: six original contexts x head seeds 17/29/43 x two arms. Each gets 2,000
updates. Primary: cross_record_meta. Matched control: same_record_meta. Frozen
references: exact_gradient from european_whole_record_policy_v1, original policy,
and no intervention. No selection among arms, seeds, checkpoints or thresholds.
Both new arms have identical initial parameters, episode identities, neural
architecture, optimizer, row exposure and operation types. Wall time may differ.
Comparison with the frozen exact_gradient is update-, not compute-matched.

The existing 385 -> 32 -> 1 gate has 12,385 parameters and the same frozen binary
anchor. There is no new trajectory forecaster. The 380 causal features and five
frozen moments retain their verified fit-only preprocessing. Future quantities
appear only as supervised loss targets. Unknown outcomes retain their causal
completion envelope; they are not removed or relabeled safe.

Each update partitions whole source-qualified recordings using only identities,
head seed and update number. Multi-record sources are balanced across sides;
singleton sources alternate in a deterministic hash order. No labels or scores
enter partitioning. All eligible fit rows appear once as support and once as
query in each update. All underlying noneligible rows remain in final evaluation.

Let L_A and L_B be normalized, record-weighted Lagrangian losses and W_A/W_B
their original weight masses. Per-row loss is action times
`-(B-H-missing_envelope) + dual_all*risk_all/scale_all + dual_easy*risk_easy/scale_easy`.
Risk terms remain `H-.02R+missing_envelope` and `EH-.02ER+missing_envelope`.
The zero-completion terms are never a substitute for measured outcomes.

Temporary parameters use one differentiable, norm-clipped step:
`theta_A = theta - .01 * clip_norm(grad L_A,5)`, likewise for B.
The objective is half the unadapted full loss plus half the query loss:

- Same-record query: `W_A L_A(theta_A) + W_B L_B(theta_B)`.
- Cross-record query: `W_B L_B(theta_A) + W_A L_A(theta_B)`.

Second-order autograd is retained, including the differentiable clipping factor.
The hard-forward sigmoid straight-through estimator remains a surrogate, not
the derivative of a smooth hard-decision objective or a safety theorem. Temperature
linearly decreases 1 to .25. Outer AdamW lr .001, decay .0001, gradient cap 5.
Per-record duals use exact UNADAPTED current-policy statistics, lr 1, cap 100.
No temporary parameters persist and no adaptation occurs at inference. Fixed
final base parameters produce actions with logit > 0 and original eligibility.

This is an MLDG-inspired symmetric controlled comparison, not a reproduction of
the published algorithm and not a novelty claim for meta-learning. The original
method motivates training against simulated domain shift:
[Li et al., AAAI 2018](https://homepages.inf.ed.ac.uk/thospeda/papers/li2017metaDG.pdf).

## Evaluation

Freeze all 36 final states before any new inner-held score. Evaluate all 144
fixed views: 72 fit and 72 inner-held, retaining every source and recording.
Whole-policy all/easy positive-harm/reference and easy degradation must obey
the unchanged 2% requirement in all required scenes; unidentified ratios do not
pass. Count-matched comparisons measure utility only. Empty matched risk remains
not_estimable. Preserve all historical failures and do not rescue by threshold.

Primary contrasts: cross minus same, cross minus exact_gradient, cross minus
origin. Secondary: same minus exact_gradient. Report full and count-matched
partial-label completion bounds, benefit/harm retention, intervention count,
known/unknown violations, and all source, context and head-seed slices. Those
completion bounds are not confidence intervals or FDE improvements.

Additional fixed uncertainty diagnostic: 3,000 paired source-cluster bootstrap
draws, seed 1342177, applied to source-averaged lower/upper utility endpoints.
Average repeated contexts and head seeds within source before resampling twelve
source clusters. Report endpoint percentiles and an outer bootstrap envelope,
not 3,000 new models, independent rows, or independent confirmation.

## Runtime and Reproduction

Native arm64 .venv-pytorch, CPU four threads, interop one, workers zero. Real
pilot: first registered context/seed, both arms, exact 10+10 versus 20 uninterrupted
updates. Only engineering outcomes determine admission, not held performance.
Resume retains model, optimizer, duals, episode hash and RNG state. Checkpoints
every 100 updates, heartbeat every 50, and process lock prevent duplicate work.
Private checkpoint cap 256MiB, disk reserve 10GiB, runtime limit 12 hours per
invocation; resume the same registered scope if interrupted. No automatic scope
reduction for slowness. No CREATE job is needed unless the measured pilot shows
local resource insufficiency. No Stage5C, SMC, metric, seconds, human-gold,
true-3D, foundation or submission-readiness claim.

Commands from the repository root:

```sh
.venv-pytorch/bin/python -m scripts.run_m3w_cross_record_policy register
.venv-pytorch/bin/python -m scripts.run_m3w_cross_record_policy pilot
.venv-pytorch/bin/python -m scripts.run_m3w_cross_record_policy train
.venv-pytorch/bin/python -m scripts.run_m3w_cross_record_policy evaluate
.venv-pytorch/bin/python -m scripts.run_m3w_cross_record_policy check
```

Append --resume to unfinished phases. Completed phase receipts are immutable.
