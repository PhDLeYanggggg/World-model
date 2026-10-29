# Full TRAIN Census Before Another Risk-Loss Repair

The fixed128-query monitors may not represent full TRAIN risk. This diagnostic
separates monitor sampling, empirical fitting and held-recording generalization
before another training intervention. It does not change the2% scientific risk
definition, forecasting task, independent roles or any deployment threshold.

Evaluate all288 frozen cost heads:72 each quadratic, utility, unit-risk and scaled
risk. All source TRAIN rows are retained at inference, including wholly unknown
labels. Actions use only causal features, support and predicted costs. Aggregate
known and completion-bound risk, five predicted/observed moments, rare easy-harm
concentration and per-recording metrics. Compute query-mean and pooled-agent
excess over the complete known TRAIN set, not128 monitor draws.

Replay the exact2,000-step sampling trace once per matched identity; report query
coverage and whether positive-easy-harm rows were ever sampled. There is no
optimizer update. Compare with the immutable prior72-view development aggregates,
marked cached_verified; no new development inference or label access is allowed.
The independent selection/calibration/confirmation roles remain unopened.

If full TRAIN risk fails despite the monitor passing, prioritize estimator,
monitor coverage and tail fitting. If full TRAIN is substantially better than
the held-recording development aggregates, prioritize harm generalization and
TRAIN-internal recording splits. These are diagnostic decision rules, not claims
of identified causation or permission to pick a policy after reading results.

Use native arm64 CPU4, workers0, bounded packet streaming, hash-verified models,
group-level atomic receipts, heartbeat and explicit resume. Stop on changed
inputs. No raw/cache/checkpoint files enter Git. This is local work; no CREATE
training job is submitted. Stage5C and SMC stay disabled; no metric/seconds,
true3D, foundation or physical-safety claim follows. Historical results remain
exploratory and this experiment does not restore their independence.
