# Cross-Moment Risk Heads

This is a research-only risk-controller experiment, not a new trajectory
dynamics model. Both registered modes must be retained; none is selected for
deployment using opened-development outcomes.

| Item | Description |
|---|---|
| Architecture | Two linear layers, GELU, width 64, 355 causal features, three outputs. |
| Parameters | 22,979 per head; 36 heads per mode, 72 across the two controls. |
| Outputs | Conditional reference-error moment, harm occurrence probability and conditional severity; bounded harm moment. |
| Inputs | Bound past-only features and causal baseline/candidate rollout envelope. |
| Labels | Fitting-only cross-fitted errors; future outcomes are loss/evaluation labels, never inference features. |
| Batch comparison | Cross-moment pair weighting versus preceding realized-share weighting. |
| Fitting comparison | Fixed training-only mean weight sum versus per-batch weight sum. |
| Other losses | Original moment MSE, occurrence BCE and positive-severity MSE. |
| Training | AdamW, 2,000 updates/head, coefficient 1, original sampling sequence and 2% decision rule. |
| Unchanged assets | Forecasts, utility heads, feature scalers, folds, seeds and model capacity. |
| Runtime | Native arm64 CPU4/inter-op1/workers0; atomic checkpoint, heartbeat and resume. |
| Role | Opened source development, four fitting/eight complete-chain-excluded localities per fit. |
| Units | Detector-track image pixels, obs8/pred12 with raw frame stride 12. |
| Risks | Weight concentration, event-conditioned pairing, overlapping observations, inaccurate moments and zero-reference support. |

See both modes' results, training diagnostics and the root conclusions for
observed performance. Lower fitting loss and exact replay are implementation
evidence, not proof of useful dynamics or safe generalization. Conditional
bootstrap intervals are dependent and not adjusted for multiple comparisons.

Historical Stage37 is not recertified. Reserved roles stay closed. No new
JEPA/Transformer fit, metric/seconds claim, human-gold label, physical-safety
guarantee, true-3D or foundation claim follows. No deployment change, Stage5C
execution or SMC is authorized by this experiment.
