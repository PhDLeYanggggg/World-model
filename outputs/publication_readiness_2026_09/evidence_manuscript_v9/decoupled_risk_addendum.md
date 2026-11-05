# Risk-Magnitude And Intervention-Decision Representations

## Experimental Question

We tested whether separate representations for risk regression and cost-aware
intervention decisions improve a frozen predictor-bank policy. The original
2% relative-risk, easy-net and reference-support evaluation contract was retained.
No identity-fallback certification or additional independent-data readout was used.

Three source-separated fitting contexts each received a narrow shared encoder,
a wider shared encoder and two separate encoders. The latter two have64,516
and64,260 parameters, respectively. All models use the same causal492-column
input, preprocessing, conservative missing-label supervision, query draws and
fixed2,000-update budget. Decision scores are logits, not calibrated risks.
The deployed-form decision requires both heads' signs to accept the proposal.

## Fitting Results

| Fitting context | Separate-model MSE reduction | All-cost reduction | Easy-cost reduction |
|---|---:|---:|---:|
| critic0 | 20.42% | -4.13% | 20.54% |
| critic1 | 28.68% | 30.71% | 75.25% |
| critic2 | 10.07% | 20.18% | 84.96% |

Positive reduction means lower fitting error relative to initialization, not a
trajectory metric or independent-test improvement. The main model does not
uniformly dominate the wide shared control: its all-cost is3.57% and12.60% worse
in the first two contexts, and1.26% better in the third. The prospective fitting
expansion screen fails; no additional held prediction readout is performed.

In critic0, harmful-acceptance cost falls75.82%, while false-rejection cost
increases20.54%. The decision branch alone has substantially lower training
sign cost, but the composed policy retains costly rejections from its risk
branch. This suggests an objective/composition mismatch for a subsequent
training experiment, not a justification for discarding the safety branch.

## Scope And Reproduction

All9 real fits complete18,000 updates. A further2,000-update exact refit and
600 pilot updates including repeats verify deterministic recovery. Initial/final
fitting metrics and all9 error decompositions replay exactly. Training used
native arm64 PyTorch CPU4, not a NumPy fallback. This remains a repeated,
one-producer-seed development diagnostic; it provides neither a sampling CI
nor independent confirmation. Existing reference-support limitations and all
earlier negative results remain unchanged.

Dual prediction/rejection learning is established in
[SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html) and
[learning to defer](https://proceedings.mlr.press/v119/mozannar20b.html).
This experiment is not a reproduction of those methods, a new consistency
theorem, or evidence that the multi-head architecture itself is novel.
Task-matched published-method comparisons and independent safety evidence are
still required. Current data use native obs8/pred12 image coordinates; no
metric, seconds, true3D, foundation or human-gold claim. Stage5C and SMC are off.

[Complete experiment](../moving_decoupled_risk_v1/README.md).
