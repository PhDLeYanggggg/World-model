# Complete Inference Chain: Frozen and Replayed

2026-09-23. The fixed forecasting and gain/harm banks now have a complete
past-input-to-scene-decision implementation. Registration was committed as
`9925db48` before the new model-chain probes. This is engineering completion of
an uncalibrated research instrument, not independent forecasting evidence.

## What Is Bound

The [policy manifest](policy_manifest.json) binds 693 dependencies, the six
source-only forecasting checkpoints, twelve matching cost heads, architectures,
feature/schema code, source provenance and numerical selection controls. Both
Transformer and EqMotion, both neural and tree cost heads, and seeds17/29/43
remain present. No endpoint, seed, family or threshold is selected in this step.

The strict guard retains the existing exact-past-stop protection, positive net
gain and predicted harm/benefit <=0.1. Full and half eligible counts use the same
predicted-harm anchor for independent ranking, unary geometry and joint geometry.
The whole-scene uniform arm is separate, not a count-matched control. Invalid
model outputs fall back; a failed or uncertified solver solution is marked as a
failure rather than an optimum. These are predicted constraints, not calibrated
realized-risk or physical-safety guarantees.

All visible context is kept. Past-eligible targets receive predictions; context
agents without supported causal CV retain a false forecast-valid mask. Unknown
edges are unpriced and explicitly reported, so their absence cannot be used to
claim collision avoidance. The external adapter and controller do not take a
future target, future-valid mask, oracle class or test-derived goal input.

## Executed Evidence

| Work | Result | Source |
|---|---|---|
| First/middle/last query in each source recording | 99 queries, 33 recordings, four exposed SDD sites | `fresh_run`, input-only probes |
| Twelve model/head/seed views | 1,188 query/view instances | `fresh_run`, no outcome readout |
| Neural target instances across views | 9,396 | Repeated targets, not independent samples |
| Visible agent instances across views | 17,244, including 168 unknown-CV instances | Retained explicit support |
| Illegal numerical model outputs | 0 | Output-bound and finiteness checks |
| Same-count solver comparisons | 0 unmatched query/views | Includes many zero-count cases |
| Separate checkpoint reload and chain replay | Exact forecast/score/decision receipts | `cached_verified` |
| Synthetic external stride1 future-tail perturbations | 12/12 views unchanged | Interface check, not external performance |
| Focused regression suite | 189 passed | Full historical suite not run |

The chain probe loops took 27.708 seconds in total, excluding startup, artifact
loading and binding checks. CPU4 / interop1 / workers0; no NumPy model substitute,
resource probe, training restart or CREATE job. Both run and replay exited0.
Checkpoints remain the preceding verified fitted artifacts; there is no new
training loss or new training in this step.

See [analysis.json](analysis.json), [replay.json](replay.json),
[execution.json](execution.json) and [mechanism_probe_summary.json](mechanism_probe_summary.json).

## Negative Mechanism Evidence

Across all views there are 181 strict-guard interventions and 33 half-count
interventions. Independent, unary-geometry and joint decisions choose the same
agents in these fixed probes. None has a supported non-additive edge at the
half count. This cannot support an interaction-contribution claim.

These 99 queries are interface probes, not an exhaustive opportunity search or
representative accuracy estimate. No future outcomes were opened, so neither
gain nor easy degradation was measured. The result is consistent with the
earlier [sparse-support diagnostic](../protected_joint_support_v1/conclusions.md),
but is not another full-source predictive comparison. The thresholds and pair
weight remain fixed; they were not retuned to force visible differences.

The fitted-cost limitations remain: complete-label selection bias, the shift
from three-site OOF producers to four-site final predictors, coordinate-scale
transport and incomplete context prediction. In particular, using the source
cost scale in the proxy tradeoff is not external scale calibration.

## Remaining Gates

The computational producer/controller chain is now frozen. DUT and DroneCrowd
still have no predictive admission through this step, and their reservation is
not an independence certificate. Source-specific observation/exposure evidence,
supported statistical calibration and one-shot confirmation are not completed.
There are only two author-described DUT locations; 27 recordings do not create
27 independent calibration sites. The 45 DroneCrowd exclusion groups likewise
cannot be counted as certified independent sites.

The next substantive work is to bind that external intake/observation contract
to this frozen chain and execute only the permitted readout. If calibration
support is insufficient, report it and retain fallback; do not inflate sample
size or replace the criterion with an easier undocumented one. No new deployment
or publication-readiness claim is made.

The main task remains8observed/12predicted native annotation steps. SDD stride12
and external stride1 are not established equal physical durations. Pixel or
dataset-local coordinates remain nonmetric/unverified; offline annotations are
not sensor-time perception or human-gold context. Stage5C and SMC remain off.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/freeze_m3w_external_policy_chain.py --audit-only
.venv-pytorch/bin/python scripts/freeze_m3w_external_policy_chain.py --verify
.venv-pytorch/bin/python scripts/summarize_m3w_external_chain.py --verify
```

Verification requires the local fitted artifacts and source input caches. The
manifest, summaries and code are public; raw data, query receipts and model
weights are not committed. The [Chinese operation note](operation_zh.md) explains
the input contract, resumption and limits of using the interface.
