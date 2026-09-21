# Matched-Coverage Execution and Reproduction

2026-09-21. Previous goal turn: progress, completing native cost training and
verification. This turn: a new fixed matched-coverage diagnostic, not a status
repeat, new model fit or independent test.

Registered code/config/design committed as `1be916ed` before new matched-policy
outcome arithmetic. Parent remains `native_gain_harm_v1`; its code and all
models stay frozen. Local arm64 CPU4/inter-op1/workers0. Cached source arrays
fit local memory; no new CREATE allocation or fresh remote-status claim.

```sh
.venv-pytorch/bin/python scripts/run_m3w_native_matched_coverage.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_native_matched_coverage.py
.venv-pytorch/bin/python scripts/run_m3w_native_matched_coverage.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_native_matched_coverage.py
.venv-pytorch/bin/python scripts/audit_m3w_native_risk_target_support.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_native_matched_coverage.py tests/test_m3w_matched_coverage_verifier.py tests/test_m3w_native_gain_harm_verifier.py tests/test_m3w_native_gain_harm.py tests/test_m3w_native_metrics.py -q
```

Preflight PID 72446/session 22078 exit 0: 392 bindings, 12 views, 175,756 indices.
Main PID 72540/session 62638 and replay PID 72554/session 43207 exit 0. Main view
processing/output phase takes about 6 seconds after initial loading. Independent
verifier session 50657 exit 0. Target feasibility session 14957 exit 0. Scoped test
session 95567 exit 0: 37 passed in 1.21 seconds. All required sessions terminal.

Replay recomputes all 144 choice hashes and 24 parent anchors; independent selection
uses a cutoff/partition method rather than the producer's lexsort. Independent
distance arithmetic checks 864 scene reductions and 120 random expected-cost
reductions. No saved prediction, model, threshold or risk tolerance was changed.

Analysis SHA256:
`bc14cefb733bca0b0a461698c1a530d674cb872b7289ebc0b8fa9ed799ac166c`.
The verification JSON files bind exact analysis and verification source code.
Private execution identity/heartbeat/logs are in
`data/stage_cvpr2027_experiments/native_matched_coverage_v1/` and remain ignored.
Re-running with complete outputs verifies byte-consistent summaries rather than
overwriting them. No source data/cache/checkpoints/third-party media are committed.

Uniform expected error is analytic marginal inclusion K/N, not fractional
forecast interpolation. The target audit is an explicitly post-hoc training-only
feasibility analysis, not a new risk-head training result. All public records are
aggregates and hashes. Full future labels are supervision/evaluation only.

No independent calibration, untouched confirmation, online policy, Stage5C or
SMC execution. Pixel/annotation-step claims only. No new publication-ready claim.
