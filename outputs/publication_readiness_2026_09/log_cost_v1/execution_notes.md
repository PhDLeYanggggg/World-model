# Execution and Verification Record

## Material Passport

2026-09-22. Real training and analysis, with cached_verified upstream assets.
Verification status: VERIFIED for the scoped checkpoint, label, weight, decision
and metric calculations; not independent research confirmation or risk calibration.
The same agent wrote both calculations; preprocessing and model forward are shared.
Conditional-quality calculations are checkpoint-replayed, not independently
reimplemented by the separate verifier. These limits are explicit in its receipt.

## Local and Remote Assets

- GitHub main was read-only verified at `173b6ce3ec1fbd82d56efa81258a89db91f74842`
  before this result update. Registration was pushed before the new training.
- The local data and upstream checkpoints are checked against 1,119 source
  bindings. No new raw-data download or source-role change was needed.
- Latest completed fit: native arm64 `.venv-pytorch`, Torch 2.12.0, CPU threads 4,
  interop 1, workers 0; real optimizer updates, no NumPy or backend fallback.
- Local available disk: 53 GiB. All 12 heads fit locally. The 100-step pilot took
  0.1174 seconds for its recorded fitting section and resumed from its checkpoint;
  the full recorded head-fit total was 146.9083 seconds. These exclude the rest
  of the pipeline and cannot be generalized to full world-model training.
- The current SSH config still provides no CREATE alias/project location. The
  [earlier inventory](../asset_runtime_inventory.md) records an authentication
  failure; that is historical, not a fresh authentication result. No current
  scheduler query, remote asset verification or new HPC job is claimed. Remote
  assets remain unknown, not absent. No duplicate HPC job was submitted.

## Completed Commands

Use the native environment from the repository root. Paths below are relative
to that root. Training and replay preserve hashes and reject changed inputs.

```sh
.venv-pytorch/bin/python scripts/run_m3w_log_cost.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_log_cost.py --view coupa_seed17 --stop-at 100
.venv-pytorch/bin/python scripts/run_m3w_log_cost.py --resume
.venv-pytorch/bin/python scripts/run_m3w_log_cost.py --evaluate
.venv-pytorch/bin/python scripts/run_m3w_log_cost.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_log_cost.py
.venv-pytorch/bin/python scripts/audit_m3w_log_cost_turnover.py
```

The turnover command ran twice and produced exactly the same immutable JSON.
Fit, readout, checkpoint replay, separate verifier, diagnosis and its replay all
returned exit 0. No required session remains running. The preserved runner lock
file is not evidence of an active process.

Local arrays/checkpoints/logs remain in the ignored
`data/stage_cvpr2027_experiments/log_cost_v1/` directory. Each trial stores the
model, optimizer, sampler, seed, source identity and completed update count;
`events.jsonl` and `heartbeat.json` record progress and PID. Full-budget resume
checks rather than overwrites existing completed fits. Outputs are not uploaded.

## Scope of Checks

The 35 scoped training/protocol tests passed before fitting and are reused for
their unchanged source version. They include identical forward parameterization,
simplex bounds, zero-distance behavior, target-gradient exclusion, finite extreme
logits, stronger corrective gradients in a synthetic near-zero-harm case, identical
sampler draws, complete-label-only fitting and exact interrupted resume.

New turnover diagnosis checks plus its shared region helper tests:

```sh
.venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_log_cost_turnover.py tests/test_m3w_review_veto_audit.py
```

**19 passed in 0.16 seconds**, including nine new turnover cases. They check
partition completeness, exact added-minus-dropped gain, shared subset denominator,
missing outcomes, zero-reference percentage refusal and invalid input rejection.
No full legacy-suite pass is claimed; its non-hermetic constraints are unchanged.

- [Checkpoint replay](replay.json): 12 endpoints, 527,268 score rows, all checks pass.
- [Separate formulas](independent_verification.json): 1,581,804 supervision-row
  checks across overlapping fit views, 12 weight vectors, 36 policy choices and
  288 scene reductions, plus fixed contrasts, bounds, draws and gates.
- [Turnover diagnosis](turnover_diagnosis.json): 144 group/subset records and
  36 decompositions; old/new site/seed metrics match the frozen result, and their
  equal-site/seed difference reconstructs the primary 0.0910936247 pp contrast.

The analysis SHA256 is
`e344fa115e4ea0a5e73b7a367e691a2a6f2a7527df5ca50e5438f9d78860cf46`.
The turnover diagnosis SHA256 is
`94f772c00bc6861365423adf12e57f96d0eadfc2bbafcd481b40ac45b5540397`.
All receipts and diagnosis bind that analysis, their code and required sources.
The registration is kept unchanged as a before-training snapshot.

## Statistical Interpretation Checks: 11/11 Covered

1. Aggregation reversal: overall easy improvement conceals scene/seed failures;
   both levels are reported, and the scene gate takes precedence.
2. Ecological inference: no safe-individual claim is inferred from scene averages.
3. Selection bias: selected-region harm is explicitly conditional; four scenes
   were development-exposed and are not a random confirmation sample.
4. Collider conditioning: retained/added/dropped and realized easy/hard groups
   are diagnostic partitions, not a causal explanation or inference inputs.
5. Base rates: decision counts, beneficial/harmful complete counts and unknown
   outcomes are reported rather than only selected-case percentages.
6. Regression to mean: the loss has a same-forecast, same-budget frozen control;
   no worst-scene-only post-hoc improvement is promoted.
7. Survivorship: incomplete/unknown outcomes remain counted; complete-cost
   diagnosis is distinguished from supported-ADE and full-grid bounds.
8. Multiple comparisons: one fixed primary comparison fails; other policies,
   objectives and subgroup readouts are diagnostics, not alternative winners.
9. Forking paths: training registration precedes fitting; post-readout turnover
   diagnosis is labelled as such. Thresholds and reference remain unchanged.
10. Causal claims: one training factor was changed, but this does not identify a
    unique real-world cause of failure or demonstrate physical safety.
11. Reverse causality: future-derived gain/easy/hard labels are used for training
    targets and evaluation only; no label-driven new decisions were produced.

The interval crossing zero is inconclusive, not equivalence. With four explored
sites the interval is limited development evidence. Neither the positive CV
contrast nor completion of these checks establishes the requested main contribution.
