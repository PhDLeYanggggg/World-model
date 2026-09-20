# Frozen Risk Forensics: Reproduction

Date: 2026-09-20. Native arm64 `.venv-pytorch`, local read-only model artifacts.
No CREATE job, neural fit, new inference, deployment or new evaluation cohort.
The diagnostic uses NumPy/Python; that is risk accounting, not a neural fallback.

## Commands

```sh
.venv-pytorch/bin/python scripts/audit_m3w_frozen_risk.py --pilot-candidates 1
.venv-pytorch/bin/python scripts/audit_m3w_frozen_risk.py --resume
.venv-pytorch/bin/python scripts/report_m3w_frozen_risk.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_frozen_risk_forensics.py tests/test_m3w_frozen_interaction.py tests/test_m3w_frozen_runtime.py tests/test_m3w_risk_calibration.py -q
```

The first command is for a new output directory only. On the completed local run,
use `--resume` directly: it validates identity, receipts, source bindings, cached
query reductions and the completed report, then adds zero new calculations.
Do not delete outputs to simulate freshness or bypass an identity mismatch.
The report script independently reduces receipt-bound query details without
calling the production summarizer, then writes aggregate Markdown/CSV/SVG.

## Completed Work

- Included one-candidate pilot: PID29356, exit0, 0.9919056 seconds.
- Remaining 23 candidates: PID29415, exit0, 23.2947336 seconds; one pilot reused.
- Completed resume: exact replay of all 24 candidates/72 summaries, no new work.
- Final scoped tests: 42 passed in 16.68 seconds. Full historical report-writing
  integrations were not rerun to avoid overwriting unrelated old results.
- Reporter: 72 independent reductions, 69,840 repeated query-status checks,
  exit0. These are repeated engineering checks, not additional research samples.
- Plot rendered and visually inspected; all 24 full-joint labels and both panels
  are visible. All 72 cells remain in the full table. Initial Fontconfig cache
  warnings did not prevent rendering; no failed plot was presented as complete.

Initial test-first import failure occurred before adding the new module and was
resolved by implementation. No experiment failure was hidden or replaced with
another protocol. All required sessions are terminal. No large data, raw images,
latent caches, models or per-query records are intended for Git.

## Identities

| Artifact | SHA256 |
| --- | --- |
| Parent comparison analysis | a9e344a1a1854c3a86d4dcf6afb3a058b55c16a4a28761c0fc1f39761320ea0f |
| Risk run identity | f13cb1d7590b503676bb0ca5a1f508eb31436ed2fc7204afd2f5623d509dd72e |
| Completed risk analysis | bc061b2fd239fdb2984890f75b02baad10f3b4ff40451ba9d2030d6edc72444c |

`analysis.json` contains model/source report bindings and original independent
parent checks. The private completion manifest binds all 24 per-candidate receipts
under `data/stage_cvpr2027_experiments/frozen_risk_forensics_v1/`. Each receipt
binds per-query records and aggregate summaries. The local preview and plotting
cache remain there too; Git contains only aggregate reports and the chart SVG.

All inputs/primary metrics/thresholds are unchanged. This is post-hoc development
diagnosis at one physical site, not independent risk calibration or confirmation.
