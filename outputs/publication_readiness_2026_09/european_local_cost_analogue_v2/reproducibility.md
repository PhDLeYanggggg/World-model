# Reproduction and Runtime Evidence

## Observed Invocations

| Phase | PID | Result | Wall time | Peak RSS |
|---|---:|---|---:|---:|
| v1 pilot | 80933 | Exit 0; 7,572 supported, controls exact | 18.213214 s | 5,314,772,992 B |
| v1 full run | 81249 | Exit 1; diagnostic alias error before first group | Not combined with successful runtime | No scientific success |
| v2 pilot | 81915 | Exit 0; 7,572 supported, controls/neighbors/normalization exact | 18.615700 s | 5,333,155,840 B |
| v2 full computation | 82222 | Exit 0; 72 views, zero neural updates | 135.378112 s | 6,634,569,728 B |
| v2 full numerical replay | 82750 | Exit 0; 72 groups and summary exact | Not separately benchmarked | Not separately benchmarked |

Scientific registration commit `c9d82416` precedes the real v1 pilot. The
diagnostic-writer repair, regression, failure record and v2 registration were
committed at `94f30934` before the v2 pilot. No scientific rule changed.
Registration SHA256:
`b82eaa973bb67aac682ec1d2d2229ff34440d13e59e0c492179753aa2eee3152`.
Complete SHA256:
`8469e14265bf235beee556ea04a91d81cb0c787eb5ba229970fecd0a505e1c3d`.
Summary SHA256:
`307b6b798d22a57ab1c2e4faa4d529c02c330e1c854e6854bc0ff849af230661`.

## Validation Scope

Full replay rereads hash-bound original-TRAIN inputs, predicts frozen neural
moments, reconstructs query selection, retrieves donors and recomputes local
fits and scores. All output group hashes and the summary match exactly.
Separate arithmetic checks 864 arm-target records, 443 reference banks, 144
donor-loss inequalities, 144 inherited controls and five screening conditions.
Twenty-one descriptive slices and the figure hash replay; the rendered figure
was visually inspected. Figure/data remain private, not part of Git.

The 74 related tests pass in 4.01 s. They include the writer regression that
failed before the fix, reference-offset sign, bounded optimum, source-unit and
frequency weighting, query-future perturbation, unknown support, exact inherited
neighbors, corruption detection and undefined metric preservation. The whole
historical artifact-writing suite was not run. Passing engineering tests and
lower donor loss do not mean the scientific gate passed; it did not.

## Commands

Use repository root and native arm64 `.venv-pytorch`:

```bash
.venv-pytorch/bin/python -m scripts.run_m3w_local_cost_analogue check
.venv-pytorch/bin/python -m scripts.verify_m3w_local_cost_analogue --check
.venv-pytorch/bin/python -m scripts.report_m3w_local_cost_analogue --check

.venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_local_cost*.py \
  tests/test_m3w_selection_conditioned*.py \
  tests/test_m3w_pooled_analogue*.py \
  tests/test_m3w_selected_analogue*.py \
  tests/test_m3w_risk_decoupling*.py \
  tests/test_m3w_query_context_risk.py \
  tests/test_m3w_query_context_readout.py
```

Completed phases reject overwrites. Do not delete results to rerun. For an
interrupted incomplete v2 run only, verify PID/heartbeat first and use `run
--resume`. Context/seed units are immutable; partially written units recompute
and must match prior receipts. Never restart on an observation timeout alone.
The v1 failed implementation is recoverable from its registration commit and
failure record, not silently reproduced by the corrected v2 entry point.

CPU threads4/inter-op1, workers0, nearest jobs1, numerical threads4. Architecture
guard precedes PyTorch imports in the new entry point. Process lock, heartbeat,
12h boundary and10GiB reserve remain. Existing authorized CREATE TRAIN packets
are streamed read-only; no scheduler submission/cancellation or credential
change. Read access is not a claim about queue or unrelated job status.

New fits/full replay are fresh_run. Existing weights, inputs and controls are
cached_verified. New neural training, held/independent inference, policy tests
and deployment are not_run. Private paths remain under
`data/stage_cvpr2027_experiments/european_local_cost_analogue_v2/`.
Public aggregates/code alone are not a redistributable complete training set;
the hash-bound private inputs and ancestors are required. No raw data, cache,
checkpoint, image or environment is included in the owned-path commit.
Unrelated staged-index fingerprint is preserved:
`c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`.

Obs8/pred12 rawstride12, pixel/local detector-silver and2% positive selected
harm/reference unchanged. Upstream exposure means not whole-pipeline OOF;
head seeds and repeated windows are not independent forecaster replications.
No metric/seconds/gold/physical-safety/true3D/foundation claim. Stage5C/SMC off.
