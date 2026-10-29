# Reproduction and Runtime Evidence

## Observed Runs

| Phase | PID | Result | Wall time | Peak RSS |
|---|---:|---|---:|---:|
| v1 input-only TRAIN audit | 73798 | Exit0; 24 packets, compatible width380 | 52.278760s | 2,454,241,280B |
| v1 engineering pilot | 74429 | Exit1 before retrieval/scoring; hash-contract mismatch preserved | See interruption record | No success claim |
| v2 real pilot | 75039 | Exit0; four views, 7,572/7,572 supported, frozen predictions exact | 22.954905s | 5,268,684,800B |
| v2 full computation | 75317 | Completion receipt; all72 views, zero neural updates | 184.717100s | 6,341,705,728B |
| v2 complete inference/retrieval replay | 75964 | Exit0; all72 group receipts and summary exact | Not separately measured | Not separately measured |

The v2 input audit is reused **cached_verified**, not rerun under a new name.
The full computation and exact replay are **fresh_run**. Existing weights and
original selections are **cached_verified** against hashes and exact outputs.
New held/independent evaluation, policy execution and deployment are **not_run**.

v2 registration was committed at5f1185b9; successful pilot at0d0e053c preceded
the full run. Registration SHA256:
`86b8c9b21f56450987653d36aa5135aa5c910cb3a1233f9fe133c913ccaec25f`.
The complete receipt binds all72 group hashes and the summary. Full replay's
terminal event and exit0 are recorded in replay_receipt.json. The original v1
pilot failure and two hash-contract regression tests remain in the repository.

## Verification

Independent arithmetic implementation, not an independent research team:
720 arm-target checks,443 recording banks and144 inherited cohort/prediction
comparisons. All3 descriptive screen criteria and the magnitude comparison
are recalculated. The 21 source/context/seed slices and figure receipt replay
without new inference. The local figure was visually inspected; no raw image
or data file is included in Git. Thirty-eight related tests passed in3.47s
before full completion; the tested scientific code has not changed since.
The historical artifact-writing full suite was not run, and no full-suite pass
is claimed. Later changes are documentation/light receipts only.

## Commands for the Existing Completed Run

```bash
.venv-pytorch/bin/python -m scripts.run_m3w_pooled_analogue check
.venv-pytorch/bin/python -m scripts.verify_m3w_pooled_analogue --check
.venv-pytorch/bin/python -m scripts.report_m3w_pooled_analogue --check

.venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_pooled_analogue.py \
  tests/test_m3w_pooled_analogue_pool.py \
  tests/test_m3w_pooled_analogue_verifier.py \
  tests/test_m3w_pooled_analogue_hashes.py \
  tests/test_m3w_pooled_analogue_report.py \
  tests/test_m3w_selected_analogue.py \
  tests/test_m3w_selected_analogue_verifier.py \
  tests/test_m3w_selected_analogue_report.py \
  tests/test_m3w_risk_decoupling.py \
  tests/test_m3w_risk_decoupling_verifier.py \
  tests/test_m3w_query_context_risk.py \
  tests/test_m3w_query_context_readout.py
```

Do not delete completed outputs to rerun phases. `run --resume` is only for
an interrupted incomplete run, after checking PID and heartbeat. Four-source
context/seed units resume deterministically; a partial unit is recomputed and
existing immutable receipts must match. A completed phase rejects overwrite.

## Environment and Access

Native arm64 `.venv-pytorch`, Python3.11/PyTorch2.12; CPU threads4/inter-op1,
single process/workers0; nearest-neighbor jobs1 with numerical thread limit4.
No default x86 Conda or Rosetta training. Ten-GiB reserve and12-hour boundary
remain, with atomic receipts, process lock and heartbeat. At closeout22GiB
remains free. This is a local computation over existing authorized read-only
CREATE original-TRAIN packets. No new scheduler job, cancellation or credentials
change. Packet access does not establish scheduler or unrelated job status.

Private data/cache/heartbeat/figure stay under
`data/stage_cvpr2027_experiments/european_pooled_analogue_v2/`; no raw data,
checkpoint, image, environment or unrelated staged file is included in the
owned-path commit. The unrelated staged-index fingerprint is preserved:
`c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`.
Existing material bindings are necessary for reproduction: public aggregate
reports alone are not a complete distributable training dataset.

Obs8/pred12 rawstride12, detector-silver pixel/local only; 2% positive selected
harm/reference unchanged. No metric/seconds/gold/physical-safety/true3D/foundation
claim. Upstream exposure means no whole-pipeline OOF, no independent CI and no
deployment promotion. Stage5C/SMC stay off; the research goal remains active.
