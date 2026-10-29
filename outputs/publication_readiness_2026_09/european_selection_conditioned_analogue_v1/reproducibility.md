# Reproduction and Runtime Evidence

## Completed Work

| Phase | PID | Evidence | Wall time | Peak RSS |
|---|---:|---|---:|---:|
| Real-data engineering pilot | 77820 | Exit 0; four views; 7,572 queries supported; frozen outputs and normalization exact | 27.645489 s | 5,287,936,000 B |
| Full fitting computation | 78105 | Completion receipt binds 72 group hashes and summary; zero neural updates | 231.309565 s | 6,710,427,648 B |
| Full inference/retrieval replay | 78696 | All 72 groups and summary exact; final verified event at 04:48:51 UTC | Not separately measured | Not separately measured |

The replay's terminal handle expired before its exit status could be recovered.
Its final `verified` event is written only after all numerical and hash checks;
the process was subsequently observed absent. `replay_receipt.json` therefore
records a verified completion event and a null exit code, not an invented exit 0.

The protocol was committed at `2ece4a8f` before the pilot. Pilot verification and
receipt-check code were committed at `6bb57ee9` while the full run was active.
Registration SHA256:
`6768f2b440ebc1598a5a70e40ce01c744cbc3bc4424e4a5dd108d57ca6f52683`.
Complete receipt SHA256:
`61b31308e9bab779357e59c83cd941a7cc51f23faac405eb8a5210d2ad7a8548`.
Summary SHA256:
`12b09251b7fd2be57f64f3eed7148e3eb4be81067c604beb9fb2dcfb33c1ef58`.

Fitting retrieval and full replay are **fresh_run**. Existing models, original
selections, input bindings and inherited controls are **cached_verified**. New
neural training, policy execution, held readout and independent evaluation are
**not_run**. No threshold, primary arm or dataset role changed after readout.

## Checks and Their Scope

A separate arithmetic implementation validates 1,440 arm-target records across
two mapped checks, including shared controls twice; these are not 1,440 unique
predictions. It also validates 443 selected reference banks, 288 decomposition
records, 360 inherited controls and all five magnitude-screen conditions.
Receipt verification and 21-slice/figure-hash checks exited 0 at closeout.
The local figure was visually inspected. No image or private feature data is
included in the public result package.

The 53 related tests passed in 4.03 s on the same scientific code. Subsequent
changes are documentation and light result receipts only. The historical full
artifact-writing test suite was not run; no full-suite pass is claimed. These
tests and replays establish internal reproducibility, not independent scientific
replication, generalization, or a successful risk model. The magnitude gate fails.

## Commands

Run from the repository root using native arm64 Python:

```bash
.venv-pytorch/bin/python -m scripts.run_m3w_selection_conditioned_analogue check
.venv-pytorch/bin/python -m scripts.verify_m3w_selection_conditioned_analogue --check
.venv-pytorch/bin/python -m scripts.report_m3w_selection_conditioned_analogue --check

.venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_selection_conditioned_analogue.py \
  tests/test_m3w_selection_conditioned_verifier.py \
  tests/test_m3w_selection_conditioned_report.py \
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

Do not delete completed outputs to rerun phases. `run --resume` is for an
interrupted incomplete computation only, after checking its PID and heartbeat.
Atomic context/seed receipts resume deterministically; existing outputs must
match rather than being overwritten. A completed phase rejects overwrite.

## Runtime and Material Access

Native arm64 `.venv-pytorch`, CPU threads 4/inter-op 1, workers 0, nearest-neighbor
jobs 1 and numerical thread limit 4. No x86 Conda or Rosetta training. A 10-GiB
reserve, 12-hour boundary, process lock and heartbeat remain in force. Existing
authorized CREATE original-TRAIN packets are streamed read-only and checked by
manifest, bytes and hashes. No new scheduler job, cancellation or credential
change occurred. Data access does not establish scheduler status.

Private logs, cache and figure remain under
`data/stage_cvpr2027_experiments/european_selection_conditioned_analogue_v1/`.
Only code, aggregate reports and light receipts belong in Git. No raw data,
weights, history/latent cache, image or environment is part of this change.
The unrelated staged-index fingerprint before the owned-path commit is
`c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`.
Reproduction requires the hash-bound private inputs and ancestor models; the
public aggregate package is not a complete redistributable training dataset.

All 62,902 repeated queries, including 1,567 unknown outcomes, are retained.
Whole query recordings are excluded from donors, but upstream model and
preprocessing exposure remains: this is not whole-pipeline OOF. No independent
CI or significance claim is made from repeated windows or head seeds.
Obs8/pred12 rawstride12, detector-silver pixel/local and 2% positive selected
harm/reference remain unchanged. No metric, seconds, human-gold, physical-safety,
true-3D, foundation or deployment claim. Stage5C/SMC remain off.
