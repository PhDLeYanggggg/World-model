# Execution and Verification

The scientific files and matrix were committed in
`6cc5b0dbd427af49abc3d430d0bed45c158a8490` before the full readout. This completion
package does not edit the registered model, runner, config, tests or matrix.

| Phase | PID | Completion UTC, 2026-09-24 | Evidence source |
|---|---:|---|---|
| 100-update real Torch pilot | 25748 | 23:52:41 | fresh_run, resumed inside the fixed budget |
| All 36 heads | 25817 | 23:54:53 | fresh_run; 18 ridge, 18 neural, 36,000 neural updates |
| All 24 policy readouts | 25817 | 23:57:26 | fresh_run on cached_verified forecasts and utility |
| Complete metric reconstruction | 26220 | 23:59:21 | cached_verified, complete readout reproduced |
| All 36 checkpoint replays | 26220 | 23:59:47 | fresh_run, 4,096 sampled rows each, exact predictions |
| Accounting and numerical diagnosis | Separate local processes | After complete readout | fresh arithmetic and targeted replay, not new training |

All required processes returned successfully. Training did not stop because of
slowness. The local date is 2026-09-25 in Europe/London; UTC logs remain unchanged.
Raw events, checkpoints and score arrays remain private in
`data/stage_cvpr2027_experiments/european_conditional_risk_v1/`.

## Runtime

Native arm64 `.venv-pytorch`, Torch 2.12.0, NumPy 2.4.6, CPU compute threads 4,
inter-op 1, DataLoader workers 0. The entrypoint rejects x86_64 macOS before
Torch import. No resource probing, multiprocessing or NumPy training fallback.
The full process was observed at about 8.4 GB RSS, within local resources.

Each neural head has 22,914 parameters and 2,000 updates. The summed fit-loop
time is 26.35 seconds, including the resumed pilot; this excludes assembly,
identity checks, cached inference, solver work and bootstrap. Full matrix plus
readout took approximately four minutes after input verification. These are
small risk heads on frozen forecasts, not an end-to-end world model trained
in seconds. Training minibatch losses are not validation errors, calibration
scores or convergence proof. See [training_losses.md](training_losses.md).

No new CREATE job was submitted. Earlier handoff information is not a fresh
scheduler query; the exact remote M3W directory remains unverified. Protected
simulation jobs, directories, credentials and authentication were untouched.

## Bound Evidence

Analysis SHA256:
`ad28268fd93beee6706bf85a6171cd531bf6e1dfd17feef78faa929ab89db673`.

- Complete metric reconstruction binds to that exact analysis.
- All 36 checkpoint replays match exactly and bind sampled row/checkpoint hashes.
- Accounting verifies 15 unchanged reference readouts, nine exact all/easy
  sampler/preprocessing pairs and 72 decision receipts with reconstructed
  recording/frame query keys and actual matched intervention counts.
- Every risk head has completed its declared endpoint before first comparative
  readout. The 100-update pilot is included, not counted as additional training.
- The separate numerical repair passes all ten observed singleton failure views
  with unchanged predictions; no scientific V1 output is rewritten.
- The generated tradeoff figure was visually checked for readable, nonoverlapping
  labels. Only generated SVG, not raster media, belongs to the public commit.

The completion run passes 152 tests across 16 scoped files covering event-risk,
feasibility pruning, CV-reference reports, source producers/roles, gain/harm,
metrics, joint controls and forecast replay. This is not the full legacy suite.
Tests, cached verification, fresh sampled inference and actual scientific
support are distinct evidence categories.

## Reproduction and Publication Boundaries

See [operation_zh.md](operation_zh.md) for exact commands. Identity mismatch
requires diagnosis, not overwriting checkpoints or changing a frozen version.
No independent selection, calibration or confirmation outcomes were opened.
All intervals are conditional source-development locality bootstraps, not a
physical-safety guarantee or independent test result.

The public package contains code, tests, aggregate metrics and reports only.
No raw recordings, packed data, row-level prediction arrays, checkpoints,
third-party media or environment is committed. Unrelated staged work remains
untouched. Stage5C and SMC remain disabled; no deployment or submission occurred.
