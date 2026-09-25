# Execution and Reproduction

## Actual Run

Registered and pushed at `ded7d344` before fitting. Native arm64 Python 3.11.1,
Torch 2.12.0, CPU computation threads4, inter-op1, DataLoader workers0. No
Rosetta/Conda execution, multiprocessing loader or resource probing.

| Phase | PID | Final UTC, 25 September 2026 | Status |
|---|---:|---|---|
| Frozen-head scaling diagnosis | 43766 | See diagnostic receipt | Finished |
| Matched-input preparation | 46614 | 09:24:17 | Finished |
| 100-update pilot | 46764 | 09:25:37 | Finished |
| Training, score replay and evaluation | 46797 | 09:31:18 | Finished |
| Full verification | 47533 | 09:33:45 | Finished |
| Completion checks and scoped tests | See completion receipt | 09:34:32 | Finished |

Pilot fit time was 0.1511 seconds for 100 updates. The pilot continued to the
fixed 2,000-update budget, rather than adding 100 extra updates. Each head has
22,914 parameters. All 54 heads finished 2,000 updates, totaling 108,000.
Summed head fitting time is 137.23 seconds. This excludes data preparation,
inference, score caching, replay, metrics, plots and reporting; it is not the
whole experiment's wall time. The training/replay/evaluation phase ran from
09:25:52 to 09:31:18 UTC. Local peak observed memory was about 7.5 GB.

Checkpoints save model, optimizer, RNG and draw state every200 updates. Exact
resume is tested. All required processes reached terminal success; none were
killed or restarted because of slow progress. The local pilot supported local
execution. No remote training was necessary or claimed.

## Provenance

Fresh work: 54 frozen-head scaling checks, 54 real Torch head fits, new score
banks, all-view evaluation and replay. Cached but verified: source manifests,
18 OOF and nine final forecasters, 54 old heads, geometry and role identities.
All existing forecasts remain unchanged. No future targets, future masks,
central velocity, or test endpoints as goals are added to inference.

Each new head exactly reproduces scores on the first4,096 excluded-index rows.
This is not a random sample or full-bank re-inference. Separately, complete
144-view decision/metric accounting was rerun; all36 original views match the
previous frozen-producer study. All54 sampler count arrays match their original
heads. Training drew no unknown-label rows. No fitting began on excluded scenes.

SHA-256 receipts:

```text
analysis: 9257f63c9bc674e629fb528393655b4d2632c8831bfda951a46d4e6ae828fc07
summary: 1bbc64fb206bee66a1c213cdbc3a2b3d5b455e54b4a5f065bca5410370561560
loss CSV: d7fc4116cf20890c34fec1609f9a9f7607c644473e60034db8990650209d43df
scoped tests: f5155bdac2c2924349992bf1d1cfc121bef1a61a0a6e14588195ca687eb5dd5f
```

`verification.json`, `head_replay.json` and `completion_checks.json` bind the
frozen implementation, source lineage and local checkpoint hashes. 210 tests
across32 scoped files pass; the full historical test suite was not run.

## Reproduction

Run from the repository root with its native `.venv-pytorch` environment.
Commands and safe resume instructions are in [operation_zh.md](operation_zh.md).
Existing verified checkpoints are reused, not silently retrained. The registered
code, configuration and model tests must remain frozen; changes require a new
experiment identity. Figures use aggregate metrics, not individual trajectories.

Local private products live under
`data/stage_cvpr2027_experiments/european_geometric_cost_v1/`. Raw tracks,
checkpoints, row-level score banks and upstream training assets are not committed.
Cloning the repository alone cannot reproduce training without those inputs.

CREATE status is `cached_verified` from the authenticated read-only queue
observation at05:50:09UTC, not a fresh queue query this run. The remote M3W
asset directory is `not_run/unverified`. No remote job was submitted, altered
or cancelled. These missing remote inventory details did not block a justified
local fit.

All scientific results are opened-source, detector-track, pixel/raw-step
development evidence. No reserved confirmation, new calibration, threshold
search, deployment change, Stage5C or SMC occurred.
