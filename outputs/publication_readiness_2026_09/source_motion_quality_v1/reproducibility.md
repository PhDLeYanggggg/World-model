# Reproducing The Source Motion-Quality Diagnostic

## Frozen Protocol And Inputs

Pre-computation commit `bb206551`; registration
`configs/m3w_source_motion_quality_v1.json`, SHA256
`6ed141cb921eede270bf7c4aafbe23155710d89255a78f98c321aba395c7d1c5`.
Ten direct frozen bindings and inherited data/model-control bindings remain
unchanged. A changed dependency fails rather than silently changing a result.

The verifier and aggregate renderer are downstream reporting code, not frozen
fit dependencies. The verifier preserves its initial PID/time receipt while
rechecking all scientific fields on subsequent invocations. A new PID is printed
in the fresh verification output, not substituted into historical evidence.

## Commands

From the repository root, using native arm64 `.venv-pytorch`:

```bash
.venv-pytorch/bin/python scripts/run_m3w_source_motion_quality.py --registration configs/m3w_source_motion_quality_v1.json --phase prepare
.venv-pytorch/bin/python scripts/run_m3w_source_motion_quality.py --registration configs/m3w_source_motion_quality_v1.json --phase probe
.venv-pytorch/bin/python scripts/run_m3w_source_motion_quality.py --registration configs/m3w_source_motion_quality_v1.json --phase replay
.venv-pytorch/bin/python scripts/run_m3w_source_motion_quality.py --registration configs/m3w_source_motion_quality_v1.json --phase analyze
.venv-pytorch/bin/python scripts/verify_m3w_source_motion_quality.py --registration configs/m3w_source_motion_quality_v1.json
.venv-pytorch/bin/python scripts/report_m3w_source_motion_quality.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_source_motion_quality.py tests/test_m3w_motion_quality_replay.py tests/test_m3w_source_motion_candidate.py tests/test_m3w_source_crossfit.py tests/test_m3w_source_modality_continuation.py tests/test_m3w_source_cost_dynamics.py tests/test_m3w_recording_diagnostic.py tests/test_m3w_crossfit_cost_diagnostic.py tests/test_m3w_direction_null.py
```

Preparation hashes all 29 raw sources and the cached query artifact on reuse.
First execution builds raw alignment; subsequent verified reuse is not a new
raw conversion. Fit receipts identify the exact training IDs, preprocessing,
config, software and models. Completed probe trials are skipped; unfinished
trials rerun individually. No within-tree interrupted-fit resume is claimed.
Each trial took approximately 0.85--1.75 seconds. Replaying predicts every held
probability again and requires bitwise equality. The verifier checks training
roles, feature widths, all scores, reference prevalence and zero-fit resume.

## Completed Runtime And Verification

Local native arm64; sklearn 1.8.0; four fitting threads, single-process data path.
Torch is imported by the inherited infrastructure, but no new Torch training
occurred. Serial forest probability reduction makes replay exact and restores
the four-thread fitting setting afterward.

Preparation PID 69308 exited 0, log span 5.171 seconds. Probe PID 69366 exited 0:
48 new fits, log span 69.538 seconds, summed fitting 64.135 seconds. Replay PID 69498,
analysis PID 69521 and verification all exited 0. Completed-resume PID 69578 added
zero fits; a second verifier run also passed with resume PID 70247 and preserved
the original verification receipt. No required task process remains running.

Forty-eight exact replays, 48 score/prevalence recomputations, 48 loaded-label
checks and 1,077 raw future-box mutation checks. The latter sample first/last
queries per scoped track, not 1,077 independent tracks. There are 149 unchanged
artifacts across completed resume. All 44 focused tests pass. Full legacy suite
not rerun: integration tests are not isolated from existing training/reports.

The aggregate figure was visually checked. Fontconfig initially warned about
unwritable system caches; Matplotlib generated its local font cache and both
outputs successfully. This was not a training runtime failure. A separate
system hash utility initially rejected the inherited locale; `LC_ALL=C`
recomputed the hashes successfully. No experiment data or result was changed.

| Artifact | SHA256 |
| --- | --- |
| preparation.json | 51653a7908f59bf7e0e5d626190c8836daddee8566ea831a26caa9908a827b84 |
| probes.json | 751c300e6d72a1fecc71a880d8f1ff77ad509afbb6d650199d22bfd6a1653320 |
| replay.json | 02eeb5d80b3b9a17d5d595b2b63647c3a65b9a36a6789d0c55608ea820e0dfd6 |
| analysis.json | 680862850ad71683f98c5b0ae56160b38fa2768e1a332c736db9aa1f2b901bb7 |
| verification.json | 8cbf7ebadf49cab3226eb29275f9c580065dbb3afe634ba70fc61d3e41738214 |
| private rows.npz | 06f92e7e773a2af41949664f8696e85c8de9a6b4e1ab15117b994043f8bda343 |

## Asset And Claim Boundaries

Raw data, per-query rows/probabilities, forests and logs remain private under
`data/stage_cvpr2027_experiments/source_motion_quality_v1/`. Only code, config,
aggregate tables/reports and an aggregate scientific SVG go to Git. A public
clone alone cannot reproduce results without matching locally acquired source
data and earlier verified artifacts.

Fresh CREATE access failed at public-key authentication with an MFA notice;
no scheduler/remote asset inventory was obtained, no remote job submitted.
This is not evidence that the remote project is absent. Local completion does
not need CREATE. No new deployment or main test evaluation. Stage5C/SMC off.
