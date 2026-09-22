# Reproducing the Fixed-Count Controls

Local native arm64 `.venv-pytorch`, Python3.11.1, existing verified parent runtime.
Four compute threads, no DataLoader workers or multiprocessing, no new fitting.
Run from the repository root; the ignored source caches/checkpoints must already
exist and match every manifest. Missing assets are an error, not fallback data.

```sh
.venv-pytorch/bin/python scripts/run_m3w_risk_ranking.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_risk_ranking.py
.venv-pytorch/bin/python scripts/run_m3w_risk_ranking.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_risk_ranking.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_risk_ranking.py tests/test_m3w_risk_ranking_verifier.py tests/test_m3w_forest_cost_head.py tests/test_m3w_forest_verifier.py
```

Preflight PID25782 exited0 with1,295 bindings. Registration commit259b2144
was pushed before evaluation. Readout PID25931 exited0; replay PID25999 exited0;
separate-verification session61308 exited0. All22 scoped tests passed in1.53s.
The attempted later read-only process sample found PID25931 already gone and
the original session independently returned exit0; no restart was inferred.
No new peak-memory or training-throughput result is claimed. Disk check49GiB
available. No current CREATE queue query or HPC job was made; connection/project
conditions remain unverified, not proof that remote assets are absent.

Private output: `data/stage_cvpr2027_experiments/risk_ranking_v1/`:
identity, immutable decisions, decision manifest, PID heartbeat and event log.
An interrupted analysis can be rerun: an existing artifact must reproduce exactly
or execution fails. Advisory file locking prevents duplicate runs. Logs record
process identity and phase; no background training is left pending.

22 tests comprise13 ranking tests,1 independent-ranking test,6 unchanged forest
tests and2 unchanged forest-verifier tests. They are not22 new tests or a full
legacy-suite rerun. Parent checkpoints/scores were previously replayed and are
hash-verified here, not newly trained or newly independent.

Analysis SHA256:
`54575f34fc6c0d0ea79a78b89ff98658ccb66433f7b7f05c6f12ac9e3605fcfc`.
Replay SHA256:
`4e17cd406251648954c78f084e834c25f512c18fd246087a14d7b749b49c3266`.
Separate-verification SHA256:
`3554a3a1f505c44723967f7ba11779f100bc12201f53cd948937fce6c1b51582`.

No raw data, private scores, history/latent caches, checkpoints, images, videos,
third-party source or environment is part of the public result commit.
