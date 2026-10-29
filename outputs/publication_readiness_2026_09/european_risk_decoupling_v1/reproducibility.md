# Reproduction Contract

## Material Passport

Registered TRAIN-internal risk-head continuation. Code and protocol were committed
in `8d5d74b5` before the real pilot. Pilot receipt was committed in `85d1b207`
before the 36-fit run. The final result and runtime are recorded separately;
this file describes the reproduction contract, not a completion claim.

## Runtime and Provenance

Native arm64 `.venv-pytorch` Python/PyTorch, CPU4, inter-op1, workers0.
No default x86_64 Conda, DataLoader multiprocessing or device resource probing.
The 300-update pilot passed exact resume, matched draws and bit-exact protected
benefit/reference outputs. Original frozen risk-head checkpoints and approved
CREATE TRAIN packets are required; this is not a standalone public-data recipe.

Each objective arm uses the same original checkpoint, AdamW moments, input and
normalizer hashes, sampler state, causal features and 2,000 additional updates.
Eighteen origins yield 36 new risk-head fits, not 36 new forecaster seeds.
The action-coupled scalar loss includes signed dual terms and can be negative;
its magnitude is not an FDE, a risk percentage or directly comparable to the
estimation-only scalar loss. Fixed-cohort signed-risk MSE is the routing metric.

## Execution Order

On an authorized new reproduction checkout, register and commit bindings first:

```bash
.venv-pytorch/bin/python -m scripts.run_m3w_risk_decoupling register
.venv-pytorch/bin/python -m scripts.run_m3w_risk_decoupling pilot
.venv-pytorch/bin/python -m scripts.run_m3w_risk_decoupling train
# Commit training_freeze.json and all 36 fit receipts before inference.
.venv-pytorch/bin/python -m scripts.run_m3w_risk_decoupling fit-eval
.venv-pytorch/bin/python -m scripts.run_m3w_risk_decoupling check
.venv-pytorch/bin/python -m scripts.verify_m3w_risk_decoupling
```

For completed artifacts, use only `check` and
`scripts.verify_m3w_risk_decoupling --check`. Do not delete completed outputs
to bypass immutable phases. `train --resume` and `fit-eval --resume` are for
interrupted incomplete phases only. Check the PID, heartbeat and checkpoint
before recovery; a remote observation timeout is not a failed training job.

Checkpoints, event log, process lock and atomic heartbeat remain private under
`data/stage_cvpr2027_experiments/european_risk_decoupling_v1/`.
Checkpoint/heartbeat cadence is 100 updates, recovery boundary 12h, disk reserve
10 GiB, new private checkpoint cap 256 MiB. No remote training was required by
the pilot cost estimate. Do not submit duplicate CREATE jobs.

## Verification Scope

The replay command regenerates all 72 fitting predictions and aggregates from
frozen checkpoints and verifies original prediction/cohort hashes. The separate
receipt checker reconstructs 864 cohort-target checks, 216 policy checks, 72
matched counts and all nine routing conditions. It is an alternate arithmetic
implementation in this repository, not an external team's reproduction.

```bash
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_risk_decoupling.py tests/test_m3w_risk_decoupling_verifier.py tests/test_m3w_query_context_risk.py tests/test_m3w_query_conditioning.py tests/test_m3w_query_context_readout.py tests/test_m3w_rational_harm.py
```

Thirty-two scoped tests pass in 3.67s. Initial new verifier tests used an invalid
synthetic label with simultaneous positive benefit and harm; the real label
contract rejected it. The synthetic fixture was corrected, not the frozen
training code or real labels. The legacy full suite writes historical artifacts
and was not run. Completed same-version checks are reused.

## Evidence Boundaries

All data are exposed original TRAIN; inner-held rows are discarded before any
forward. Independent selection, calibration, confirmation and outer-development
roles remain unopened. Unknown outcomes remain in causal inference and support
accounting, but cannot supply supervised targets or known-outcome MSE.

This screen uses equal-view source-normalized MSE and repeated row occurrences,
not independent observation counts, ADE/FDE or a population safety guarantee.
Even a routing pass needs a separately frozen held design. No retrospective
threshold search, independent readout, deployment, Stage5C or SMC is authorized
by a fit-only result. Rawstride12 obs8/pred12, detector-silver and pixel/local
coordinates remain unchanged. Routine verification is handled under standing
authorization; formal submission and author certification remain separate.
