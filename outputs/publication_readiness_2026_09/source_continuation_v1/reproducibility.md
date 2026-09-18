# Full-Training Continuation Reproduction

## Material Passport

Registered source-training-only diagnostic. No held-source/main scoring or
scientific success claim follows from these commands. Existing checkpoints and
source caches are hash-verified, not recreated or published with this document.

Use the nativearm64 `.venv-pytorch` from the repository root. The entry point
sets CPU4/inter-op1, no DataLoader workers. All raw data, per-row forecasts,
images and checkpoints remain in ignored storage. Parent assets are required;
this does not promise data/weights distributed through GitHub.

```bash
.venv-pytorch/bin/python scripts/run_m3w_source_continuation.py --registration configs/m3w_source_continuation_v1.json --audit-only
.venv-pytorch/bin/python scripts/run_m3w_source_continuation.py --registration configs/m3w_source_continuation_v1.json
.venv-pytorch/bin/python scripts/run_m3w_source_continuation.py --registration configs/m3w_source_continuation_v1.json --replay
.venv-pytorch/bin/python scripts/verify_analyze_m3w_source_continuation.py --registration configs/m3w_source_continuation_v1.json
```

Resume uses the same training command. Do not launch a second writer while its
process is alive. Check recorded PID/session and heartbeat against live process
state; a stale note or heartbeat alone does not prove a terminal process. Atomic
checkpoints retain optimizer, sampler, Torch RNG and cumulative draws. A
completed run checks identities/hashes and adds zero optimizer updates.

The initial timing pilot is `--trial constant_seed17 --stop-at 2100`, not a
separate extra budget. Three parent models each contain2000 old updates; six
schedule branches each add8000. Report48000 new plus6000 unique inherited
updates, not60000 independent fresh updates. All steps2000/4000/6000/10000 are
retained. No best-checkpoint selection on test or on these training curves.

Focused tests cover original-algorithm equivalence, both-schedule interrupted
resume, constant/cosine matching, parent immutability and inherited causal input
boundaries. The full legacy suite has side effects and is not implied run by
these checks. Mathematical zero-majority observations concern input-independent
paths only, never impossibility of conditional forecasting.

Held roles, development selection, independent calibration/confirmation and
deployment promotion remain outside this diagnostic. Raw-frame units and
annotation provenance restrictions remain unchanged. Stage5C/SMC stay off.
