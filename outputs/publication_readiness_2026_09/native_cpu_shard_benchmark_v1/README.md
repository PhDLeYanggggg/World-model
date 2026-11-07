# Native CPU Sharding: Exact Development Benchmark

The full 40-recording calibration job is unchanged. To assess a local
runtime option for the larger confirmation role, the benchmark runs
three whole development query shards in serial and then in separate
processes. They contain 18 past-only queries and 1,255 target instances.
Both schedules use the same models, batch size 256 and four Torch
threads per worker, with zero DataLoader workers. No external input
or outcome is opened by the benchmark.

| Measure | Serial | Three independent workers |
| --- | ---: | ---: |
| End-to-end wall time | 28.59 s | 14.88 s |
| Neural predictions checked | 27 | 27 |
| Stored policy arrays checked | 36 | 36 |

Speedup is **1.92x** on this run, which includes process startup and
checkpoint loading. All raw neural output arrays, CV/OLS4 paths,
target IDs, and sparse switch arrays have identical byte hashes.
EqMotion consumes about 94% of measured serial neural forward time.
This is a small runtime sample with the existing calibration job
present, not a timing guarantee for all 72 confirmation recordings.
If the measured speedup carries over, roughly 13 serial hours would
be about 7 parallel hours before scoring.

The [runtime amendment](../dronecrowd_confirmation_readout_v1/runtime_amendment.json)
permits only process-isolated whole-recording inference. Three workers
balance disjoint recordings using previously audited workload metadata.
They share no Torch tensors. They use the original per-record inference
and checkpoint formats. After workers finish, the original inference
entry point verifies every checkpoint and seals the whole roster. Scoring
and scientific criteria do not change. No confirmation labels can be
accessed by this executor; admission still requires a qualifying,
replayed calibration result and committed selection.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/benchmark_m3w_native_cpu_shards.py
.venv-pytorch/bin/python scripts/run_m3w_dronecrowd_confirmation_parallel.py register
```

The timing command is one-write and does not overwrite existing evidence.
The optional execution command, after admission, is the parallel script
with `infer`. It is not run now. See [measured results](analysis.json).
