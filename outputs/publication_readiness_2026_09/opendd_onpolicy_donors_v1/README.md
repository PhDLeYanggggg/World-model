# Train on policy-matched allocations

The training allocation generator now matches the algorithm used to produce
the evaluated incumbent: learned recording counts followed by the score-only
count/movement-constrained optimizer. The selector features, architecture,
updates and evaluated action mask are unchanged.

## Status before scoring

105 triple-excluded nominee rankers, 63 double-excluded count heads and 42
incumbent-inclusive selection heads are trained. Total: 121,800 optimizer
updates. All 21 outer-location/seed choice sets are frozen. The choice freeze
and producer manifest are committed before outcome scoring. These are new
Torch fits, not a NumPy substitute or reused weights presented as new fits.

The original 2% rule and empty-support treatment are unchanged. No external
confirmation labels or release policies have been modified. This is exposed
OpenDD development evidence, not independent confirmation.

## Reproduce

```bash
cd /Users/yangyue/Downloads/World
export LC_ALL=C OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_onpolicy_donors.py register
.venv-pytorch/bin/python scripts/run_m3w_opendd_onpolicy_donors.py producers
.venv-pytorch/bin/python scripts/run_m3w_opendd_onpolicy_donors.py fit
.venv-pytorch/bin/python scripts/run_m3w_opendd_onpolicy_donors.py score
.venv-pytorch/bin/python scripts/audit_m3w_onpolicy_donors.py
```

Existing completed checkpoints resume with zero updates. The local caches and
weights live under `data/stage_cvpr2027_experiments/opendd_onpolicy_donors_v1/`
and are not included in Git. Upstream registered datasets, maps and checkpoints
must also be present. The registry binds their prior experiment lineage.

The pilot completed 4,000 real updates in 7.254 seconds with peak RSS 568 MB.
Completing the producer matrix added 101,000 updates in 172.859 seconds;
selector training took 108.443 seconds, peak RSS 1.67 GB. Local execution was
sufficient; no CREATE job was submitted. Every 100 updates saves optimizer,
weights and RNG. Per-invocation logs and the process lock are in the cache.

Nested lineage verification checked 105 ranks, 63 count heads and 315 teacher
caches. Teacher-recording statuses include one unresolved optimization, which
was excluded rather than relabeled. Repeated teacher rows are not independent
recordings. The 29 focused tests cover the new role exclusions, budget
preservation, unresolved handling and zero-count aggregation alongside the
incumbent, listwise, risk and matched-exchange regressions; this is not a
whole-repository test claim.

See [the frozen protocol](protocol.md) for the primary comparisons and
[registration](registration.json), [producer freeze](producer_freeze.json)
and [choice freeze](head_freeze.json) for the bound inputs and outputs.
