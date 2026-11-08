# Learn the Value of Waiting, Not the Value of a Later Edit Alone

The [timing experiment](../slot_timing_v1/README.md) found useful delayed
assignments in hindsight but not with greedy local scores. I trained an
explicit release-time comparison: keep the original forecast edit, or wait
up to two queries for the first positive past-only opportunity at the same
radius. The head predicts the paired change in conservative gain and the
three recording-risk terms. Training uses only other OpenDD locations and
deterministic largest-native-radius recipients at three fixed amplitudes;
it does not use held-location-trained nominees to produce donor labels.
Future labels are used only to supervise this comparison on fitting locations.

At evaluation, decisions are made at the original releases. A waiting
edit can spend only credit left after reserving the original schedule's
balance at the current query. Any original edit not marked for waiting
is preserved at its original time and recipient. The comparison uses the
same offered slots, not the same realized edit count: if a pending slot
expires, its missing edit remains part of the policy outcome.

| Rule | Deferred / moved / dropped | Actual edits | Joint passes | Undefined easy | Direct gain vs anchor (pp, 95% CI) |
|---|---:|---:|---:|---:|---:|
| Frozen anchor | 0 / 0 / 0 | 1,457 | 77 | 334 | reference |
| Defer on negative past score | 534 / 414 / 120 | 1,337 | 71 | 333 | -0.054 [-0.119, -0.010] |
| Learned gain | 479 / 379 / 100 | 1,357 | 70 | 337 | -0.025 [-0.065, +0.002] |
| Learned gain and nonpositive risk increments | 118 / 96 / 22 | 1,435 | 74 | 334 | -0.007 [-0.017, -0.001] |

The conservative learned variant stays closer to the baseline than the
simple and gain-only rules, but still loses direct gain, clears fewer joint
checks and does not reduce undefined easy risk. It is not deployed. The
label-aware timing witness is not evidence that this timing head works.
Under the original ledger, more training cannot solve the recordings with
no reachable credit. For payable recordings, the next useful test is
to fit the **same** recipient-time decision problem as the oracle, with
paired labels and explicit penalties for expiry and credit competition,
rather than predict a fixed local opportunity and hope its rank transfers.
Compare that with a simple anchor-preserving control, and keep unreachable
recordings in the full readout.

```sh
.venv-pytorch/bin/python scripts/run_m3w_release_wait.py
.venv-pytorch/bin/python scripts/run_m3w_release_wait.py --replay
```

21 fresh Torch fits, three seeds, 16,800 optimizer updates in 13.08 seconds
on native arm64 CPU. Checkpoints and PID heartbeat every 100 updates;
the original command resumes an interrupted fit. Confidence intervals
resample seven development locations 3,000 times after averaging seeds;
606 recording-seed views are not independent. The original 2% rule is
unchanged, the 30 DroneCrowd confirmation groups remain sealed, and no
metric or seconds-level claim, deployment, Stage5C or SMC follows.

[Registration](registration.json) | [Results](analysis.json) |
[Checkpoint receipts](head_freeze.json).
