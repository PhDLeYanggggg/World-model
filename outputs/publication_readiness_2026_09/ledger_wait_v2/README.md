# Closed-Loop Supervision for Causal Deferral

I tested whether the previous wait head failed because its training labels
ignored what actually happens when edits compete for recording credit. At
each release, I compare keeping the original edit to waiting two queries,
with the same radius and the same future positive-prefix destination rule.
The controlled experiment changes only the supervision:

- **Local:** score the first available positive candidate as if the slot
  were alone.
- **Ledger:** toggle that release while keeping all other release flags
  fixed, replay both full schedules, and compare the emitted forecasts.
  This includes collisions, expiry, other displaced pending edits and
  missing-label debits. It is a difference in forecast error, not an
  estimate of the effect of changing anyone's real behavior.

Both arms use the same pair-excluded donors, architecture, observed
trajectory features and five causal ledger features. The nomination
and count models never see the donor or the held evaluation location
they are used to prepare. Training evaluates all-keep, observed-prefix
and randomized background release flags. Inference has no future labels,
uses only the current observed query and accumulated ledger, and preserves
the credit reserved for untouched edits. Thresholds remain the fixed
sign rules. No independent confirmation data were used.

## Controlled Result

The 26,484 donor comparison instances are reused across folds, seeds
and background schedules; they are not independent samples. The real
ledger changed 2,292 labels (8.65%) and flipped the gain sign on 623
(2.35%). Across the counterfactual schedules, 314 other slots were lost
when waiting replaced keeping the original edit.

| Policy | Actual edits | Joint passes | Undefined easy | Direct gain vs frozen anchor (pp, 95% CI) |
|---|---:|---:|---:|---:|
| Frozen anchor | 1,457 | 77 | 334 | reference |
| Previous risk-gated wait | 1,435 | 74 | 334 | -0.007 [-0.017, -0.001] |
| Local, gain | 1,339 | 67 | 351 | -0.019 [-0.031, -0.007] |
| Ledger, gain | 1,325 | 67 | 350 | -0.022 [-0.032, -0.011] |
| Local, gain and risk | 1,409 | 73 | 347 | -0.0044 [-0.0094, -0.0005] |
| Ledger, gain and risk | 1,407 | 75 | 344 | -0.0040 [-0.0091, +0.0004] |

The direct ledger-versus-local risk-gated difference is **+0.0001 pp
[-0.0022, +0.0022]**. The ledger head does not establish a gain over
the local head or the anchor and still fails the recording gate.
Intervals average three seeds before bootstrapping
seven OpenDD development locations 3,000 times. Counts are across 606
recording-seed views (202 distinct recordings), not independent tests.

## Where the Unused Opportunity Remains

The [capacity diagnostic](capacity/analysis.json) is separate and
label-aware. It does not select a deployment rule.

- The exact two-query timing solver previously had 298 witnesses. Keeping
  only positive-prefix destinations and the original edges leaves 276 at
  the same per-recording count, radii and streaming credit. This solver
  does not impose the actual decoder's conservative reserve or first
  positive destination choice.
- With the **actual** decoder, change at most one release per recording
  and retain all original edit slots. The witness count is 138 versus 77
  with no change. Allowing a slot to expire gives 146, but is not
  count-matched. Of 1,457 possible single toggles, 1,157 actually move a
  slot and 300 expire it. 506 moves have positive paired utility; 132
  moves both improve the anchor directly and clear the full joint check.

The last result rejects the explanation that the decoder has no useful
one-change opportunity. Accounting for exact credit during training
removes a real mismatch, but neither target lets the model find enough
of the useful changes from the current inputs. The next experiment
should measure the information in recent competing trajectories and
pending-slot context on this actual, count-preserving contrast before
repeating a larger version of the same loss. The existing zero-credit
limitation and the original 2% rule are unchanged. No deployment,
Stage5C, SMC, metric, seconds-level or independent generalization claim.

This is a fixed-background rollout comparison, related to cost-to-go
training in [Ross and Bagnell](https://arxiv.org/abs/1406.5979) and
[multiagent rollout](https://arxiv.org/abs/1910.00120). It is not an
implementation of either algorithm and does not inherit their guarantees.

```sh
.venv-pytorch/bin/python scripts/run_m3w_ledger_wait.py
.venv-pytorch/bin/python scripts/run_m3w_ledger_wait.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_ledger_wait_capacity.py
```

42 native arm64 Torch fits, 33,600 updates in 39.54 seconds, four compute
threads and zero DataLoader workers. Checkpoint and PID heartbeat every
100 steps. The first v1 attempt stopped before training because a tuple
in the checkpoint was a JSON list in the receipt. v2 uses the existing
canonical pair loader, still verifying the checkpoint hash, fitting
locations and step count. The failed registration is retained. The
zero-update replay regenerates the immutable aggregate.

[Registration](registration.json) | [Results](analysis.json) |
[Checkpoint receipts](head_freeze.json) | [Capacity registration](capacity/registration.json).
