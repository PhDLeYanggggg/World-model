# Direction changes at the same intervention budget

**Decision: retain the existing policy.** Learned directions did not beat
the original directions. The hindsight allocation improved when it could
change recipients and path identities, but at the same query counts it
still cannot satisfy the original risk rule in half the recordings.

## Learned direction at fixed recipients

I used the same MovingDroneCrowd++ fitting-only roster as the timing
study: 24 recordings, nine source groups, and 8 observed / 12 predicted
annotation steps in image coordinates. Each fold fits six sources,
calibrates on two, and scores on the remaining one. All three arms make
95,769 replacements across 72 recording-seed views. For each selected
row and each step, the edit length is exactly one quarter of the frozen
nominee's displacement from CV. The learnable variable is its angle.

The two learned arms have the same architecture and matched seeds,
sampler, and normalized dual-reference loss. One sees the ego's history
and frozen candidate descriptors; the other also sees relative position
and velocity summaries from up to four observed neighbors. The shared
descriptors include a causal crowd count in both arms. This tests
additional local context, not a full joint predictor or a claim of
reflection equivariance. Unlike [EqMotion](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_CVPR_2023_paper.html),
this ablation fixes each waypoint's edit length to isolate direction.

| Direct conservative contrast | Gain difference (pp), paired 95% source interval |
|---|---:|
| Ego vs original quarter | -0.587 [-1.282, -0.056] |
| Neighbors vs original quarter | -0.621 [-1.342, -0.055] |
| Neighbors vs ego | -0.061 [-0.127, -0.013] |
| Neighbors vs OLS4 | -10.656 [-19.540, -1.796] |

All three fixed-count arms pass 0/72, with 45 all-risk violations and
15 undefined easy views each. Worst signed easy degradation is 12.19%
for the original quarter path, 8.17% for ego, and 8.29% with neighbors.
None qualifies on any of the 27 inner fold-seeds. The calibrated arms
make zero changes; this is not evidence of efficacy or safety under
the original rule. Intervals use 3,000 paired source draws, averaging
the three training seeds within sources.

## Capacity after allowing more directions and different recipients

At a fixed per-step radius, the closest point on the circle is aligned
with the observed CV residual. For wholly unknown rows, alignment to
the OLS4 displacement gives its minimum possible pairwise disagreement
without reducing the CV debit. This simultaneously optimizes the
available-step errors and both conservative reference gains for a
given action; it uses outcomes and is not an inference method.

At the frozen recipients, this direction oracle passes only **4/24**
risk checks, and its macro conservative OLS4 gain is -7.299 pp
[-15.651, +1.349]. Allowing each query to reassign its **same number** of
edits over eligible agents and the ten candidate path profiles gives:

| Hindsight menu | Risk and positive gain vs both | Risk infeasible | Negative upper bound | Unresolved |
|---|---:|---:|---:|---:|
| Original quarter paths | 6 | 12 | 6 | 0 |
| Quarter radii with arbitrary waypoint directions | 10 | 12 | 2 | 0 |

One circle solve timed out; projecting its verified quarter-path
witness onto the same circles and checking both gains and all risk
terms recovers it without changing the recorded solver outcome. No
new split, external outcome, risk threshold, calibration rule, or
deployment. The next test should change who is eligible and how the
query allocates or abstains, not increase the depth of a direction
head at this fixed allocation.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/run_m3w_matched_direction.py
.venv-pytorch/bin/python scripts/run_m3w_matched_direction.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_direction_recipient_capacity.py
.venv-pytorch/bin/python scripts/replay_m3w_direction_capacity_witnesses.py
```

Fresh native arm64 CPU run: 54 fits, three seeds, 500 updates each,
four threads, zero workers, 42.30 s. No-refit replay: 26.11 s.
The solver recorded a checkpoint per recording. Thirty-six targeted
tests passed. The initial pilot was stopped (exit 143) when separately
centered ego positions were found to be unsuitable for neighbor
distances; no pilot result is included. The corrected run reads the
encoded ego-aligned neighbor positions and masks, with an integration
test against the real encoder. Raw inputs and checkpoints stay local.

[Fit and score](analysis.json) |
[Original solver outcome](recipient_capacity.json) |
[Verified witness recovery](recipient_capacity_replay.json) |
[Working supplement](../evidence_manuscript_v46/README.md).
