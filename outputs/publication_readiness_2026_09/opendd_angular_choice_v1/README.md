# Predict whether to rotate, with an explicit keep option

The constant-angle oracle finds useful directions without replacing
recipients. I fit two heads to predict the gain for each of 12
nonzero angles. The direct head learns observed and conservative gain;
the factor head learns observed gain and missing-step disagreement.
Both can keep the exact incumbent at score zero. The donor-only
constant control chooses one angle per training fold from average
donor gain. The factor head also runs on exactly the direct head's
rotation mask to separate action selection from angle ranking.

| Rule | Rotated edits / 780 | Dual passes / 606 | Risk passes | Lower gain vs incumbent, pp [95% location interval] |
| --- | ---: | ---: | ---: | ---: |
| Keep incumbent | 0 | 60 | 88 | Reference |
| Donor constant | 754 | 64 | 94 | -0.00362 [-0.00849, +0.00099] |
| Direct | 693 | 67 | 95 | -0.00934 [-0.02628, +0.00253] |
| Factored | 710 | 70 | 98 | -0.00893 [-0.02838, +0.00371] |
| Factored on direct mask | 693 | 70 | 98 | -0.00853 [-0.02631, +0.00399] |

The factored head's direct gain over the direct head on the same
rotation mask is +0.00077 pp [-0.00078, +0.00266]. The split does
not establish better ranking, and the zero threshold still rotates
most edits. Although the factored arm raises the dual-pass count,
its conservative utility is not better than keeping the incumbent.
All arms preserve the same 780 actual edit rows, original per-step
displacements and 397 undefined-risk views. None is a release policy.

This is a fresh local development comparison over seven outer held
locations and three seeds. Forty-two heads used 16,800 updates in
22.4 seconds. Checkpoint replay took 8.2 seconds and matched with
zero updates. Intervals use 3,000 location draws after averaging
seeds. Only the training donors supply labels, and no held labels
choose the zero threshold or the angle grid.

[Protocol](protocol.md) | [Readout](readout.json) | [Checkpoint receipts](head_freeze.json)

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_angular_choice.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_angular_choice.py --replay
```

The training entry uses four Torch threads, no data workers, a PID
heartbeat and a resumable checkpoint every 100 updates.
