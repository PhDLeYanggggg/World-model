# Direct Action Value and Peer Context

**Outcome: the peer head did not qualify.** This is a development-only
experiment on seven previously exposed OpenDD locations and 202 distinct
recordings, using the frozen nested CV-to-OLS4 menu, eight observed and
12 forecast native steps, the same online token rule, and original
CV-relative 2% recording risk. DroneCrowd confirmation was not opened.

For every legal one- or two-edit choice the heads learn conservative gain
relative to CV and incremental gain relative to OLS4, signed all/easy
risk excess, easy reference mass and net easy change. Four event logits
predict positive CV gain, positive all/easy excess and easy support.
Only labels use the future. The base head uses the existing motion
features and causal prefix error; the peer head adds the two nearest
simultaneous relative histories per nominated agent. The base input
reserves the same channels filled with zeros. Training nominations and
cutoffs exclude both the donor and the outer site. Thresholds and risk
penalties are fixed, not tuned against held outcomes.

| Online controller | Edits | Risk passes | Risk pass and beat both | Undefined easy | Gain vs CV (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Observed-prefix heuristic | 1,340 | 76 | 56 | 320 | +0.073 |
| Motion, gain only | 1,457 | 87 | 61 | 334 | +0.068 |
| Motion, gain and risk | 1,346 | 80 | 57 | 341 | +0.056 |
| Peers, gain only | 1,482 | 77 | 50 | 336 | +0.089 |
| Peers, gain and risk | 1,311 | 76 | 48 | 355 | +0.108 |

Counts are out of 606 recording-seed views, not 606 independent
recordings. The direct peer contrast with motion is +0.037 pp
[-0.048, +0.171]; with the prefix score, +0.023 pp [-0.073, +0.126].
Positive CV gain AUROC is 0.657 for motion and 0.637 with peers;
positive all-risk excess AUROC is 0.690 and 0.663. Easy-excess AUROC
is undefined in three of 21 location-seed cells; no whole-roster
average is reported. Motion and peers rank easy support at 0.655 and
0.667 AUROC, respectively. Better support ranking alone does not yield
an improved causal decision. No interval establishes a peer benefit.

**Same-count diagnostic.** A label-blind retrospective match gives
each head the smaller online total for each recording while preserving
the chronological token constraints. Both heads make exactly 1,126
edits; 371/606 recording-seed views have a nonzero matched count.
The motion arm gets 54 joint passes, peers get 34. Direct peer-minus-motion
gain is **+0.012 pp [-0.031, +0.081]**. The match sees all query histories
in a recording and is not online. Intervals resample seven locations
after averaging seeds, using 3,000 draws. Stronger risk discrimination
is still needed, especially where predicted gain and easy support disagree.

`fresh_run`: 42 native CPU Torch fits, 29,400 updates, three seeds,
four threads, zero workers. `cached_verified`: frozen input rankers and
zero-update checkpoint replay. No deployment change, Stage5C, SMC or
independent confirmation. [Scores](analysis.json) | [Setup](registration.json).

```sh
.venv-pytorch/bin/python scripts/run_m3w_direct_count_value.py
.venv-pytorch/bin/python scripts/run_m3w_direct_count_value.py --replay
```
