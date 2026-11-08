# Joint Versus Local Teacher on TUMDOT

The [three-reference diagnostic](../tumdot_three_reference_v1/README.md)
finds a strict positive risk-feasible joint assignment in all 27
half-radius views. I test whether supervising that assignment
improves the causal selector. The comparison uses the exact same
six directions, recipients and radii as the preceding experiments.

Both models fit centered gain and all/easy harm versus hindcast
plus a direction preference. The only change is the preferred
direction label: independently maximize the paired row gain, or
imitate the donor's exact three-reference recording assignment.
The two models see the same input features, initialization, draw
sequence and loss weights. The optimizer's labels come only from
the four donor recordings, excluding both the held intersection and
collection day; exact released track IDs are checked disjoint.
Evaluation uses only observed-history features. Inference is a
row-wise decoder trained on a joint teacher, not a claim that a
joint risk program was run online.

| Half-radius model | CV lower (pp) | Risk passes / 27 | Risk and positive vs CV and OLS4 | Paths changed from hindcast |
| --- | ---: | ---: | ---: | ---: |
| Fixed hindcast | -0.03654 | 24 | 3 | 0 |
| Local teacher | -0.03566 | 24 | 5 | 643 |
| Joint teacher | -0.03710 | 24 | 5 | 878 |
| Local, guarded | -0.03386 | 24 | 4 | 343 |
| Joint, guarded | -0.03931 | 24 | 5 | 439 |

The direct joint-minus-local conservative contrast is **-0.02184
pp [-0.08223, +0.01343]**; with the same guard on both arms it is
**-0.02030 pp [-0.05892, +0.00129]**. Guarded joint minus fixed
hindcast is -0.03664 pp, with site values +0.00494, +0.01395 and
-0.12881 pp. The third location remains the largest loss; the
bootstrap is descriptive with only three intersections of one
previously exposed campaign. At quarter radius, risk is 27/27
but none has positive mean conservative CV utility.

**Decision:** no policy promotion. Better labels in a feasible
joint menu do not by themselves produce a better causal ranking.
The next useful step is to measure whether the failure carries
across acquisition campaigns with verified causal input lineage,
or to use observed local interaction context rather than another
teacher loss on the same rows. The full-roster 2% rule remains.
Do not use complete-only results for a deployment claim. DroneCrowd
remains 15 calibration groups and 30 sealed confirmation groups;
no new metric/seconds, Stage5C or SMC claim.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_tumdot_three_reference.py
.venv-pytorch/bin/python scripts/run_m3w_tumdot_joint_teacher.py
.venv-pytorch/bin/python scripts/run_m3w_tumdot_joint_teacher.py --replay
.venv-pytorch/bin/python scripts/build_m3w_tumdot_joint_replay.py
```

`fresh_run`: 54 fits, 21,600 updates, 14.368 s, PID 91808.
Native arm64 Torch, four threads, zero loader workers; checkpoints
and heartbeat every 100 updates. Rerun to resume.
`cached_verified`: original scorer, exact solver replay, zero-update
model replay, and anonymous scalar reconstruction. No CREATE or
independent confirmation was run.
