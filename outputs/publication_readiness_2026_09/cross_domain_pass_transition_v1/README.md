# Does a pass-transition decision transfer from OpenDD to Moving?

I trained a model to predict whether one same-query recipient exchange
adds or destroys a whole-recording pass. The displaced edit is moved
to another agent in the OLS4 direction with exactly the same peak
displacement. The rest of the original plan is unchanged. A pass means
meeting the original 2% risk and support checks and beating both CV
and OLS4 in conservative utility. Both datasets use the same shortlist,
features, score margins and one-exchange-per-recording policy.

The 24 fresh fits (14,400 updates) all use OpenDD: 21 site-held heads
and three pooled models, one per seed. The frozen Moving anchor has
its prior source-separated lineage. The new transition head does not
learn from Moving outcomes and does not change that anchor's setup.
The 24 MovingDroneCrowd++ recordings, from nine fitting-source groups,
are previously exposed development recordings, not Moving A or final
confirmation. The 72 views repeat those recordings at three seeds.

| Domain and policy | Exchange queries | Original risk passes | Risk and both gains |
| --- | ---: | ---: | ---: |
| OpenDD original | 0 | 112 | 77 |
| OpenDD geometric | 411 | 99 | 73 |
| OpenDD gain only | 272 | 105 | 67 |
| OpenDD transition | 116 | 112 | 77 |
| OpenDD gain at the same query times | 116 | 111 | 76 |
| OpenDD hindsight one-exchange capacity | Not online | 173 | 147 |
| Moving original | 0 | 18 | 2 |
| Moving geometric | 61 | 18 | 2 |
| Moving gain only | 56 | 18 | 1 |
| Moving zero-shot transition | 27 | 18 | 2 |
| Moving gain at the same query times | 27 | 18 | 2 |
| Moving hindsight one-exchange capacity | Not online | 18 | 2 |

OpenDD retains 1,457 edits in 606 views; Moving retains 54,748 in
72 views. The OpenDD head adds nine passes and destroys nine. Against
gain at the same query times, its conservative effect is **+0.00003 pp
[-0.00062, +0.00071]**. Moving makes 27 swaps and adds no pass. Against
gain at identical query times, the direct paired lower contrast is
**-0.00058 pp [-0.00112, -0.00009]**. Intervals resample source groups
3,000 times after averaging seeds; each domain keeps its registered
utility weighting and coordinate convention. Worst signed easy
degradation remains 11.28% in OpenDD and 30.25% in Moving.

## Is the target failure just a ranking problem?

OpenDD offers 108 rescue options among 2,642 shortlist entries; Moving
offers **zero among 57,285**. These are dependent candidate options,
not independent samples. To test whether a wider shortlist would
help, I ignored risk and bounded the gains of arbitrary path shapes:

| Target-domain relaxation | Views where OLS4 gain is impossible | Median OLS4 upper gain (CV-normalized pp) |
| --- | ---: | ---: |
| Frozen anchor, no exchange | 69 / 72 | -37.65 |
| Any recipient and direction, one exchange | 69 / 72 | -37.63 |
| Any path assignment across one whole query, same total peak | 69 / 72 | -36.89 |

The last bound uses the reverse triangle inequality. At one query
with total peak displacement `R`, common query weight `w`, and `T`
future steps, the utility increase is at most `2*T*w*R`, even if all
paths can change. For a single exchange the tighter bound also uses
the exact cost of removing the original edit and bounds the incoming
improvement by its radius. Neither computation is a deployable policy.
In particular, the bound does not cover more than one changed query
or a different initial forecast or displacement budget.

This is a transfer test of a replacement policy, not adaptation of the
trajectory predictor as in [Transferable GNN](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_Adaptive_Trajectory_Prediction_via_Transferable_GNN_CVPR_2022_paper.html).
The next useful intervention is to screen better **initial** causal
forecasts on fitting sources, or test changes at several query times
under the same recording budget, before investing in another
single-exchange head. Nothing here qualifies independent evaluation or
deployment. Moving A, DroneCrowd's 30 confirmation groups, Stage5C,
and SMC remain untouched.

```sh
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_cross_domain_pass_transition.py --replay
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/audit_m3w_single_exchange_envelope.py
```

Omit `--replay` to train or resume. Checkpoints and prediction arrays
stay local. [Protocol](protocol.md) | [Readout](readout.json) |
[Utility upper bound](relaxed_envelope.json) |
[Manuscript v154](../evidence_manuscript_v154/README.md)
