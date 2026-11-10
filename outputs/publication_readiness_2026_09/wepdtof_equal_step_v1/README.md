# WEPDTOF: equal steps do not change the original conclusion

I used the 15 previously scored clips from 13 sources and three
seeds. The archived proposals already match edit count and peak
departure from CV. For each joint-versus-control comparison I
held recipients and directions fixed, then matched the departure
from CV at each of the twelve steps. All 198 edited query-seed
views and 366 edits per arm remain; no query collapses. For joint
versus independent, the common step profile retains the peak and
97.34% of integrated movement.

| Joint minus independent | Observed gain (pp) | Full-grid envelope (pp) |
| --- | ---: | ---: |
| Original | -0.01371 | [-0.04501, +0.01760] |
| Equal steps | -0.01176 | [-0.04181, +0.01829] |

The same-source change in observed gain is **+0.00195 pp**,
95% bootstrap **[-0.00173, +0.00546]**, on ten estimable sources.
The other three sources remain undefined for this contrast. The
relative-position contrast under equal steps is +0.00608 pixels
on common labeled pair-steps, with a full-grid envelope of
[-0.01248, +0.03643]. The envelope and the bootstrap answer
different questions: unknown targets versus source composition.

The joint and no-pair-penalty outputs are **exactly identical in all 36
edited views**. WEPDTOF's control removes the entire pair penalty,
not just a cross term. The two arms make the same choices on this
sample; this does not establish the penalty's effect in other settings.
[Frozen score diagnosis](../wepdtof_penalty_breakpoint_v1/README.md).
Same-family comparisons, movement
retention, and the uniformly smaller-peak control are in the
[readout](readout.json). The smaller-peak control reproduces
the original to rounding precision because the common peak
is unchanged.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_wepdtof_equal_steps.py freeze
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_wepdtof_equal_steps.py score
```

The first command verifies the [geometry freeze](geometry_freeze.json)
without loading targets. The second reproduces original utility on
the previously scored labels before evaluating the new variants.
This is retrospective sensitivity, not a new independent test,
new fit or revised release rule. The original 2% and zero joint
dual passes are unchanged. [Protocol](protocol.md).
