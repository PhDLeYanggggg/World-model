# WEPDTOF: why the pair penalty changes no frozen choice

The WEPDTOF control removes the **entire pair penalty**. Moving
removes only the cross term of its squared-relative score. Songdo
compares separately trained donor heads. The frozen WEPDTOF
objective is `U - 0.10 P`, where `U` is the minimum summed
predicted gain against the two motion references minus positive
summed risk excess. `P` is the mean norm of the difference in the
two edits' departures from CV, weighted by `exp(-d/s)`. Here
`d` is last-observed pair separation and `s` is twelve times the
query's median observed step length, with the original positive floor.

I replayed the donor heads and enumerated **11,820 legal choices**
on 198 edited query-seed views. The pair-penalty and no-pair-penalty
paths both reproduce exactly. Across all 36 edited clip-seed
views the two arms are identical.

| Reason at the fixed weight of 0.10 | Query-seed views |
| --- | ---: |
| One edited recipient, so the pair penalty is zero | 30 |
| Two edits, but no alternative has lower penalty | 10 |
| Two edits, and first possible change is above 0.10 | 158 |

For an unpenalized winner `a`, a lower-penalty competitor `b`
can take over only when its penalty advantage outweighs its
utility deficit. The first possible crossover is
**0.1101297**. This diagnosis does not recommend that value,
assess utility at a new weight, or change the original release.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_wepdtof_penalty_breakpoints.py
```

No target labels were opened and no model was fitted. The previous
equal-step utility values remain unchanged. [Protocol](protocol.md) |
[Verified forecast-only readout](readout.json).
