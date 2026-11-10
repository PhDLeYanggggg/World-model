# OpenDD: exact full-grid headroom under the original movement budget

I optimized the frozen OpenDD prefixes with each recording's original
edit count, chronological credits, and cap on the sum of peak departures
from CV. Only the distribution of edits can change; the forecasts cannot.
I kept the original 2% and two-reference tests, then added the complete-grid
conditions: two additive risk excesses and at least one complete easy edit.
Every feasible witness is checked again using the original scorer and the
direct full-grid calculation.

| Decision at the fixed budget | Original dual passes / 606 | Also satisfies full-grid risk |
| --- | ---: | ---: |
| Frozen score allocation, outcomes not used for choice | 60 | 55 |
| Hindsight, at most one exchange from that allocation | 181 | 175 |
| Hindsight, any legal prefix allocation | 212 | 210 |

The original unrestricted-prefix search verified 210 and left two
numeric cases unresolved. Exhaustively searching the legal allocations
in those two cases finds a witness in each after seven plans checked
in total. Both also pass the stricter full-grid test. Among the 210
previously verified cases, 207 original witnesses survive directly,
one has a different full-grid witness, and two have no full-grid plan.
There are no unresolved views in either result after these checks.

The equal-location difference from frozen scores is **+14.41 pp**
(95% location bootstrap **[+5.37, +24.37]**). Relative to one exchange,
unrestricted reallocation adds **35** raw passes, but the paired
location interval includes zero (**+2.41 pp [0.00, +5.79]**).
The 606 views repeat 202 recordings across three seeds; the bootstrap
resamples seven locations 3,000 times after averaging seeds. These are
hindsight feasibility results on exposed development data, not a
past-only policy or evidence of external generalization.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_opendd_full_grid_count_motion.py
```

`fresh_run`: full-grid coefficients, constrained search and exact
enumeration; 21 location-seed cells took about seven seconds locally.
Checkpoints bind each view to the original budget and store witnesses
only in ignored local data. `cached_verified`: previous counts, paths
and positive witnesses. `not_run`: training, CREATE, new external
outcomes, and any release change.

[Protocol](protocol.md) | [Registration](registration.json) | [Readout](readout.json).
