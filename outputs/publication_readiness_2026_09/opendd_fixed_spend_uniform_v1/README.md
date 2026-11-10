# OpenDD: uniform allocation at the original count and movement cap

I compared frozen OpenDD scores with uniformly chosen legal prefix
allocations on the same 606 exposed development views (202 recordings,
three seeds, seven locations). Both use the original 780 edits, the
same forecaster and prefix menu, chronological credits, movement caps,
the original 2% risk rule, and the additional complete-grid check.
To count as a pass, a plan also needs conservative gain over CV and
OLS4. No new outcomes were opened.

| Allocation | Full-grid dual passes / 606 |
| --- | ---: |
| Uniform legal allocation | 49.29 expected |
| Frozen score allocation | 55 |
| Hindsight, one exchange from frozen allocation | 175 |
| Hindsight, any legal allocation | 210 |

The frozen scores do not have a reliable advantage over uniform:
**-0.37 pp** in the equal-location mean (95% location bootstrap
**[-1.82, +1.07]**). The raw total is higher for the frozen scores
because locations have different numbers of views. Hindsight exceeds
uniform by **+14.04 pp [+5.39, +23.57]**. Among the 210 views with a
verified feasible allocation, the median probability of a pass under
uniform allocation is **18.18%**. A useful plan is not vanishingly
rare in a typical feasible view, but neither the hindsight result nor
this random comparator shows that the score learns to find one.

I enumerated 154 views and used 8,192 uniform pre-cap draws each on
56 larger views, keeping only plans below the original cap. The
remaining 396 views contribute exact zeros from the prior feasibility
analysis, including no-edit views. All sampled views retained legal
draws. The 0.057 expected-pass Monte Carlo standard error is separate
from the location interval. The interval averages seeds within
location and uses 3,000 paired location draws, holding the simulated
uniform probabilities fixed. This is a whole-recording development
diagnostic, not a deployable policy or a new independent evaluation.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_opendd_fixed_spend_uniform.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_fixed_spend_uniform.py tests/test_m3w_full_grid_count_capacity.py
```

[Protocol](protocol.md) | [Registration](registration.json) |
[Readout](readout.json).
