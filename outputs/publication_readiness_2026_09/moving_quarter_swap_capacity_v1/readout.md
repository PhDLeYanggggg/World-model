# Sparse Exchanges At Fixed Count And Amplitude

**Finding:** Sparse replacements can recover utility and reference support
in a subset of the development recordings. The earlier learned full
reranking did not realize that opportunity.

## Experiment

Start from the original quarter roster (15/102 passes). Replace a selected
agent with a different agent in the same query, retaining the count and
the quarter-amplitude blend. Allow one replacement per query and up to
1, 4 or 16 per recording. Optimize the paired completion-lower utility
using outcomes, under the original recording rule: 2% total, easy and
signed-easy limits and a defined easy reference. Every witness is checked
on its reconstructed paths with the original scoring implementation.

The experiment reuses 34 exposed recordings, 14 source groups and
three frozen forecaster seeds. The 102 views are not independent scenes.
The 4,875,493 candidate swaps are repeated candidate occurrences. An
exchange keeps all other forecasts intact; unknown outcomes pay the
actual path disagreement for both the insertion and deletion.

## Results

| Max Exchanges | Valid Witnesses From That Solve | Newly Passing | New Passes With Originally Undefined Easy | Median Paired Lower Gain For New Passes (pp) | Unresolved |
|---|---:|---:|---:|---:|---:|
| 1 (exact enumeration) | 41 / 102 | 26 | 12 | +0.02451 | 0 |
| 4 (primary) | 43 / 102 | 32 | 14 | +0.11522 | 31 |
| 16 | 50 / 102 | 39 | 14 | +0.41066 | 31 |

All returned newly passing witnesses have strictly positive paired
completion-lower gain. Single exchanges find new repairs in 11 of 14
source groups. The cap-4 and cap-16 solves prove infeasibility for
28 and 21 views respectively, but the unresolved views are not failures
of feasibility. A valid smaller-budget witness is also valid at any
larger cap. The cumulative union therefore certifies **at least 41,
52 and 59** passing views at caps 1, 4 and 16, including the original
15. These union counts are a descriptive recombination of validated
witness receipts; the per-solve counts and statuses above remain the
registered readout.

## Interpretation

The dense learned reranking passed only 7/102 at the same count. In
contrast, even one label-aware replacement finds 26 repairs. This
supports testing a sparse learned replacement rule rather than another
full rerank. It does not prove that these replacements are identifiable
from history: the score must anticipate support, harm and utility in
new sources. And 18 views started with zero switches, so this matched
action set cannot add a reference there at all. It is a useful control,
not a complete intervention policy for the entire roster.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/run_m3w_quarter_swap_capacity.py score
.venv-pytorch/bin/python scripts/run_m3w_quarter_swap_capacity.py replay
```

Registered in `ef14fbcb`. Local scoring took 469 seconds and saved
one private receipt per recording/seed. Replay reconstructed and scored
the saved witnesses in 9.7 seconds and reproduced the same
[analysis](analysis.json). Seventeen targeted tests passed. The
cumulative counts above are the numbers of distinct private receipts
with a non-null witness at any cap up to the displayed cap; no returned
witness failed the original-scorer postcheck.

No new model was trained or deployed. The study uses 8 observed and
12 future annotation steps in image coordinates, not meters or seconds.
No independent outcomes were opened. The original rules, Stage5C and
SMC statuses are unchanged.
