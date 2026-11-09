# OpenDD: how much can the pair decision improve at the same budget?

In the [matched OpenDD experiment](../opendd_slate_shared_menu_v1/README.md),
the pair score changed many decisions but did not reliably beat the
additive control. I asked a narrower question: if we already knew the
future, could different assignments of the same edit slots beat that
control and satisfy the original evaluation?

## Exact-slot hindsight comparison

I reused the frozen OLS4 and median-slope menu, 3,690 edits, and the exact
peak displacement for every edit slot in each window. Each slot must be
filled by a distinct agent and at least one agent stays on CV. I used the
original query weighting, missing-future debit, donor-only easy cutoffs,
and all three 2% risk checks. No data role, threshold, or forecast changed.

The solver maximizes conservative utility relative to the matched additive
policy. It must also achieve positive conservative utility over both CV
and OLS4. Per-agent gain and risk costs add within a recording; a binary
assignment imposes one candidate per slot and no duplicate recipient.
Every solver witness is re-scored with the original code. The relaxed
diagnostic still fills exactly the same slots but drops the risk and dual
reference conditions. The solver uses a 0.000001 percentage-point positive
margin for the two reference comparisons; the original evaluator is not
changed. No case hit the time limit or failed the scoring cross-check.

| Scope of hindsight search | Positive vs additive, without the risk and dual conditions | Passes risk and beats CV and OLS4 | Also beats matched additive |
| --- | ---: | ---: | ---: |
| Change all edit assignments | 333 / 439 | 51 / 439 | 26 / 439 |
| Keep single-edit assignments fixed | 194 / 439 | 38 / 439 | 9 / 439 |

The 439 estimable views are from the same 202 development recordings
evaluated at three training seeds, with 167 additional no-edit views
reported separately. The original learned joint policy had 34 dual passes.
In the stricter search, the nine positive views come from **five distinct
recordings at two locations**: four recordings at `rdb1` and one at `rdb6`.
The other estimable views are either optimal without a positive direct
advantage or blocked by absent easy support (112), a nonpositive reference
upper bound (221), or solver infeasibility under the combined constraints
(68). See the [site-level counts](readout.json).

## What this changes

The drop from 194 relaxed positive views to nine verified wins means the
limited gain cannot be attributed to the pair head alone. On this menu and
at the realized budgets, most potential improvements cannot simultaneously
beat both references, satisfy risk, and beat the matched additive policy.
Do not spend another training run widening only the pair head. The next
useful change is to the causal forecast menu or budget allocation, tested
against an equally capable additive control. This is a hindsight ceiling
on exposed OpenDD locations, not a deployed or independent result and not
a ceiling for a different forecast menu or budget. No reserved outcomes
were opened. The working paper and deployment are unchanged.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/audit_m3w_exact_slot_headroom.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_exact_slot_headroom.py tests/test_m3w_slate_shared_menu.py tests/test_m3w_pair_effect.py
```

The run uses local arm64 CPU and four Torch threads, with no workers or new
model fits. The program resumes from bound site-seed checkpoints and writes a
heartbeat after each checkpoint. A separate run in an empty cache reproduced
all 21 checkpoint hashes and the aggregate results. [Registration](registration.json).
