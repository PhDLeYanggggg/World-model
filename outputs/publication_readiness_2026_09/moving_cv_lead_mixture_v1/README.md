# Lead-Dependent Path Mixtures and the Edit Coverage Limit

I held the recipients fixed and changed the trajectories instead of
retuning the selector. Two equally sized networks combine the same
six causal motion paths. One uses constant weights across the forecast;
the other interpolates early, middle and late simplex weights. Both
use the same OLS4-relative conservative loss and CV-relative harm
penalty. Predictions and calibration decisions were frozen before
held scoring.

| Held fitting-source strategy | Edits | Original risk passes /72 | Undefined easy /72 | Worst signed known easy change |
|---|---:|---:|---:|---:|
| Prior frozen paths | 2,317 | 14 | 38 | +0.84% |
| Shared weights | 2,317 | 21 | 38 | +9.30% |
| Lead-dependent weights | 2,317 | 19 | 38 | +11.56% |
| Shared weights, matched peak displacement | 2,317 | 21 | 38 | +9.30% |
| Lead-dependent, matched peak displacement | 2,317 | 20 | 38 | +11.47% |
| Either calibration-selected fallback | 0 | 0 | 72 | Undefined edited support |

The learned mixtures pass more complete risk checks than the prior
paths, but both worsen the worst known easy change and lose
conservative utility to the prior paths. Lead-dependent minus shared
weights gives **-0.0060 pp [-0.0132, -0.0009]** in the direct
full-grid lower contrast and **-0.0030 pp [-0.0071, +0.0012]** when
peak displacement from CV is matched per recipient. The corresponding
raw neighbor lower contrast is -0.0122 pixels [-0.0244, -0.0004].
These are lower-bound contrasts, not estimates of complete unobserved
futures. Intervals resample source groups 3,000 times after averaging
seeds. **Neither model qualifies any of nine calibration cells.**

## More candidate paths versus more coverage

Adding either mixture or both to the original screened bank produces
the same three verified dual-reference wins out of 72 at exact
recording edit counts and cap eight per query. None of the 216
addback solver cells is unresolved. This is a hindsight search with
the original 2% rule, not a learned strategy.

I also relaxed the candidate library and dropped the risk constraint
**for a necessary upper bound only**. An edited row is granted a
perfect forecast on known steps and no distance from the comparator
on unknown steps. Maximizing these ideal benefits separately for CV
and OLS4 is optimistic because the same path need not achieve both
maxima. Untouched rows still use CV, so their deficit to OLS4 remains.
For each comparator and query, keep the largest eight ideal benefits,
then take the best benefits up to that recording's edit budget.

| Bound at the current recording edit counts | Views not ruled out /72 |
|---|---:|
| Perfect paths at the original recipients | 4 |
| Perfect paths at freely chosen recipients, cap eight | 8 |
| Perfect paths at freely chosen recipients, no query cap | 9 |

At cap eight, **40 views require more total edits** even in the
relaxation; **24 cannot reach the positive threshold at any allowed
count**. For the 48 views where the cap permits a positive bound, the
median lower bound on additional edits is 43.5. Without a query cap,
the median necessary count is 128.5, versus a current median of 5 on
the same 72 views. Those counts are necessary, not sufficient; they
do not imply feasibility under 2% risk.

The main bottleneck for this sparse policy is coverage of the
untouched CV trajectories, not simply the expressiveness of an
individual edit. I would stop iterating on proposal shape at the
same tiny budget. The next experiment should test broader causal
coverage under the unchanged original risk rule, with a separate
accuracy comparison against OLS4 and fallback retained.

The shared expert mix is related to
[feature-based forecast averaging](https://www.sciencedirect.com/science/article/pii/S0169207019300895).
The ablation here tests whether varying weights across one fixed
path horizon adds value at matched recipients and radii, not whether
forecast combinations are new.

## Reproduction

`fresh_run`: 18 native Torch fits, 18,000 updates, 48.33 s; 24 fitting
recordings in nine source groups and three seeds. Checkpoint every
100 updates, four CPU threads, zero loader subprocesses.
`cached_verified`: prior recipient freeze and donor predictions;
new fit/forecast replay exactly matches with zero additional updates.
All positive solver witnesses were separately rescored. Thirty-two
focused tests pass; the legacy suite was not rerun.

```sh
.venv-pytorch/bin/python scripts/run_m3w_cv_lead_mixture.py
.venv-pytorch/bin/python scripts/run_m3w_cv_lead_mixture.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_lead_mixture_capacity.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_minimum_edit_count.py
```

`not_run`: new A or DroneCrowd calibration readout, independent
confirmation, deployment, Stage5C and SMC. The original 2% rule is
unchanged and the 30 independent groups remain sealed. Native 8/12
annotation steps and image pixels are not seconds or physical units.
This is not a 3D or foundation model.

[Protocol](protocol.md) | [Readout](readout.json) |
[Capacity](hindsight_capacity.json) | [Coverage bound](minimum_edits.json) |
[Manuscript v129](../evidence_manuscript_v129/README.md)
