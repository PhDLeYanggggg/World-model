# Can Recording-Level Reallocation Recover the Missing Gains?

I fixed one screened candidate trajectory per agent and kept the risk
and preference models unchanged. The comparisons vary where the edit
budget is spent. The offline rows below use exactly the same count in
each recording and a cap of eight edits per query. They may compare
scores from later queries and are **not** online controllers. The two
streaming rows use only the current and earlier queries, and are **not**
count-matched with one another.

| Strategy | Edits | Original risk passes /72 | Undefined easy /72 | Worst known signed easy change |
|---|---:|---:|---:|---:|
| Frozen local allocation | 2,317 | 14 | 38 | +0.84% |
| Offline gain-ranked allocation | 2,317 | 4 | 42 | +77.26% |
| Offline risk-priced allocation | 2,317 | 9 | 41 | +59.38% |
| Streaming without carry | 916 | 6 | 48 | +60.22% |
| Streaming with carry | 1,018 | 6 | 46 | +60.22% |
| Calibration-selected streaming fallback | 0 | 0 | 72 | Undefined edited support |

The two offline strategies move 681 and 670 slots between queries,
respectively. Risk pricing restores five net passes versus gain-only,
but its direct conservative utility contrast is -0.0181 pp
[-0.0308, -0.0071]. Compared with frozen local allocation, risk pricing
gains three risk passes and loses eight; its conservative utility
contrast is -0.0188 pp [-0.0735, +0.0248]. For neighbor displacement,
its observed paired gain is +0.0283 pixels [+0.0016, +0.0672], but the
missing-step lower contrast is -0.0716 pixels [-0.1702, -0.0143].
Visible pairs alone therefore give a different impression from the
full-grid contrast. These are differences on the common roster and use
3,000 source-group bootstrap draws after averaging seeds.

Streaming spends 219 previously unused credits. It recovers easy
reference support in two views, but gains one full pass and loses one.
The conservative utility contrast versus the no-carry streamer is
+0.00015 pp [-0.00166, +0.00204]. **Calibration qualifies 0/9 cells.**
No setting clears more than two calibration recordings in any cell.
Streaming fallback is unchanged.

## Did the policy fail to find options that existed?

An outcome-aware solver was allowed to choose recipients only from the
same fixed screened paths. A witness must use the exact specified
counts, pass the unchanged risk rule, and deliver positive full-grid
lower gain against the required comparator(s). Every positive witness
was independently rescored.

| Required gain | Exact per-query counts | Exact total per recording, timing free |
|---|---:|---:|
| Over CV | 26/72 | 39/72 |
| Over both CV and OLS4 | 0/72 | 3/72 |

Fifteen views have no local edit budget; sixteen more have a positive
budget but no easy reference in the entire screened pool. The solver
does not turn these 31 views into passes. In the other 41 views,
reallocation makes more solutions possible versus CV. Against both
comparators, 36 of those views have an upper bound at or below the
required positive margin and two
are infeasible, leaving three verified wins and no unresolved solver
cells. Thus there is a ranking gap versus CV, but reallocating these
particular screened paths is not enough to catch OLS4. The next
priority is to improve the proposals and their CV-relative harm before
adding more scheduling logic.

The online constraint is inspired by work on
[resource allocation with unknown horizon](https://arxiv.org/abs/2206.13606).
This is a fixed credit-carry control, not a regret-guaranteed method:
its credits are derived from the current query, and the risk rule
also needs a supported reference at the recording level.

## Reproduction

`fresh_run`: 24 recordings across nine fitting sources and three seeds;
51.97 s for selection and scoring on native arm64 CPU, four threads.
The first engineering attempt was interrupted before held scoring due
to a byte-integer overflow when summing recording quotas. The fixed run
passed the count assertions and a regression covering 240 credits.
The neighbor readout and 288 hindsight solver cells are separate fresh
analyses. `cached_verified`: parent heads, source-excluded forecasts,
local matched choices, frozen score replay and witness replay. Twenty-seven
focused tests pass; the legacy suite was not rerun.

```sh
.venv-pytorch/bin/python scripts/run_m3w_cv_record_allocation.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_record_allocation_geometry.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_record_allocation_capacity.py
```

Reruns rescore frozen choices and verify saved solver witnesses. The
main run and capacity audit are separate, with PID and heartbeat
records in the ignored run directory. No new model training or CREATE
job was required. `not_run`: new A readout, new DroneCrowd calibration,
independent confirmation, deployment, Stage5C and SMC. The 30
confirmation groups remain sealed; the 2% rule is unchanged. Native
8/12 annotation steps and image pixels are not seconds or physical
units, and this is not 3D or a foundation model.

[Protocol](protocol.md) | [Readout](readout.json) | [Geometry](geometry.json) |
[Hindsight](hindsight_capacity.json) | [Manuscript v128](../evidence_manuscript_v128/README.md)
