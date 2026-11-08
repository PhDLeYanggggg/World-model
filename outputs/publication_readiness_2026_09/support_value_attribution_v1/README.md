# Gain, Harm, and Easy-Support Attribution

This is a hindsight and causal-control comparison on 202 distinct
OpenDD development recordings (606 recording-seed views). All arms use
the same nominated CV-to-OLS4 actions, eight observed and 12 forecast
steps, and original CV-relative 2% recording rule. Independent DroneCrowd
confirmation remains sealed. No model was fitted in this audit.

The frozen online motion gain-only policy makes 1,457 edits and passes
risk with positive gain against both references in 61 views. Of its 334
views with undefined easy risk, 172 have no edits, 160 have easy references
in the recording but none selected, and two have no easy reference at all.
Another 185 have a defined risk that fails. Among the 14,361 legal nonzero
query actions, 5,974 have positive CV gain and 1,954 also carry easy
reference mass. Among the 1,402 chosen query actions, 633 have positive
gain and 194 also carry easy mass. Counts include repeated seeds and
both legal nonzero counts where available, not independent agents.

**Exactly matched totals.** Every arm below makes the same number of edits
as the original gain-only policy in each recording and seed, respecting
the same token constraints. Each local substitution changes only the
listed predicted moments. Allocation sees all query scores in a recording;
these are retrospective diagnostics, not online results. Oracles use
future outcomes and never supply deployable features.

| Retrospective allocation | Risk pass and beat both | Direct gain vs matched motion (pp, 95% location interval) |
| --- | ---: | ---: |
| Learned motion gain and risk | 60/606 | reference |
| True easy support bonus only | 76/606 | -0.085 [-0.198, -0.009] |
| True conservative gain only | 129/606 | +0.099 [+0.049, +0.143] |
| True all/easy excess and net easy change only | 146/606 | +0.052 [-0.044, +0.123] |
| All true local score terms | 244/606 | +0.055 [-0.119, +0.179] |
| Exact global constrained hindsight | 304/606 | not compared by this contrast |

The global solver fixes those same realized recording totals, verifies
each witness with the original recording evaluator, and finds 304
positive cases. Of the rest, 172 have zero edits, 11 have no legal easy
option, 74 are risk-infeasible, 44 have a nonpositive upper bound, and one
is unresolved. The local substitutions identify limitations of this
fixed score and action menu, not an information-theoretic ceiling.
Knowing easy support alone trades away forecast value. Accurate gain
and harm estimates address different parts of the failure.

**Causal control.** Estimate easy support and signed risk from errors
within the observed window (three prefix splits); positive CV error
quartiles come from other locations. The prefix-risk rule reduces
undefined views from 320 to 276 versus prefix gain alone, but reduces
joint passes from 56 to 50. Direct online gain difference is -0.007 pp
[-0.127, +0.128]. At identical realized counts it gives -0.101 pp
[-0.221, -0.013] versus the matched learned motion rule. Simply moving
edits toward likely easy cases does not recover the missing utility.

`fresh_run`: original risk readout, fixed-total certificates and causal
prefix-risk scores. `cached_verified`: frozen model and previous online
controls (exact replay). No deployment change, Stage5C, SMC or independent
confirmation. [Metrics](analysis.json) | [Registration](registration.json).

```sh
.venv-pytorch/bin/python scripts/audit_m3w_support_value_attribution.py
```
