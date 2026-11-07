# Fixed-Count Support-Aware Proposal Retrieval

**Decision:** Do not promote the support-ranked policy. It retrieves more
locally feasible candidates but does not reliably recover the primary
recording coverage endpoint, loses accuracy at matched count, and still
fails the original recording gate.

The [frozen protocol](protocol.md) compares gain and support ordering of the
same replacement shortlist, leaving the forecast bank and gate fixed.
There are 42 fresh ridge fits over 14 held-out development source groups
and three forecaster seeds. The 102 views are 34 exposed recordings crossed
with three seeds, not independent recordings. All matched arms propose
9,675 agent switches in the same 4,839 seed-query sets.

| Proposal ranking | Query sets with easy reference | Views with easy reference | Views with a feasible single-query witness | Views where gate kept a feasible set | Emitted agent switches | Original recording passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original joint | 1,849 | 78 | 57 | 9 | 90 | 3/102 |
| Gain | 1,984 | 79 | 55 | 18 | 190 | 3/102 |
| Easy support | 2,415 | 80 | 67 | 18 | 222 | 5/102 |

All intervals use 3,000 paired source draws after averaging seeds. The
registered primary endpoint (view-level easy support) changes by
**+1.90 pp [-1.43, +7.14]** for support versus gain. Three views gain
support and two lose it. Query-level easy support increases by
**+7.79 pp [+4.64, +10.99]**. Views with a feasible single-query witness
increase by **+16.43 pp [+4.76, +29.76]**: 12 new views, no losses, in the
unweighted counts. This is label-aware potential, not a learned policy
pass. The gated pass difference is +2.38 pp [-1.59, +7.94].

## Accuracy And Gate Retention

At exactly matched counts, the support ranker's conservative utility is
**-0.477 pp [-1.235, -0.046]** versus gain ranking, relative to the common
CV denominator. Its observed relative-path ADE is **0.093 pixels worse**
[0.041, 0.152]; the completion-effect bounds span zero
[-1.225, +1.101] pixels in reduction units. After applying the fixed gate,
the support ranker has conservative gain **+0.00020 pp**
[-0.00060, +0.00122] versus CV and **-9.824 pp** [-21.999, -0.555]
versus OLS4. No meaningful gain over the accuracy comparator is established.

The [posthoc gate audit](gate_retention.json) checks the frozen decisions:
the gate keeps **40/755** locally feasible support-ranked queries versus
**31/685** gain-ranked queries. Each arm retains at least one feasible
set in 18 views, with eight gained and eight lost. The support ranker also
emits 44 nonpositive-lower-gain sets, versus 27 for gain ranking, and leaves
easy risk undefined in 75/102 views. Local feasibility does not imply that
all emitted sets in a recording pass jointly.

Retrieval and operation require different evidence. A useful next model
should learn the joint event of *positive conservative gain, nonpositive
risk excess and valid support* on the candidate sets it actually sees.
Simply increasing the chance of easy reference does not optimize that
event. This is distinct from [selective subgroup MSE](https://proceedings.mlr.press/v162/shah22a.html)
and does not make expert selection itself novel relative to
[multi-expert deferral for regression](https://proceedings.mlr.press/v235/mao24d.html).

## Reproduction

```sh
.venv-pytorch/bin/python scripts/run_m3w_native_support_retrieval.py fit
.venv-pytorch/bin/python scripts/run_m3w_native_support_retrieval.py score
.venv-pytorch/bin/python scripts/run_m3w_native_support_retrieval.py replay
.venv-pytorch/bin/python scripts/audit_m3w_native_support_retrieval.py
```

[Metrics](analysis.json), [fit receipt](fit_execution.json),
[score receipt](score_execution.json), [replay receipt](replay_execution.json).
No independent input or outcome access, identity certification, deployment
change, Stage5C execution or SMC. Units are dataset pixels and annotation steps.
