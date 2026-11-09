# AV2: Reassigning Two Edits Without Spending More Displacement

The [steering study](../av2_steered_reference_v1/README.md) showed
that turning two fixed edits could not repair most references. I
now keep both complete radial profiles and ask whether assigning
them to other actors changes what is possible.

## Same-budget comparison

An edit is a twelve-step CV-relative offset sequence. A scene keeps
its original two sequences, including every step's radius, and
assigns them to two distinct actors. The finite bank rotates each
sequence from its original actor's observed heading to the new
actor's observed heading, without rescaling it. Keeping the original
assignment reproduces the original prediction bit-for-bit.

For the hindsight capacity calculation I enumerate all ordered
assignments. At a fixed assignment, the [closed-form direction
solution](../av2_steered_reference_v1/README.md#exact-relaxed-direction-ceiling)
minimizes error on each known step and OLS4 disagreement on each
missing step. It can be unrestricted or bounded to 45 degrees around
the transported path. Additive gain and risk terms permit exhaustive
comparison without storing all full-scene trajectories. Every passing
assignment replays through the original scalar scorer. The risk
requirements, label support, and both motion floors are unchanged.
These ceilings do not require physically plausible directions or
demonstrate any inference policy.

The causal rules replay the same paths against observed prefixes.
I compare keeping the two actors, ranking all assignments by the
sum of single-actor scores, requiring a predicted easy actor when
available, and adding a fixed pair term to that same support rule.
The pair term uses past relative displacement error, not predicted
collision or physical safety. The last two arms share the same bank
and predicted support requirement. These forced predictions were
frozen before scoring; no release thresholds or models were fit.

## Results

| Hindsight menu | Eligible views / 1,020 | Fully labeled / 465 |
|---|---:|---:|
| Original actors and profile assignment, free direction | 176 | 116 |
| Same actors, profiles may swap, free direction | 190 | 121 |
| Any two actors, transported directions fixed | 317 | 187 |
| Any two actors, ±45-degree per-step directions | 398 | 231 |
| Any two actors, free directions | 505 | 290 |

Changing recipients rather than swapping the two original profiles
adds **315** feasible views under the same free-direction relaxation.
The city-balanced increment is **30.72 pp [24.84, 37.20]**. True edited
easy support is possible in 855 views after reassignment, versus
382 at the original recipients. Unrestricted directions improve the
ceiling further, but are not needed to establish that the fixed
recipients leave substantial capacity unused.

| Forced forecast | Standalone eligible | Risk pass | Edited easy defined | Conservative gain vs original (pp) |
|---|---:|---:|---:|---:|
| Original | 63 | 109 | 382 | 0.000 |
| History, fixed actors | 59 | 104 | 382 | -1.103 |
| History, additive reassignment | 54 | 106 | 314 | -2.896 |
| History, predicted easy support | 66 | 123 | 529 | -4.316 |
| History, same support plus pair term | 72 | 132 | 536 | -3.492 |

The pair term recovers 18 passes and loses 12 relative to the
support-matched additive control. Its eligibility change is
**+0.690 pp [-0.016, +1.376]** and its direct pair-error gain is
**-0.000239 [-0.001038, +0.000755]** in native units. Against the
original trajectory it loses **3.492 pp [2.262, 4.777]** in
conservative utility; available ADE/FDE worsen from 0.2480/0.5518
to 0.2550/0.5696. Its worst view p95 ADE rises from 5.929 to 7.390.
This is not evidence of a usable joint improvement.

Of 317 views with a qualifying assignment in the exact transported
bank, the pair rule qualifies 72. Its first failing check is
undefined easy support in 112, defined but failed risk in 117,
and a motion floor in 16. That points to a better training target:
rank assignments against actual support and risk, while retaining
the no-change forecast and using the same bank for additive and
joint heads. I would not add more geometric variants first.

The 1,020 views cover 340 existing development scenarios, six cities
and three seeds. Intervals use 3,000 paired city draws after averaging
seeds within each scenario. The coordinates are native units on the
unchanged 8/12 grid. No reserved outcomes, confirmation or deployment.

## Reproduction

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_recipient_reassignment.py freeze
.venv-pytorch/bin/python scripts/run_m3w_av2_recipient_reassignment.py score
.venv-pytorch/bin/python scripts/run_m3w_av2_recipient_reassignment.py replay
.venv-pytorch/bin/python scripts/audit_m3w_av2_recipient_reassignment.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2*.py \
    tests/test_m3w_interaction_frontier.py tests/test_m3w_slate_relative_gain.py
```

Native arm64, four Torch threads, no data workers. Freeze took
3.41 s and scoring 8.01 s at 0.53 GB maximum RSS. The focused
suite passed 82 tests. Per-fold receipts allow restart; replay
and attribution read scalar scores. No new CREATE job, neural
training, manuscript change, Stage5C or SMC.

[Registration](registration.json) · [Freeze](freeze.json) ·
[Results](summary.json) · [Attribution](attribution.json)
