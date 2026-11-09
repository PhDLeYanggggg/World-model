# OpenDD: A Feasibility Shortlist Drops Most Useful Pair Choices

The previous ranker found more locally feasible slates but made
relative-motion forecasts worse. I trained a separate head for
the missing-step-debited pair objective, then measured how much
of that objective a fixed feasibility shortlist could recover.

## Experiment

I trained 21 relative-gain heads: seven held locations and three
seeds, with the same separate calibration site and five fitting
sites in each fold. A linear additive selector is the shared
reference at inference. When building labels for a donor site,
that reference is trained only on the other fitting locations;
its fitting cutoffs also exclude that donor. Neither calibration
nor held outcomes enter either fit. This reference differs from
the additive control in the previous ranking run, so the pair
effects in this report are compared only to controls with the
same new reference.

For each donor query I keep slates that beat CV and OLS4, pass
the original risk and reference-support checks, and do not reduce
conservative accuracy relative to the additive reference. The
head estimates observed relative gain and missing-grid debit
separately. There are 281 to 664 sampled conditional slates per
fit. All inputs use the eight past observations.

I froze the original feasibility scores and reproduced their
forecasts and scores exactly. The control uses the top-scoring
feasibility slate. The new arm chooses the best predicted
positive conservative pair gain within the top eight feasibility
candidates, reverting to the control's choice when none is
positive. A shortlist is not a risk certificate. Both arms use
the same 966 slots and exact per-query peaks in the forced
comparison. Optional release uses the unchanged calibration
rule and all three original 2% checks for both compared policies.

## Results

| Forced selector | Known ADE | Available FDE | Risk passes / 159 edited views | Pair gain vs common additive (native) |
|---|---:|---:|---:|---:|
| Feasibility only | 0.54786 | 1.25465 | 16 | −0.003424 |
| Conditional pair gain | 0.54772 | 1.25408 | 15 | −0.005929 |

The direct paired contrast of pair-gain ranking **against**
feasibility ranking is +0.002109 on co-observed pairs, with a
95% location interval [−0.001304, +0.005662]. On the full grid,
with missing relative positions debited, it is **−0.007682**
[−0.029089, +0.005354]. The largest negative full-grid site effect
is rdb5 (−0.039668). The direct conservative trajectory gain is
−0.4116 pp [−1.5609, +0.3567]. These direct effects are recomputed
from matched predictions over 53 recordings and four estimable
locations; they are not differences between separately computed
lower bounds. Intervals use 3,000 paired location draws.

The conditional arm changed 284 of 483 forced query-seed
decisions. It still falls short of CV on conservative trajectory
gain (−0.0517 pp [−0.1567, +0.0275]). Known ADE/FDE for CV are
0.54726 / 1.25329, its mean recording p95 is 1.38783 versus 1.39120
for the conditional arm, and the latter's hard gain is −0.031%.
For both arms, 49 edited views exceed 2% signed easy degradation
and 52 lack an edited-easy reference. Neither arm passed any
of its 483 nonempty score breakpoints on the calibration sites.
Both optional policies revert to CV; the empty match is not an
estimated interaction effect or a risk pass.

## Where the Opportunities Went

| Calibration-only, query-seed views | Count |
|---|---:|
| A qualifying pair-improving slate exists anywhere in the menu | 141 |
| At least one such slate is in the top eight | 42 |
| Selected by feasibility alone | 16 |
| Selected by conditional pair gain | 22 |

The local target requires better full-grid pair-relative error
and noninferior conservative trajectory accuracy versus the
same additive reference, in addition to the local feasibility
checks. The shortlist excludes 99 of the 141 opportunities
before the gain head can act. Within the 42 retained views,
the gain head finds 22. This conditional improvement does not
transfer to held locations: feasible pair capture changes from
21 to 20 of 150 opportunities. Local hindsight opportunities
do not certify an entire recording.

The next experiment should retrieve candidates with the **two
objectives jointly**, instead of letting the feasibility head
discard promising pair choices first. Compare at the same
shortlist size and slots, and model missing-pair debit separately.
Do not tune the risk limit or open independent confirmation to
compensate for a failed development rule.

## Reproduce

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_conditional_pair_gain.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_conditional_pair_gain.py score
.venv-pytorch/bin/python scripts/report_m3w_conditional_pair_gain.py
.venv-pytorch/bin/python scripts/audit_m3w_conditional_pair_shortlist.py
```

The arm64 fit and forecast freeze took 42.05 seconds; scoring
took 50.88 seconds. Maximum RSS stayed below 796 MB. Four Torch
threads, zero loader workers, atomic checkpoints every 50 steps
and a phase/PID heartbeat. Fifteen focused tests passed.
No CREATE job, Stage5C, SMC or reserved outcome access.
[Registration](registration.json), [metrics](summary.json),
[direct contrast](direct_contrast.json), [shortlist audit](shortlist_audit.json).
