# Accuracy repair at fixed predicted risk

I tried to recover the accuracy lost by the lower-coverage joint policy
without raising its predicted risk or changing which queries were eligible
for edits. I used nine exposed development source groups, 24 recordings,
three role rotations and three seeds. I trained two PyTorch gain heads for
25,200 updates in total. The risk model, analytic rays, original 2% rule,
and 21,242 edits matched query by query were unchanged.

| Choice rule | Conservative change vs. incumbent, pp [95% CI] | Original risk passes | Worst signed easy degradation |
| --- | ---: | ---: | ---: |
| Frozen joint incumbent | Reference | 20/72 | +12.48% |
| Geometry only | +0.0002 [-0.0001, +0.0006] | 20/72 | +12.48% |
| Direct gain regression | -0.0076 [-0.0197, -0.00004] | 20/72 | +19.92% |
| Observed gain + expected missing debit | -0.0022 [-0.0056, -0.00007] | 20/72 | +19.92% |
| Outcome-aware rounded search (diagnostic) | -0.0128 [-0.0502, +0.0118] | 20/72 | +12.48% |

The intervals use 3,000 paired source-group resamples after averaging seeds.
The direct and factorized models changed only 41 and 26 active query-seed
views, respectively. The factorized model did not reliably beat direct
regression (+0.0009 pp, 95% CI [-0.0005, +0.0032]). Its known-step result
improved by 0.0010 pp versus the incumbent, but the paired conservative
result did not. Keeping predicted risk fixed did not keep actual easy-case
risk fixed. All compared arms still had 29 views with undefined easy risk.

The outcome-aware search maximizes the conservative score against OLS4,
not the paired bound against the incumbent. Against OLS4 it gained 0.0073 pp
over the incumbent's score but remained below OLS4. The two conservative
contrasts use different missing-step debits and need not agree in sign.
This search is neither a deployable policy nor a proof of integer optimality.

I then solved the outcome-aware fractional relaxation on all 1,627 active
query-seed views. At most 286 had more than numerical tolerance of
OLS4-relative headroom. Even allowing fractional actions, the upper
increment averaged only +0.0577 pp across the nine sources and three seeds;
the largest source-seed increment was +1.1955 pp. The same OLS4-relative
score starts at -55.5921 pp, so exact discrete search or a better ranker
cannot close that conservative gap inside these frozen constraints.
This bound is conditional on the existing candidate bank, edit counts and
predicted budgets; it is not a bound for a different policy. The next
experiment should change the available paths or predicted allocation
constraints without changing the actual 2% test.

**Decision:** 0/9 calibration cells qualified. I keep the existing fallback
and do not open independent confirmation. DroneCrowd keeps 15 calibration
groups and 30 separate confirmation groups under the original rule. This
study did not use either group; all findings above are from already exposed
development sources. No metric, seconds-level, physical-safety, or deployment
claim follows from these local-pixel trajectory experiments.

**Provenance:** Training and matched readout are `fresh_run`. Incumbent
choices and risk/shape checkpoints are `cached_verified` against their
frozen identities and hashes. DroneCrowd confirmation is `not_run`.
The local runner saves optimizer, model and RNG every 100 updates and
resumes when invoked again. `--replay` requires completed checkpoints.

[Protocol](protocol.md) | [Registration](registration.json) |
[Freeze](freeze.json) | [Readout](readout.json) |
[Fractional capacity](capacity_readout.json)
